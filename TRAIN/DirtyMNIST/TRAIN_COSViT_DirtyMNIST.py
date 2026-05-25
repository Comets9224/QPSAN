# coding=utf-8
"""
CosViT训练脚本 - DirtyMNIST 二分类

使用余弦注意力VisionTransformer (CosViT) 训练，与ViT对照组保持完全相同的训练设置以公平对比。
DirtyMNIST: 28×28灰度图, 二分类 (FastMNIST vs NoisyMNIST)
"""
from __future__ import absolute_import, division, print_function

import sys
import os

# 添加项目根目录到sys.path，确保能从子目录运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import argparse
import random
import numpy as np
import hashlib
import json

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from models.COSVIT import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.early_stopping import EarlyStopping
from utils.dist_util import get_world_size
from data.DirtyMNIST.dataloader import load_dirty_mnist_binary
from utils.training_plotter import TrainingPlotter, plot_average_curves
from utils.path_utils import (
    get_output_paths,
    create_output_paths,
    get_checkpoint_path,
    get_tensorboard_log_path,
    get_results_path,
    get_plot_path
)

# 分类指标计算
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """
    分类指标计算器

    用于二分类任务的指标计算
    """

    @staticmethod
    def compute_metrics(labels, preds, probs=None):
        """
        计算分类指标

        Args:
            labels: 真实标签，shape: [N], 值∈{0, 1}
            preds: 预测标签，shape: [N], 值∈{0, 1}
            probs: 预测概率（用于AUC-ROC），shape: [N, 2] 或 [N]

        Returns:
            dict: 包含各种指标的字典
        """
        # 基础指标
        accuracy = (preds == labels).mean()
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

        # AUC-ROC（如果提供了概率）
        if probs is not None:
            try:
                # 如果probs是[N, 2]，取第1列（正类概率）
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs_positive = probs[:, 1]
                else:
                    probs_positive = probs

                auc = roc_auc_score(labels, probs_positive)
                metrics['auc'] = float(auc)
            except Exception:
                # 如果计算失败（如只有一个类别），设置默认值
                metrics['auc'] = 0.0

        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()  # 转为列表方便JSON序列化

        return metrics

    @staticmethod
    def format_metrics(metrics):
        """
        格式化输出指标

        Args:
            metrics: 指标字典

        Returns:
            str: 格式化的字符串
        """
        lines = [
            f"  Accuracy: {metrics['accuracy']:.4f}",
            f"  Precision: {metrics['precision']:.4f}",
            f"  Recall: {metrics['recall']:.4f}",
            f"  F1 Score: {metrics['f1']:.4f}",
        ]

        if 'auc' in metrics:
            lines.append(f"  AUC-ROC: {metrics['auc']:.4f}")

        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            lines.append("\n  Confusion Matrix:")
            lines.append("         Predicted")
            lines.append("            0    1")
            lines.append(f"      0  [{cm[0][0]:3d}  {cm[0][1]:3d}]  Actual")
            lines.append(f"      1  [{cm[1][0]:3d}  {cm[1][1]:3d}]")

        return "\n".join(lines)

    @staticmethod
    def format_confusion_matrix(cm):
        """
        格式化混淆矩阵（用于日志输出）

        Args:
            cm: 混淆矩阵列表 [[TN, FP], [FN, TP]]

        Returns:
            str: 格式化的字符串
        """
        lines = [
            "           Predicted",
            "              0    1",
            f"        0  [{cm[0][0]:>4}   {cm[0][1]:>4}]  Actual",
            f"        1  [{cm[1][0]:>4}   {cm[1][1]:>4}]"
        ]
        return "\n".join(lines)


class MetricsSummary:
    """
    多次运行指标汇总器

    功能：
    1. 收集每次运行的最佳指标
    2. 计算平均值、标准差、中位数、最佳值
    3. 格式化输出
    """

    def __init__(self):
        self.runs = []  # 存储每次运行的最佳指标

    def add_run(self, best_metrics):
        """
        添加一次运行的最佳指标

        Args:
            best_metrics: dict, 包含 accuracy, precision, recall, f1, auc, confusion_matrix
        """
        self.runs.append(best_metrics)

    def compute_summary(self):
        """
        计算汇总统计

        Returns:
            dict: 包含各项指标的统计信息
        """
        if not self.runs:
            return None

        # 提取各项指标
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        summary = {
            'num_runs': len(self.runs),
            'means': {},
            'stds': {},
            'bests': {},
            'medians': {},
            'best_run_idx': {},  # 最佳运行索引（用于混淆矩阵）
        }

        # 计算各项指标的统计量
        for metric in metrics_names:
            values = [run[metric] for run in self.runs]

            summary['means'][metric] = float(np.mean(values))
            summary['stds'][metric] = float(np.std(values))
            summary['bests'][metric] = float(np.max(values))
            summary['medians'][metric] = float(np.median(values))

            # 找出最佳运行（基于f1）
            if metric == 'f1':
                best_idx = int(np.argmax(values))
                summary['best_run_idx'][metric] = best_idx

        # 保存最佳运行的混淆矩阵
        best_f1_idx = summary['best_run_idx']['f1']
        summary['best_confusion_matrix'] = self.runs[best_f1_idx]['confusion_matrix']

        return summary

    def format_summary(self, summary):
        """
        格式化输出汇总结果

        Args:
            summary: compute_summary() 返回的汇总字典

        Returns:
            str: 格式化的字符串
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"{summary['num_runs']} Runs Summary")
        lines.append("=" * 70)

        # 表头
        lines.append(f"{'Metric':<12} {'Mean ± Std':<20} {'Best':<10} {'Median':<10}")
        lines.append("-" * 70)

        # 各项指标
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            mean_val = summary['means'][metric]
            std_val = summary['stds'][metric]
            best_val = summary['bests'][metric]
            median_val = summary['medians'][metric]

            names = {'accuracy': 'Accuracy', 'precision': 'Precision',
                    'recall': 'Recall', 'f1': 'F1 Score', 'auc': 'AUC-ROC'}

            mean_str = f"{mean_val:.4f} ± {std_val:.4f}"
            lines.append(f"{names[metric]:<12} {mean_str:<20} {best_val:.4f}    {median_val:.4f}")

        lines.append("=" * 70)

        # 混淆矩阵（最佳运行的）
        cm = summary['best_confusion_matrix']
        lines.append("\nBest Run Confusion Matrix:")
        lines.append("         Predicted")
        lines.append("            0      1")
        lines.append(f"      0  [{cm[0][0]:>4}   {cm[0][1]:>4}]  Actual")
        lines.append(f"      1  [{cm[1][0]:>4}   {cm[1][1]:>4}]")

        return "\n".join(lines)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, run_idx=None):
    """
    保存模型checkpoint

    Args:
        args: 训练参数
        model: 模型
        run_idx: 运行索引 (多次运行时使用)
    """
    model_to_save = model.module if hasattr(model, 'module') else model

    # 使用路径工具获取checkpoint路径
    checkpoint_path = get_checkpoint_path(
        args.paths,
        run_idx=run_idx,
        model_name=args.name,
        ext="bin"
    )

    torch.save(model_to_save.state_dict(), checkpoint_path)

    run_info = f" (run {run_idx})" if run_idx is not None else ""
    logger.info(f"Saved model checkpoint{run_info} to: {checkpoint_path}")


class CheckpointManager:
    """
    Checkpoint管理器 - 支持保存所有checkpoint

    功能：
    - save_all=False: 只保存最佳精度checkpoint（默认行为），只保留最终最佳的单个文件
    - save_all=True: 每个评估步都保存checkpoint，同时更新最佳checkpoint
    """

    def __init__(self, args, run_idx=None):
        """
        Args:
            args: 训练参数
            run_idx: 运行索引 (多次运行时使用)
        """
        self.args = args
        self.run_idx = run_idx
        self.save_all = getattr(args, 'save_all_checkpoints', False)
        self.best_acc = 0.0  # 每次run开始时重置，用于跟踪本次run内最佳
        self.best_epoch = None  # 记录本次run内最佳epoch
        self.best_checkpoint_path = None  # 记录当前最佳checkpoint的路径

        # 获取checkpoint目录
        if run_idx is not None and "run_checkpoint_dirs" in args.paths:
            self.checkpoint_dir = args.paths["run_checkpoint_dirs"][run_idx - 1]
        else:
            self.checkpoint_dir = args.paths["checkpoint_base"]

    def should_save(self, current_acc):
        """判断是否应该保存checkpoint"""
        is_best = current_acc > self.best_acc

        if self.save_all:
            # 保存所有：每次都保存
            should_save = True
        else:
            # 只保存最佳：只有精度提升时才保存
            should_save = is_best

        if is_best:
            self.best_acc = current_acc

        return should_save, is_best

    def save(self, model, step, current_acc, epoch=None):
        """
        保存checkpoint (Save checkpoint)

        Args:
            model: 模型 (Model)
            step: 当前步数 (Current global step)
            current_acc: 当前精度 (Current accuracy)
            epoch: 当前epoch数 (Current epoch number, optional)
        """
        should_save, is_best = self.should_save(current_acc)

        if not should_save:
            return

        model_to_save = model.module if hasattr(model, 'module') else model

        if self.save_all:
            # 保存带step和epoch标签的checkpoint (Save checkpoint with step and epoch labels)
            if epoch is not None:
                step_filename = f"{epoch+1:03d}epoch.bin"
            else:
                step_filename = f"{self.args.name}_step{step}_checkpoint.bin"
            step_path = os.path.join(self.checkpoint_dir, step_filename)
            torch.save(model_to_save.state_dict(), step_path)

            if is_best:
                # 使用相同的命名格式 (隐藏保存日志)
                best_filename = f"{epoch+1:03d}epoch.bin"
                best_path = os.path.join(self.checkpoint_dir, best_filename)
                torch.save(model_to_save.state_dict(), best_path)
                self.best_epoch = epoch + 1
                # 隐藏保存日志
            else:
                # 隐藏保存日志
                pass
        else:
            # 只保存最佳checkpoint，并删除旧的最佳checkpoint
            if is_best:
                # 删除该目录下所有旧的checkpoint文件
                import glob
                old_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "*epoch.bin"))
                for old_file in old_checkpoints:
                    try:
                        os.remove(old_file)
                    except OSError:
                        pass  # 如果删除失败，继续保存新的

                # 保存新的最佳checkpoint，使用 {epoch+1:03d}epoch.bin 格式（三位数字，支持到999 epoch）
                best_filename = f"{epoch+1:03d}epoch.bin"
                best_path = os.path.join(self.checkpoint_dir, best_filename)
                torch.save(model_to_save.state_dict(), best_path)
                self.best_acc = current_acc
                self.best_epoch = epoch + 1  # ✅ 更新best_epoch
                self.best_checkpoint_path = best_path  # 记录当前最佳checkpoint路径
                # 隐藏保存日志


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    # 二分类任务
    num_classes = 2

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, in_channels=1)  # DirtyMNIST固定为单通道
    model.to(args.device)

    # 只在指定了预训练路径且模型类型不是 tiny 时加载预训练权重
    if args.pretrained_dir and not args.model_type.startswith('ViT-Tiny'):
        model.load_from(np.load(args.pretrained_dir))
        logger.info("Loaded pretrained weights from: %s", args.pretrained_dir)
    else:
        logger.info("Training from scratch (no pretrained weights loaded)")

    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def generate_seed_pool(base_seed, num_runs):
    """
    Generate a pool of random seeds.
    Ensures deterministic seed generation for reproducibility.

    Parameters:
        base_seed: Base seed for generating the pool
        num_runs: Number of runs/seeds to generate

    Returns:
        List of seeds, each in range [0, 2^31)
    """
    seed_pool = []
    for i in range(num_runs):
        # Use SHA256 hash for uniform distribution
        seed_str = f"{base_seed}_{i}"
        hash_bytes = hashlib.sha256(seed_str.encode()).digest()
        # Convert first 4 bytes to integer and take modulo 2^31
        seed = int.from_bytes(hash_bytes[:4], byteorder='big') % (2**31)
        seed_pool.append(seed)
    return seed_pool


def valid(args, model, writer, test_loader, global_step):
    """
    验证函数，计算分类指标

    Returns:
        dict: 包含accuracy, precision, recall, f1, auc, confusion_matrix等指标
    """
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label, all_logits = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=True)  # 隐藏验证进度条
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)  # 获取概率用于AUC

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_logits.append(probs.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_logits[0] = np.append(
                all_logits[0], probs.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds = all_preds[0]
    all_label = all_label[0]
    all_logits = all_logits[0]

    # 计算分类指标
    metrics = ClassificationMetrics.compute_metrics(all_label, all_preds, all_logits)
    metrics['loss'] = float(eval_losses.avg)

    # TensorBoard记录
    writer.add_scalar("test/accuracy", scalar_value=metrics['accuracy'], global_step=global_step)
    writer.add_scalar("test/precision", scalar_value=metrics['precision'], global_step=global_step)
    writer.add_scalar("test/recall", scalar_value=metrics['recall'], global_step=global_step)
    writer.add_scalar("test/f1", scalar_value=metrics['f1'], global_step=global_step)
    writer.add_scalar("test/auc", scalar_value=metrics['auc'], global_step=global_step)

    return metrics


def get_loader(args):
    """获取 DirtyMNIST 二分类数据加载器"""
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # 根据 noise_ratio 计算 fast_ratio
    # 如果指定了 noise_ratio，训练集使用对应噪声比例，验证集强制干净
    if args.noise_ratio is not None:
        train_fast_ratio = 1.0 - args.noise_ratio
        test_fast_ratio = 1.0  # 验证集始终干净
        logger.info(f"Noise experiment mode: train_noise={args.noise_ratio:.2f}, "
                   f"train_fast_ratio={train_fast_ratio:.2f}, test_fast_ratio={test_fast_ratio:.2f}")
    else:
        train_fast_ratio = args.fast_ratio
        test_fast_ratio = args.fast_ratio

    # 加载训练集 (分层采样，确保干净/模糊比例一致)
    trainset = load_dirty_mnist_binary(
        root="./data/DirtyMNIST/data",
        train=True,
        download=True,
        class_a=args.class_a,
        class_b=args.class_b,
        max_samples_per_class=args.train_samples_per_class,
        fast_ratio=train_fast_ratio,  # FastMNIST (干净样本) 比例
        img_size=args.img_size
    )

    # 加载测试集 (验证集始终干净，与训练集噪声脱钩)
    testset = load_dirty_mnist_binary(
        root="./data/DirtyMNIST/data",
        train=False,
        download=True,
        class_a=args.class_a,
        class_b=args.class_b,
        max_samples_per_class=args.test_samples_per_class,
        fast_ratio=test_fast_ratio,  # 验证集始终干净 (noise_ratio模式下)
        img_size=args.img_size
    ) if args.local_rank in [-1, 0] else None

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


def train_with_tracking(args, model, train_loader, test_loader, run_idx=None):
    """
    Train the model and track detailed metrics.

    Args:
        run_idx: 运行索引 (多次运行时使用)

    Returns:
        Dictionary containing:
            - seed: Random seed used
            - best_acc: Best validation accuracy
            - final_acc: Final validation accuracy
            - train_losses: List of training losses at each validation step
            - val_accs: List of validation accuracies
            - steps: List of global step numbers
            - train_accs: Optional list of training accuracies
    """
    if args.local_rank in [-1, 0]:
        # 使用路径工具获取TensorBoard日志路径
        log_dir = get_tensorboard_log_path(args.paths)
        writer = SummaryWriter(log_dir=log_dir)

        # Initialize plotter if enabled
        plotter = None
        if getattr(args, 'enable_realtime_plot', False):
            # 传入正确的输出目录
            plotter = TrainingPlotter(
                args.paths["output_dir"],
                model_name=args.name,
                run_idx=run_idx,
                plot_every=getattr(args, 'plot_every', 10)
            )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # 计算每个epoch的步数 (Calculate steps per epoch)
    steps_per_epoch = len(train_loader)
    # 总训练步数 = num_epochs * steps_per_epoch
    t_total = args.num_epochs * steps_per_epoch
    # warmup步数 = warmup_epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    scaler = GradScaler() if args.fp16 else None

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total epochs = %d", args.num_epochs)
    logger.info("  Steps per epoch = %d", steps_per_epoch)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup epochs = %d", args.warmup_epochs)
    logger.info("  Warmup steps = %d", warmup_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility
    losses = AverageMeter()
    global_step, current_run_best_acc = 0, 0  # ← 修复：改名避免与 checkpoint_manager.best_acc 混淆
    best_metrics = None  # 保存最佳指标字典

    # Tracking data
    train_losses = []
    val_accs = []
    train_accs = []
    val_losses = []  # 【FASHIONMNIST专用】验证loss记录，仅FashionMNIST画图需要，其他数据集无需添加
    steps = []

    # 创建checkpoint管理器
    checkpoint_manager = CheckpointManager(args, run_idx)

    # 创建早停器 (patience > 0 时启用)
    early_stopper = None
    if args.early_stopping_patience > 0:
        early_stopper = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    # Epoch-based training loop (基于epoch的训练循环)
    for epoch in range(args.num_epochs):
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc=f"Epoch {epoch+1}/{args.num_epochs} (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 更新进度条描述 (不显示瞬时loss，只显示epoch进度)
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{args.num_epochs}"
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

        # Epoch结束后评估 (Evaluate after each epoch)
        if (epoch + 1) % args.eval_every == 0 and args.local_rank in [-1, 0]:
            # Track training accuracy before validation
            train_acc = compute_train_accuracy(args, model, train_loader)

            metrics = valid(args, model, writer, test_loader, global_step)
            accuracy = metrics['accuracy']
            val_loss = metrics['loss']
            current_lr = scheduler.get_last_lr()[0]

            # 多行输出：每个指标一行
            best_epoch_str = str(checkpoint_manager.best_epoch) if checkpoint_manager.best_epoch is not None else "N/A"
            logger.info("")
            logger.info(f"  train_loss: {losses.avg:.5f}")
            logger.info(f"  val_loss: {val_loss:.5f}")
            logger.info(f"  train_acc: {train_acc:.4f}")
            logger.info(f"  val_acc: {accuracy:.4f}")
            logger.info(f"  best_epoch: {best_epoch_str}")
            logger.info(f"  lr: {current_lr:.6f}")
            logger.info(f"  batch_size: {args.train_batch_size}")
            logger.info("")

            # Record data - 使用losses.avg作为epoch的平均loss
            steps.append(global_step)
            train_losses.append(losses.avg)  # 记录epoch平均loss
            val_accs.append(accuracy)
            train_accs.append(train_acc)
            val_losses.append(val_loss)  # 【FASHIONMNIST专用】验证loss记录，仅FashionMNIST画图需要

            # Update plotter if enabled
            if plotter:
                plotter.update(global_step, losses.avg, train_acc=train_acc, val_acc=accuracy)

            # 使用checkpoint_manager保存 (包含epoch信息)
            checkpoint_manager.save(model, global_step, accuracy, epoch=epoch)

            # 更新current_run_best_acc和best_metrics
            if accuracy > current_run_best_acc:  # 用自己的变量判断
                current_run_best_acc = accuracy  # 更新新变量
                best_metrics = metrics.copy()  # 复制指标字典
                best_metrics['train_acc'] = train_acc  # 添加训练精度
                best_metrics['epoch'] = epoch + 1  # 记录最佳epoch

            # 早停检查
            if early_stopper and early_stopper(accuracy, epoch=epoch + 1):
                logger.info(f"早停触发于 epoch {epoch + 1}/{args.num_epochs}")
                break

            model.train()

        losses.reset()  # 在评估和记录数据之后再reset

    if args.local_rank in [-1, 0]:
        writer.close()

        # Save final plots
        if plotter:
            plotter.save_final_plot()

    # 输出最佳指标总结（带时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_epoch_str = str(checkpoint_manager.best_epoch) if checkpoint_manager.best_epoch is not None else "N/A"
    logger.info("\n========== Best Metrics Summary ==========")
    logger.info("Timestamp: %s" % timestamp)
    logger.info("Best Val Accuracy: %.4f" % current_run_best_acc)
    logger.info("Best Epoch: %s" % best_epoch_str)
    if best_metrics:
        logger.info("Best Train Accuracy: %.4f" % best_metrics.get('train_acc', 0.0))
        logger.info("Best Precision: %.4f" % best_metrics['precision'])
        logger.info("Best Recall: %.4f" % best_metrics['recall'])
        logger.info("Best F1 Score: %.4f" % best_metrics['f1'])
        logger.info("Best AUC-ROC: %.4f" % best_metrics['auc'])
        if 'confusion_matrix' in best_metrics:
            logger.info("Best Confusion Matrix:")
            logger.info(ClassificationMetrics.format_confusion_matrix(best_metrics['confusion_matrix']))
    logger.info("==========================================\n")

    return {
        'seed': args.seed,
        'best_acc': current_run_best_acc,  # 使用新变量名
        'best_metrics': best_metrics,  # 添加完整的最佳指标字典
        'final_acc': val_accs[-1] if val_accs else 0.0,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'train_accs': train_accs,
        'val_losses': val_losses,  # 【FASHIONMNIST专用】验证loss记录，仅FashionMNIST画图需要
        'steps': steps
    }


def train(args, model):
    """ Train the model (legacy function for backward compatibility) """
    train_loader, test_loader = get_loader(args)
    result = train_with_tracking(args, model, train_loader, test_loader)
    return result


def compute_train_accuracy(args, model, train_loader, num_batches=10):
    """
    Compute training accuracy on a subset of training data.

    Parameters:
        args: Training arguments
        model: Model to evaluate
        train_loader: Training data loader
        num_batches: Number of batches to evaluate (default: 10)

    Returns:
        Training accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            logits = model(x)[0]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def run_multiple_experiments(args, model_factory):
    """
    Run multiple experiments with different random seeds.

    Parameters:
        args: Training arguments (should have num_runs, seed_base)
        model_factory: Function that creates a model given args

    Returns:
        List of result dictionaries from each run
    """
    num_runs = getattr(args, 'num_runs', 1)
    seed_base = getattr(args, 'seed_base', 42)

    if num_runs == 1:
        # Single run
        logger.info("Running single experiment")
        set_seed(args)
        model = model_factory(args)
        model.to(args.device)
        train_loader, test_loader = get_loader(args)
        result = train_with_tracking(args, model, train_loader, test_loader, run_idx=None)
        return [result]

    # Multiple runs
    seed_pool = generate_seed_pool(seed_base, num_runs)

    all_results = []

    for run_idx, seed in enumerate(seed_pool, start=1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run_idx}/{num_runs} with seed={seed}")
        logger.info(f"{'='*60}")

        # Update seed
        args.seed = seed
        set_seed(args)

        # Create model
        model = model_factory(args)
        model.to(args.device)

        # Train
        train_loader, test_loader = get_loader(args)
        result = train_with_tracking(args, model, train_loader, test_loader, run_idx=run_idx)
        all_results.append(result)

    return all_results


def summarize_and_plot(results, model_name, args):
    """
    Summarize multiple experiment results and plot average curves.

    Parameters:
        results: List of result dictionaries from train_with_tracking
        model_name: Name of model (for logging and filenames)
        args: Training arguments

    Returns:
        Dictionary with summary statistics
    """
    if len(results) == 1:
        # Single run - just print basic info
        result = results[0]
        print("\n" + "="*60)
        print(f"{model_name} - Single Run Summary")
        print("="*60)
        print(f"Seed: {result['seed']}")
        print(f"Best Accuracy: {result['best_acc']:.6f} ({result['best_acc']*100:.2f}%)")

        if result.get('best_metrics'):
            metrics = result['best_metrics']
            print(f"\nBest Metrics:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {metrics['auc']:.4f}")
            print(f"\n{ClassificationMetrics.format_metrics(metrics)}")

        print("="*60)

        # 单run也保存results.json，与多run格式保持一致
        results_path = get_results_path(args.paths, model_name)
        save_data = {
            'model_name': model_name,
            'num_runs': 1,
            'seeds': [result['seed']],
            'best_accuracies': [result['best_acc']],
            'final_accuracies': [result['final_acc']],
            'best_f1s': [result['best_metrics']['f1']] if result.get('best_metrics') else [],
            'best_precisions': [result['best_metrics']['precision']] if result.get('best_metrics') else [],
            'best_recalls': [result['best_metrics']['recall']] if result.get('best_metrics') else [],
            'best_aucs': [result['best_metrics']['auc']] if result.get('best_metrics') else [],
            # 【FASHIONMNIST专用】历史数据记录，仅FashionMNIST画图需要
            'train_losses': result.get('train_losses', []),
            'val_losses': result.get('val_losses', []),
            'train_accs': result.get('train_accs', []),
            'val_accs': result.get('val_accs', []),
            'metrics_summary': {
                'accuracy': {
                    'mean': float(result['best_acc']),
                    'std': 0.0,
                    'best': float(result['best_acc']),
                    'median': float(result['best_acc']),
                },
                'precision': {
                    'mean': float(result['best_metrics']['precision']) if result.get('best_metrics') else 0.0,
                    'std': 0.0,
                    'best': float(result['best_metrics']['precision']) if result.get('best_metrics') else 0.0,
                    'median': float(result['best_metrics']['precision']) if result.get('best_metrics') else 0.0,
                },
                'recall': {
                    'mean': float(result['best_metrics']['recall']) if result.get('best_metrics') else 0.0,
                    'std': 0.0,
                    'best': float(result['best_metrics']['recall']) if result.get('best_metrics') else 0.0,
                    'median': float(result['best_metrics']['recall']) if result.get('best_metrics') else 0.0,
                },
                'f1': {
                    'mean': float(result['best_metrics']['f1']) if result.get('best_metrics') else 0.0,
                    'std': 0.0,
                    'best': float(result['best_metrics']['f1']) if result.get('best_metrics') else 0.0,
                    'median': float(result['best_metrics']['f1']) if result.get('best_metrics') else 0.0,
                },
                'auc': {
                    'mean': float(result['best_metrics']['auc']) if result.get('best_metrics') else 0.0,
                    'std': 0.0,
                    'best': float(result['best_metrics']['auc']) if result.get('best_metrics') else 0.0,
                    'median': float(result['best_metrics']['auc']) if result.get('best_metrics') else 0.0,
                },
                'best_confusion_matrix': result['best_metrics'].get('confusion_matrix', [[0,0],[0,0]]) if result.get('best_metrics') else [[0,0],[0,0]],
            },
            'summary': {
                'best_mean': float(result['best_acc']),
                'best_std': 0.0,
                'best_min': float(result['best_acc']),
                'best_max': float(result['best_acc']),
                'final_mean': float(result['final_acc']),
                'final_std': 0.0,
            }
        }
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        return {
            'best_accs': [result['best_acc']],
            'final_accs': [result['final_acc']],
            'mean_best': result['best_acc'],
            'std_best': 0.0,
            'mean_final': result['final_acc'],
            'std_final': 0.0,
            'summary': result.get('best_metrics', {})
        }

    # Multiple runs - 使用MetricsSummary汇总
    print("\n" + "="*70)
    print(f"{model_name} - {len(results)} Runs Summary")
    print("="*70)
    print(f"Seeds: {[r['seed'] for r in results]}")
    print()

    # 创建汇总器
    metrics_summary = MetricsSummary()

    # 收集每次运行的最佳指标
    for result in results:
        if result.get('best_metrics'):
            metrics_summary.add_run(result['best_metrics'])

    # 计算汇总统计
    summary = metrics_summary.compute_summary()

    # 打印格式化结果
    print(metrics_summary.format_summary(summary))

    # Plot average curves
    plot_average_curves(results, model_name, args.paths["output_dir"])

    # 使用路径工具保存结果
    results_path = get_results_path(args.paths, model_name)

    # 保存完整结果
    best_accs = [r['best_acc'] for r in results]
    final_accs = [r['final_acc'] for r in results]
    seeds = [r['seed'] for r in results]

    # ========== 新增：提取每次运行的指标列表 ==========
    best_f1s = [r['best_metrics']['f1'] for r in results if r.get('best_metrics')]
    best_precisions = [r['best_metrics']['precision'] for r in results if r.get('best_metrics')]
    best_recalls = [r['best_metrics']['recall'] for r in results if r.get('best_metrics')]
    best_aucs = [r['best_metrics']['auc'] for r in results if r.get('best_metrics')]

    # 【FASHIONMNIST专用】收集所有run的历史数据用于画图
    all_train_losses = [r.get('train_losses', []) for r in results]
    all_val_losses = [r.get('val_losses', []) for r in results]
    all_train_accs = [r.get('train_accs', []) for r in results]
    all_val_accs = [r.get('val_accs', []) for r in results]

    save_data = {
        'model_name': model_name,
        'num_runs': len(results),
        'seeds': seeds,
        'best_accuracies': best_accs,
        'final_accuracies': final_accs,
        # ========== 新增：每次运行的指标列表 ==========
        'best_f1s': best_f1s,
        'best_precisions': best_precisions,
        'best_recalls': best_recalls,
        'best_aucs': best_aucs,
        # 【FASHIONMNIST专用】所有run的历史数据，用于画图分析
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses,
        'all_train_accs': all_train_accs,
        'all_val_accs': all_val_accs,
        # 新增：完整指标汇总
        'metrics_summary': {
            'accuracy': {
                'mean': float(summary['means']['accuracy']),
                'std': float(summary['stds']['accuracy']),
                'best': float(summary['bests']['accuracy']),
                'median': float(summary['medians']['accuracy']),
            },
            'precision': {
                'mean': float(summary['means']['precision']),
                'std': float(summary['stds']['precision']),
                'best': float(summary['bests']['precision']),
                'median': float(summary['medians']['precision']),
            },
            'recall': {
                'mean': float(summary['means']['recall']),
                'std': float(summary['stds']['recall']),
                'best': float(summary['bests']['recall']),
                'median': float(summary['medians']['recall']),
            },
            'f1': {
                'mean': float(summary['means']['f1']),
                'std': float(summary['stds']['f1']),
                'best': float(summary['bests']['f1']),
                'median': float(summary['medians']['f1']),
            },
            'auc': {
                'mean': float(summary['means']['auc']),
                'std': float(summary['stds']['auc']),
                'best': float(summary['bests']['auc']),
                'median': float(summary['medians']['auc']),
            },
            'best_confusion_matrix': summary['best_confusion_matrix'],
        },
        # 保留旧格式兼容
        'summary': {
            'best_mean': float(np.mean(best_accs)),
            'best_std': float(np.std(best_accs)),
            'best_min': float(min(best_accs)),
            'best_max': float(max(best_accs)),
            'final_mean': float(np.mean(final_accs)),
            'final_std': float(np.std(final_accs)),
        }
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return {
        'best_accs': best_accs,
        'final_accs': final_accs,
        'mean_best': np.mean(best_accs),
        'std_best': np.std(best_accs),
        'mean_final': np.mean(final_accs),
        'std_final': np.std(final_accs),
        'metrics_summary': summary
    }


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", choices=["CosViT-Classical-FashionMNIST"],
                        default="CosViT-Classical-FashionMNIST",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Where to search for pretrained ViT models. "
                             "If None or model_type starts with ViT-Tiny, train from scratch.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=28, type=int,
                        help="Resolution size (28 for DirtyMNIST native)")
    # DirtyMNIST固定为单通道，不需要in_channels参数
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1, type=int,
                        help="Run prediction on validation set every N epochs (default: 1 = every epoch)."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=5e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    # Epoch模式训练参数 (Epoch-based training parameters)
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=3, type=int,
                        help="Number of epochs for learning rate warmup.")
    parser.add_argument("--early_stopping_patience", default=20, type=int,
                        help="Early stopping patience (default: 15 epochs without improvement). "
                             "Set to 0 or negative to disable early stopping.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    # DirtyMNIST 二分类参数
    parser.add_argument("--class_a", type=int, default=4,
                        help="First class index (will be mapped to label 0)")
    parser.add_argument("--class_b", type=int, default=9,
                        help="Second class index (will be mapped to label 1)")
    parser.add_argument("--train_samples_per_class", type=int, default=1000,
                        help="Number of training samples per class")
    parser.add_argument("--test_samples_per_class", type=int, default=300,
                        help="Number of test samples per class")
    parser.add_argument("--fast_ratio", type=float, default=0.5,
                        help="Ratio of FastMNIST (clean) samples in each class (0-1). "
                             "Default 0.5 means 50%% clean, 50%% blurred. "
                             "Same ratio is applied to both train and test for consistent distribution.")
    parser.add_argument("--noise_ratio", type=float, default=None,
                        help="Training set noise ratio (0-1). When specified, fast_ratio=1-noise_ratio for train, "
                             "and test set is forced to be clean (fast_ratio=1.0). Overrides --fast_ratio.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    # New parameters for multiple runs and plotting
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of training runs with different seeds")
    parser.add_argument("--seed_base", type=int, default=42,
                        help="Base seed for generating seed pool")
    parser.add_argument("--enable_realtime_plot", action='store_true',
                        help="Enable realtime training curve plotting")
    parser.add_argument("--plot_every", type=int, default=10,
                        help="Plot curves every N validation steps")
    parser.add_argument("--save_all_checkpoints", action='store_true',
                        help="Save checkpoint at every evaluation step (not just best)")

    args = parser.parse_args()

    # 确定数据集名称
    dataset_name = "DirtyMNIST"

    # 获取输出路径结构
    paths = get_output_paths(dataset_name, args.name, args.num_runs)

    # 创建所有输出目录
    create_output_paths(paths)

    # 将路径添加到args中，供后续使用
    args.paths = paths
    args.output_dir = paths["output_dir"]  # 保持兼容性

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging (简化输出格式，隐藏时间戳等前缀)
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model Setup - define factory function for multiple runs
    def model_factory(args):
        """Factory function to create model given args"""
        args_local, model = setup(args)
        return model

    # Run experiments
    results = run_multiple_experiments(args, model_factory)

    # Summarize and plot
    if args.local_rank in [-1, 0]:
        summary = summarize_and_plot(results, args.name, args)


if __name__ == "__main__":
    main()
