# coding=utf-8
"""
Quantum self-attention ViT training script - FashionMNIST binary classification

Train quantum attention ViT using QuantumVisionTransformer.
"""

from __future__ import absolute_import, division, print_function

import sys
import os

# Add project root to sys.path to allow running from subdirectory
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

from models.QVIT import QuantumVisionTransformer, QUANTUM_CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.early_stopping import EarlyStopping
from utils.dist_util import get_world_size
from data.FashionMNIST.dataloader import load_fashion_mnist_binary
from utils.training_plotter import TrainingPlotter, plot_average_curves
from utils.path_utils import (
    get_output_paths,
    create_output_paths,
    get_checkpoint_path,
    get_tensorboard_log_path,
    get_results_path,
    get_plot_path
)

# Classification metrics computation
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """
    Classification metrics calculator

    For binary classification metric computation
    """

    @staticmethod
    def compute_metrics(labels, preds, probs=None):
        """
        Compute classification metrics

        Args:
            labels: Ground truth labels, shape: [N], values in {0, 1}
            preds: Predicted labels, shape: [N], values in {0, 1}
            probs: Predicted probabilities (for AUC-ROC), shape: [N, 2] or [N]

        Returns:
            dict: Dictionary containing various metrics
        """
        # Basic metrics
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

        # AUC-ROC (if probabilities are provided)
        if probs is not None:
            try:
                # If probs is [N, 2], take column 1 (positive class probability)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs_positive = probs[:, 1]
                else:
                    probs_positive = probs

                auc = roc_auc_score(labels, probs_positive)
                metrics['auc'] = float(auc)
            except Exception:
                # Set default value if computation fails (e.g., only one class)
                metrics['auc'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization

        return metrics

    @staticmethod
    def format_metrics(metrics):
        """
        Format metrics for output

        Args:
            metrics: Metrics dictionary

        Returns:
            str: Formatted string
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
        Format confusion matrix (for log output)

        Args:
            cm: Confusion matrix list [[TN, FP], [FN, TP]]

        Returns:
            str: Formatted string
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
    Multi-run metrics summarizer

    Features:
    1. Collect best metrics from each run
    2. Compute mean, std, median, best
    3. Format output
    """

    def __init__(self):
        self.runs = []  # Store best metrics from each run

    def add_run(self, best_metrics):
        """
        Add best metrics from a run

        Args:
            best_metrics: dict, containing accuracy, precision, recall, f1, auc, confusion_matrix
        """
        self.runs.append(best_metrics)

    def compute_summary(self):
        """
        Compute summary statistics

        Returns:
            dict: Dictionary containing statistics for each metric
        """
        if not self.runs:
            return None

        # Extract metrics
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        summary = {
            'num_runs': len(self.runs),
            'means': {},
            'stds': {},
            'bests': {},
            'medians': {},
            'best_run_idx': {},  # Best run index (for confusion matrix)
        }

        # Compute statistics for each metric
        for metric in metrics_names:
            values = [run[metric] for run in self.runs]

            summary['means'][metric] = float(np.mean(values))
            summary['stds'][metric] = float(np.std(values))
            summary['bests'][metric] = float(np.max(values))
            summary['medians'][metric] = float(np.median(values))

            # Find best run (based on f1)
            if metric == 'f1':
                best_idx = int(np.argmax(values))
                summary['best_run_idx'][metric] = best_idx

        # Save confusion matrix of best run
        best_f1_idx = summary['best_run_idx']['f1']
        summary['best_confusion_matrix'] = self.runs[best_f1_idx]['confusion_matrix']

        return summary

    def format_summary(self, summary):
        """
        Format summary results for output

        Args:
            summary: Summary dictionary returned by compute_summary()

        Returns:
            str: Formatted string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"{summary['num_runs']} Runs Summary")
        lines.append("=" * 70)

        # Table header
        lines.append(f"{'Metric':<12} {'Mean ± Std':<20} {'Best':<10} {'Median':<10}")
        lines.append("-" * 70)

        # Each metric
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

        # Confusion matrix (from best run)
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
    Save model checkpoint

    Args:
        args: Training arguments
        model: Model
        run_idx: Run index (used for multiple runs)
    """
    model_to_save = model.module if hasattr(model, 'module') else model

    # Use path utility to get checkpoint path
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
    Checkpoint manager - supports saving all checkpoints

    Features:
    - save_all=False: Only save best accuracy checkpoint (default), keep single best file
    - save_all=True: Save checkpoint at every eval step, also update best checkpoint
    """

    def __init__(self, args, run_idx=None):
        """
        Args:
            args: Training arguments
            run_idx: Run index (used for multiple runs)
        """
        self.args = args
        self.run_idx = run_idx
        self.save_all = getattr(args, 'save_all_checkpoints', False)
        self.best_acc = 0.0  # Reset at start of each run, track best within this run
        self.best_epoch = None  # Record best epoch within this run
        self.best_checkpoint_path = None  # Record path of current best checkpoint

        # Get checkpoint directory
        if run_idx is not None and "run_checkpoint_dirs" in args.paths:
            self.checkpoint_dir = args.paths["run_checkpoint_dirs"][run_idx - 1]
        else:
            self.checkpoint_dir = args.paths["checkpoint_base"]

    def should_save(self, current_acc):
        """Determine whether to save checkpoint"""
        is_best = current_acc > self.best_acc

        if self.save_all:
            # Save all: always save
            should_save = True
        else:
            # Only save best: save only when accuracy improves
            should_save = is_best

        if is_best:
            self.best_acc = current_acc

        return should_save, is_best

    def save(self, model, step, current_acc, epoch=None):
        """
        Save checkpoint

        Args:
            model: Model (Model)
            step: Current global step
            current_acc: Current accuracy
            epoch: Current epoch number (optional)
        """
        should_save, is_best = self.should_save(current_acc)

        if not should_save:
            return

        model_to_save = model.module if hasattr(model, 'module') else model

        if self.save_all:
            # Save checkpoint, naming format: {epoch:03d}epoch.bin (e.g.: 005epoch.bin)
            if epoch is not None:
                step_filename = f"{epoch+1:03d}epoch.bin"
            else:
                step_filename = f"{self.args.name}_step{step}_checkpoint.bin"
            step_path = os.path.join(self.checkpoint_dir, step_filename)
            torch.save(model_to_save.state_dict(), step_path)

            if is_best:
                # Also update best checkpoint (using same naming format)
                best_filename = f"{epoch+1:03d}epoch.bin"
                best_path = os.path.join(self.checkpoint_dir, best_filename)
                torch.save(model_to_save.state_dict(), best_path)
                self.best_epoch = epoch + 1
                # Suppress save log
            else:
                # Suppress save log
                pass
        else:
            # Only save best checkpoint, and delete old best checkpoint
            if is_best:
                # Delete all old checkpoint files in this directory
                import glob
                old_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "*epoch.bin"))
                for old_file in old_checkpoints:
                    try:
                        os.remove(old_file)
                    except OSError:
                        pass  # If deletion fails, continue to save new one

                # Save new best checkpoint, using {epoch+1:03d}epoch.bin format (3 digits, supports up to 999 epochs)
                best_filename = f"{epoch+1:03d}epoch.bin"
                best_path = os.path.join(self.checkpoint_dir, best_filename)
                torch.save(model_to_save.state_dict(), best_path)
                self.best_acc = current_acc
                self.best_epoch = epoch + 1  # Update best_epoch
                self.best_checkpoint_path = best_path  # Record current best checkpoint path
                # Suppress save log


def setup(args):
    # Prepare model
    config = QUANTUM_CONFIGS[args.model_type]

    # Binary classification task
    num_classes = 2

    # ========== Key change: Use QuantumVisionTransformer instead of VisionTransformer ==========
    model = QuantumVisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, in_channels=args.in_channels)
    model.to(args.device)

    # Quantum attention model does not load pretrained weights (different architecture), train from scratch
    logger.info("Training quantum attention model from scratch")

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
        base_seed: Base seed for generating pool
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
    Validation function, compute classification metrics

    Returns:
        dict: Contains accuracy, precision, recall, f1, auc, confusion_matrix, etc.
    """
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label, all_logits = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=True)  # Hide validation progress bar
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)  # Get probabilities for AUC

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

    # Compute classification metrics
    metrics = ClassificationMetrics.compute_metrics(all_label, all_preds, all_logits)
    metrics['loss'] = float(eval_losses.avg)

    # TensorBoard logging
    writer.add_scalar("test/accuracy", scalar_value=metrics['accuracy'], global_step=global_step)
    writer.add_scalar("test/precision", scalar_value=metrics['precision'], global_step=global_step)
    writer.add_scalar("test/recall", scalar_value=metrics['recall'], global_step=global_step)
    writer.add_scalar("test/f1", scalar_value=metrics['f1'], global_step=global_step)
    writer.add_scalar("test/auc", scalar_value=metrics['auc'], global_step=global_step)

    return metrics


def get_loader(args):
    """Get FashionMNIST binary classification data loader"""
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Load training set
    trainset = load_fashion_mnist_binary(
        root="./data",
        train=True,
        download=True,
        class_a=args.class_a,
        class_b=args.class_b,
        max_samples_per_class=args.train_samples_per_class,
        img_size=args.img_size,
        in_channels=args.in_channels
    )

    # Load test set
    testset = load_fashion_mnist_binary(
        root="./data",
        train=False,
        download=True,
        class_a=args.class_a,
        class_b=args.class_b,
        max_samples_per_class=args.test_samples_per_class,
        img_size=args.img_size,
        in_channels=args.in_channels
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
    Train model and track detailed metrics.

    Args:
        run_idx: Run index (used for multiple runs)

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
        # Use path utility to get TensorBoard log path
        log_dir = get_tensorboard_log_path(args.paths)
        writer = SummaryWriter(log_dir=log_dir)

        # Initialize plotter if enabled
        plotter = None
        if getattr(args, 'enable_realtime_plot', False):
            # Pass correct output directory
            plotter = TrainingPlotter(
                args.paths["output_dir"],
                model_name=args.name,
                run_idx=run_idx,
                plot_every=getattr(args, 'plot_every', 10)
            )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    # Total training steps = num_epochs * steps_per_epoch
    t_total = args.num_epochs * steps_per_epoch
    # Warmup steps = warmup_epochs * steps_per_epoch
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
    global_step, current_run_best_acc = 0, 0  # Fix: renamed to avoid confusion with checkpoint_manager.best_acc
    best_metrics = None  # Store best metrics dictionary

    # Tracking data
    train_losses = []
    val_accs = []
    train_accs = []
    val_losses = []  # [FASHIONMNIST-specific] Validation loss record, only needed for FashionMNIST plotting
    steps = []

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(args, run_idx)

    # Create early stopper (enabled when patience > 0)
    early_stopper = None
    if args.early_stopping_patience > 0:
        early_stopper = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    # Epoch-based training loop
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

                # Update progress bar description (hide instantaneous loss, show epoch progress only)
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{args.num_epochs}"
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

        # Evaluate after each epoch
        if (epoch + 1) % args.eval_every == 0 and args.local_rank in [-1, 0]:
            # Track training accuracy before validation
            train_acc = compute_train_accuracy(args, model, train_loader)

            metrics = valid(args, model, writer, test_loader, global_step)
            accuracy = metrics['accuracy']
            val_loss = metrics['loss']
            current_lr = scheduler.get_last_lr()[0]

            # Multi-line output: one metric per line
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

            # Record data - Use losses.avg as epoch average loss
            steps.append(global_step)
            train_losses.append(losses.avg)  # Record epoch average loss
            val_accs.append(accuracy)
            train_accs.append(train_acc)
            val_losses.append(val_loss)  # [FASHIONMNIST-specific] Validation loss record

            # Update plotter if enabled
            if plotter:
                plotter.update(global_step, losses.avg, train_acc=train_acc, val_acc=accuracy)

            # Save using checkpoint_manager (includes epoch info)
            checkpoint_manager.save(model, global_step, accuracy, epoch=epoch)

            # Update current_run_best_acc and best_metrics
            if accuracy > current_run_best_acc:  # Use own variable for comparison
                current_run_best_acc = accuracy  # Update new variable
                best_metrics = metrics.copy()  # Copy metrics dictionary
                best_metrics['train_acc'] = train_acc  # Add training accuracy
                best_metrics['epoch'] = epoch + 1  # Record best epoch

            # Early stopping check
            if early_stopper and early_stopper(accuracy, epoch=epoch + 1):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}/{args.num_epochs}")
                break

            model.train()

        losses.reset()  # Reset after evaluation and data recording

    if args.local_rank in [-1, 0]:
        writer.close()

        # Save final plots
        if plotter:
            plotter.save_final_plot()

    # Output best metrics summary (with timestamp)
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
        'best_acc': current_run_best_acc,  # Use new variable name
        'best_metrics': best_metrics,  # Complete best metrics dictionary
        'final_acc': val_accs[-1] if val_accs else 0.0,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'train_accs': train_accs,
        'val_losses': val_losses,  # [FASHIONMNIST-specific] Validation loss record
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

        # Save results using path utility (consistent with multi-run)
        results_path = get_results_path(args.paths, model_name)

        # Build single-run save_data (consistent with multi-run format)
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
            # [FASHIONMNIST-specific] Historical data records for plotting
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

    # Multiple runs - summarize using MetricsSummary
    print("\n" + "="*70)
    print(f"{model_name} - {len(results)} Runs Summary")
    print("="*70)
    print(f"Seeds: {[r['seed'] for r in results]}")
    print()

    # Create summarizer
    metrics_summary = MetricsSummary()

    # Collect best metrics from each run
    for result in results:
        if result.get('best_metrics'):
            metrics_summary.add_run(result['best_metrics'])

    # Compute summary statistics
    summary = metrics_summary.compute_summary()

    # Print formatted results
    print(metrics_summary.format_summary(summary))

    # Plot average curves
    plot_average_curves(results, model_name, args.paths["output_dir"])

    # Use path utility to save results
    results_path = get_results_path(args.paths, model_name)

    # Save complete results
    best_accs = [r['best_acc'] for r in results]
    final_accs = [r['final_acc'] for r in results]
    seeds = [r['seed'] for r in results]

    # ========== Extract per-run metric lists ==========
    best_f1s = [r['best_metrics']['f1'] for r in results if r.get('best_metrics')]
    best_precisions = [r['best_metrics']['precision'] for r in results if r.get('best_metrics')]
    best_recalls = [r['best_metrics']['recall'] for r in results if r.get('best_metrics')]
    best_aucs = [r['best_metrics']['auc'] for r in results if r.get('best_metrics')]

    # [FASHIONMNIST-specific] Collect historical data from all runs for plotting
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
        # ========== Per-run metric lists ==========
        'best_f1s': best_f1s,
        'best_precisions': best_precisions,
        'best_recalls': best_recalls,
        'best_aucs': best_aucs,
        # [FASHIONMNIST-specific] Historical data from all runs for plotting
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses,
        'all_train_accs': all_train_accs,
        'all_val_accs': all_val_accs,
        # Complete metrics summary
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
        # Keep old format for compatibility
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
    parser.add_argument("--model_type", choices=["ViT-Quantum-Tiny", "ViT-Quantum-Tiny-1", "ViT-Quantum-FashionMNIST"],
                        default="ViT-Quantum-FashionMNIST",
                        help="Which quantum ViT variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Not used for quantum models (trained from scratch).")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=28, type=int,
                        help="Resolution size (28 for FashionMNIST native, 224 for upsampled)")
    parser.add_argument("--in_channels", default=1, type=int,
                        choices=[1, 3],
                        help="Input image channels (1 for grayscale FashionMNIST native, 3 for RGB). Default: 1 (native grayscale, more efficient)")
    parser.add_argument("--train_batch_size", default=16, type=int,
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
    # Epoch-based training parameters
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

    # FashionMNIST binary classification parameters
    parser.add_argument("--class_a", type=int, default=0,
                        help="First class index (will be mapped to label 0)")
    parser.add_argument("--class_b", type=int, default=1,
                        help="Second class index (will be mapped to label 1)")
    parser.add_argument("--train_samples_per_class", type=int, default=250,
                        help="Number of training samples per class")
    parser.add_argument("--test_samples_per_class", type=int, default=100,
                        help="Number of test samples per class")

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

    # Determine dataset name
    dataset_name = "FashionMNIST"

    # Get output path structure
    paths = get_output_paths(dataset_name, args.name, args.num_runs)

    # Create all output directories
    create_output_paths(paths)

    # Add paths to args for later use
    args.paths = paths
    args.output_dir = paths["output_dir"]  # Keep for compatibility

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

    # Setup logging (Simplified output format, hide timestamp prefix)
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
