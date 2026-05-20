import torch
import ddu_dirty_mnist
from torch.utils.data import Subset, Dataset
import numpy as np
from torchvision import transforms


class DirtyMNISTBinary(Dataset):
    """
    DirtyMNIST binary classification dataset (stratified sampling)

    Features:
        - Stratified sampling: controls clean/noisy ratio within each class
        - Training and validation sets can have the same distribution
        - Image size: 28x28 grayscale (single channel)
        - Normalization: MNIST standard (mean=0.1307, std=0.3081)
    """

    def __init__(self, root="./data/DirtyMNIST/data", train=True, download=True,
                 class_a=4, class_b=9, max_samples_per_class=None,
                 fast_ratio=0.5,  # FastMNIST ratio (0-1)
                 img_size=28, verbose=True):
        """
        Args:
            root: dataset root directory
            train: whether to use training set
            download: whether to download dataset
            class_a: first class index (relabeled as 0), default 4
            class_b: second class index (relabeled as 1), default 9
            max_samples_per_class: max samples per class (None = use all)
            fast_ratio: FastMNIST (clean) ratio (0-1), default 0.5
            img_size: target image size (28=native, other values trigger resize)
            verbose: whether to print details
        """
        self.class_a = class_a
        self.class_b = class_b
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.verbose = verbose

        # Load DirtyMNIST dataset
        dirty_mnist_full = ddu_dirty_mnist.DirtyMNIST(
            root,
            train=train,
            transform=None,
            download=download,
            device="cpu"
        )

        self.base_dataset = dirty_mnist_full

        # Get all labels and source info
        # Training: 0-59999 = FastMNIST, 60000-119999 = AmbiguousMNIST
        # Test: 0-9999 = FastMNIST, 10000-69999 = AmbiguousMNIST
        if train:
            fast_boundary = 60000
            total_fast = 60000
            total_amb = 60000
        else:
            fast_boundary = 10000
            total_fast = 10000
            total_amb = 60000

        # Get all labels
        all_labels = self._get_all_labels(dirty_mnist_full)

        # Stratified sampling: sample from FastMNIST and AmbiguousMNIST separately per class
        indices_a_fast, indices_a_amb = self._stratified_sample(
            all_labels, class_a, fast_boundary, max_samples_per_class, fast_ratio
        )
        indices_b_fast, indices_b_amb = self._stratified_sample(
            all_labels, class_b, fast_boundary, max_samples_per_class, fast_ratio
        )

        # Merge indices (class_a -> label 0, class_b -> label 1)
        indices_a = indices_a_fast + indices_a_amb
        indices_b = indices_b_fast + indices_b_amb

        self.indices = indices_a + indices_b
        self.labels = np.array([0] * len(indices_a) + [1] * len(indices_b))

        # Pre-build Resize transform
        self.resize_transform = None
        if img_size != 28:
            self.resize_transform = transforms.Resize((img_size, img_size))

        # Statistics
        self.source_stats = {
            'fastmnist': len(indices_a_fast) + len(indices_b_fast),
            'ambiguous': len(indices_a_amb) + len(indices_b_amb),
            'class_a_fast': len(indices_a_fast),
            'class_a_amb': len(indices_a_amb),
            'class_b_fast': len(indices_b_fast),
            'class_b_amb': len(indices_b_amb),
        }
        total = len(self.indices)
        self.source_stats['fastmnist_pct'] = self.source_stats['fastmnist'] / total * 100 if total > 0 else 0
        self.source_stats['ambiguous_pct'] = self.source_stats['ambiguous'] / total * 100 if total > 0 else 0

        if verbose:
            print(f"=" * 60)
            print(f"DirtyMNIST {'Training' if train else 'Test'} Dataset (Binary)")
            print(f"Selected classes: {class_a} (label 0) vs {class_b} (label 1)")
            print(f"Target FastMNIST ratio: {fast_ratio*100:.0f}%")
            print(f"-" * 60)
            print(f"Class {class_a}: {len(indices_a)} samples "
                  f"({len(indices_a_fast)} clean, {len(indices_a_amb)} blur)")
            print(f"Class {class_b}: {len(indices_b)} samples "
                  f"({len(indices_b_fast)} clean, {len(indices_b_amb)} blur)")
            print(f"-" * 60)
            print(f"Source breakdown:")
            print(f"  FastMNIST (clean):    {self.source_stats['fastmnist']} ({self.source_stats['fastmnist_pct']:.1f}%)")
            print(f"  AmbiguousMNIST (blur): {self.source_stats['ambiguous']} ({self.source_stats['ambiguous_pct']:.1f}%)")
            print(f"=" * 60)

    def _get_all_labels(self, dataset):
        """Efficiently get all labels"""
        if hasattr(dataset, 'datasets'):
            targets_list = []
            for ds in dataset.datasets:
                if hasattr(ds, 'targets'):
                    targets_list.append(ds.targets)
            if targets_list:
                return torch.cat(targets_list).numpy()
        elif hasattr(dataset, 'targets'):
            return dataset.targets.numpy()

        # Fallback method
        labels = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        return np.array(labels)

    def _stratified_sample(self, all_labels, target_class, fast_boundary, max_samples, fast_ratio):
        """
        Stratified sampling: sample from FastMNIST and AmbiguousMNIST by ratio

        Returns: (fast_indices, amb_indices)
        """
        # Find all indices for the target class
        class_mask = all_labels == target_class
        class_indices = np.where(class_mask)[0]

        # Separate FastMNIST and AmbiguousMNIST indices
        fast_indices = [idx for idx in class_indices if idx < fast_boundary]
        amb_indices = [idx for idx in class_indices if idx >= fast_boundary]

        # If max_samples specified, allocate by ratio
        if max_samples is not None:
            n_fast_target = int(max_samples * fast_ratio)
            n_amb_target = max_samples - n_fast_target

            # Random sampling
            import random
            random.seed(42)

            if len(fast_indices) > n_fast_target:
                fast_indices = random.sample(fast_indices, n_fast_target)
            else:
                fast_indices = fast_indices[:n_fast_target]

            if len(amb_indices) > n_amb_target:
                amb_indices = random.sample(amb_indices, n_amb_target)
            else:
                amb_indices = amb_indices[:n_amb_target]

        return fast_indices, amb_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[self.indices[idx]]
        img = (img - self.mean) / self.std

        if self.resize_transform is not None:
            img = self.resize_transform(img)

        label = int(self.labels[idx])
        return img, label


def load_dirty_mnist_binary(root="./data/DirtyMNIST/data", train=True, download=True,
                            class_a=4, class_b=9, max_samples_per_class=None,
                            fast_ratio=0.5, img_size=28, verbose=True):
    """
    Load DirtyMNIST binary classification dataset (stratified sampling)

    Args:
        root: dataset root directory
        train: whether to use training set
        download: whether to download dataset
        class_a: first class index (relabeled as 0), default 4
        class_b: second class index (relabeled as 1), default 9
        max_samples_per_class: max samples per class (None = use all)
        fast_ratio: FastMNIST (clean) ratio, default 0.5 (50% clean, 50% blur)
        img_size: target image size
        verbose: whether to print details

    Example:
        # Training: 1000 per class, 50% clean, 50% blur
        trainset = load_dirty_mnist_binary(train=True, max_samples_per_class=1000, fast_ratio=0.5)

        # Validation: 500 per class, same 50/50 split
        testset = load_dirty_mnist_binary(train=False, max_samples_per_class=500, fast_ratio=0.5)
    """
    return DirtyMNISTBinary(root=root, train=train, download=download,
                           class_a=class_a, class_b=class_b,
                           max_samples_per_class=max_samples_per_class,
                           fast_ratio=fast_ratio,
                           img_size=img_size, verbose=verbose)
