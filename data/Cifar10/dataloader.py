# coding=utf-8
"""
CIFAR-10 Data Loader

Supports two modes:
1. Full 10-class: load_dataset_cifar10()
2. Binary classification: load_cifar10_binary() - select 2 from 10 classes
"""

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import os

DOWNLOAD_CIFAR10 = True


# CIFAR-10 class mapping (0-9)
CIFAR10_CLASSES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


class CIFAR10Binary(Dataset):
    """
    CIFAR-10 binary classification dataset

    Selects two classes from CIFAR-10 and relabels them as 0 and 1
    """
    def __init__(self, root, train=True, transform=None,
                 class_a=0, class_b=1, download=True,
                 max_samples_per_class=None):
        """
        Args:
            root: dataset root directory
            train: True=training set, False=test set
            transform: image transforms
            class_a: first class index (0-9), relabeled as 0
            class_b: second class index (0-9), relabeled as 1
            download: whether to auto-download
            max_samples_per_class: max samples per class (None=all)
        """
        self.class_a = class_a
        self.class_b = class_b
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class

        # Load full CIFAR-10 dataset
        self.full_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # handle transforms manually
        )

        # Filter for the two target classes
        data_a = []  # class_a samples
        data_b = []  # class_b samples
        targets_a = []
        targets_b = []

        for idx, (img, label) in enumerate(zip(self.full_dataset.data, self.full_dataset.targets)):
            if label == class_a:
                data_a.append(img)
                targets_a.append(0)
            elif label == class_b:
                data_b.append(img)
                targets_b.append(1)

        # Limit samples per class
        if max_samples_per_class is not None:
            data_a = data_a[:max_samples_per_class]
            targets_a = targets_a[:max_samples_per_class]
            data_b = data_b[:max_samples_per_class]
            targets_b = targets_b[:max_samples_per_class]

        # Merge both classes
        self.data = np.array(data_a + data_b)
        self.targets = np.array(targets_a + targets_b)

        # Convert to numpy arrays
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        print(f"[CIFAR10Binary] ClassA(0): {CIFAR10_CLASSES[class_a]}, "
              f"ClassB(1): {CIFAR10_CLASSES[class_b]}, Samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]

        # Convert to PIL Image for transform
        from PIL import Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

def load_dataset_cifar10(train_transform=None, test_transform=None):
    if train_transform is None:
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2470, 0.2435, 0.2616]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    if test_transform is None:
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2470, 0.2435, 0.2616]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    import os
    # Use absolute path pointing to project root data/Cifar10
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'Cifar10')

    train_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=DOWNLOAD_CIFAR10,
        transform=train_transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=DOWNLOAD_CIFAR10,
        transform=test_transform
    )

    return train_data, test_data


def load_cifar10_binary(train_transform=None, test_transform=None,
                       class_a=0, class_b=1, img_size=32,
                       train_samples_per_class=None, test_samples_per_class=None):
    """
    Load CIFAR-10 binary classification dataset

    Args:
        train_transform: training image transforms
        test_transform: test image transforms
        class_a: first class index (0-9)
        class_b: second class index (0-9)
        img_size: target image size (default 32, CIFAR-10 native)
        train_samples_per_class: max training samples per class
        test_samples_per_class: max test samples per class

    Returns:
        trainset, testset
    """
    # CIFAR-10 normalization (RGB 3-channel)
    norm_mean = [0.4914, 0.4822, 0.4465]
    norm_std = [0.2470, 0.2435, 0.2616]

    if train_transform is None:
        if img_size == 32:
            # Native size, no resize
            # Strong augmentation for CIFAR-10: target accuracy 90%
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))
            ])
        else:
            # Need resize
            train_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))
            ])

    if test_transform is None:
        if img_size == 32:
            # Native size, no resize
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else:
            # Need resize
            test_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

    # Use absolute path pointing to project root data/Cifar10
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'Cifar10')

    trainset = CIFAR10Binary(
        root=data_path,
        train=True,
        transform=train_transform,
        class_a=class_a,
        class_b=class_b,
        download=True,
        max_samples_per_class=train_samples_per_class
    )

    testset = CIFAR10Binary(
        root=data_path,
        train=False,
        transform=test_transform,
        class_a=class_a,
        class_b=class_b,
        download=True,
        max_samples_per_class=test_samples_per_class
    )

    return trainset, testset