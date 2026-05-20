import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, transforms


class FashionMNISTBinary(Dataset):
    """
    FashionMNIST binary classification dataset
    Selects two specified classes from FashionMNIST for binary classification

    Class mapping:
        - 0: T-shirt/top
        - 1: Trouser
        - 2: Pullover
        - 3: Dress
        - 4: Coat
        - 5: Sandal
        - 6: Shirt
        - 7: Sneaker
        - 8: Bag
        - 9: Ankle boot
    """

    def __init__(self, root="./data", train=True, download=True, class_a=0, class_b=1,
                 max_samples_per_class=None, img_size=28, in_channels=1):
        """
        Args:
            root: dataset root directory
            train: whether to use training set
            download: whether to download the dataset
            class_a: first class index (mapped to label 0)
            class_b: second class index (mapped to label 1)
            max_samples_per_class: max samples per class (None = use all)
            img_size: target image size (28=native, other values will resize)
            in_channels: output channels (1=grayscale, 3=replicate to RGB, default=1)
        """
        self.class_a = class_a
        self.class_b = class_b
        self.img_size = img_size
        self.in_channels = in_channels

        # Load the full FashionMNIST dataset
        self.full_dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=download
        )

        # Filter sample indices for target classes
        indices_a, indices_b = [], []
        for idx, (_, label) in enumerate(self.full_dataset):
            if label == class_a:
                indices_a.append(idx)
            elif label == class_b:
                indices_b.append(idx)

        # Bugfix: use random sampling instead of slice sampling to avoid sampling bias
        if max_samples_per_class is not None:
            import random
            random.seed(42)
            if len(indices_a) > max_samples_per_class:
                indices_a = random.sample(indices_a, max_samples_per_class)
            else:
                indices_a = indices_a[:max_samples_per_class]

            if len(indices_b) > max_samples_per_class:
                indices_b = random.sample(indices_b, max_samples_per_class)
            else:
                indices_b = indices_b[:max_samples_per_class]

        # Merge indices and generate labels
        indices = indices_a + indices_b
        labels = [0] * len(indices_a) + [1] * len(indices_b)

        self.indices = indices
        self.labels = np.array(labels)

        # Bugfix: pre-build Resize transform to avoid repeated instantiation
        self.resize_transform = None
        if img_size != 28:
            self.resize_transform = transforms.Resize((img_size, img_size))

        print(f"=" * 60)
        print(f"FashionMNIST {'Training' if train else 'Test'} Dataset (Binary)")
        print(f"Selected classes: {class_a} -> {self.get_class_name(class_a)} "
              f"vs {class_b} -> {self.get_class_name(class_b)}")
        print(f"Total samples: {len(self.indices)}")
        print(f"Class {class_a} samples: {np.sum(self.labels == 0)}")
        print(f"Class {class_b} samples: {np.sum(self.labels == 1)}")
        print(f"=" * 60)

    @staticmethod
    def get_class_name(class_idx):
        """Get class name"""
        class_names = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }
        return class_names.get(class_idx, f"Unknown({class_idx})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get original sample
        img, _ = self.full_dataset[self.indices[idx]]

        # Convert to Tensor
        img = transforms.ToTensor()(img)

        # Replicate channel if in_channels == 3
        if self.in_channels == 3:
            # Expand single channel to 3 channels (replicate 3 times)
            img = img.repeat(3, 1, 1)
        # When in_channels == 1, keep single channel, no processing

        # Bugfix: use pre-built Resize transform
        if self.resize_transform is not None:
            img = self.resize_transform(img)

        # Apply MNIST standard normalization (mean=0.1307, std=0.3081)
        img = (img - 0.1307) / 0.3081

        # Bugfix: convert label to Python int to ensure correct DataLoader collation
        label = int(self.labels[idx])

        return img, label


def load_fashion_mnist_binary(root="./data", train=True, download=True, class_a=0, class_b=1,
                              max_samples_per_class=None, img_size=28, in_channels=1):
    """
    Load FashionMNIST binary classification dataset

    Args:
        root: dataset root directory
        train: whether to use training set
        download: whether to download the dataset
        class_a: first class index (mapped to label 0)
        class_b: second class index (mapped to label 1)
        max_samples_per_class: max samples per class
        img_size: target image size (28=native, other values will resize)
        in_channels: output channels (1=grayscale, 3=replicate to RGB, default=1)

    Returns:
        FashionMNISTBinary dataset object
    """
    return FashionMNISTBinary(root=root, train=train, download=download,
                             class_a=class_a, class_b=class_b,
                             max_samples_per_class=max_samples_per_class,
                             img_size=img_size, in_channels=in_channels)
