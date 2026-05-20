"""
FER2013 facial expression recognition dataset loader

Dataset info:
- Image size: 48x48 grayscale (original images may differ, resize applied)
- Number of classes: 7 expressions
  0: Angry
  1: Disgust
  2: Fear
  3: Happy
  4: Sad
  5: Surprise
  6: Neutral
- Dataset split:
  - Training: ~28,709 images
  - Test (PublicTest): ~7,178 images
  - Total: ~35,887 images

Data format: image folder structure
- train/{emotion}/{filename}.jpg
- test/{emotion}/{filename}.jpg

Download: https://www.kaggle.com/datasets/msambare/fer2013
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
import os


class FER2013Binary(Dataset):
    """
    FER2013 binary classification dataset (image folder format)

    Selects two specified classes from FER2013's 7 expressions for binary classification

    Bugfix version:
    - Supports image folder format (not CSV)
    - Pre-loads all image paths to avoid repeated traversal
    - Converts labels to int
    - Pre-builds Resize transform

    Class mapping:
        - 0: Angry
        - 1: Disgust
        - 2: Fear
        - 3: Happy
        - 4: Sad
        - 5: Surprise
        - 6: Neutral
    """

    # Emotion class name mapping
    EMOTION_NAMES = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    }

    # Emotion name to index mapping (for folder matching)
    EMOTION_TO_IDX = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6
    }

    def __init__(self, data_dir, usage='Training', class_a=3, class_b=5,
                 max_samples_per_class=2000, img_size=48, in_channels=1, augmentation='weak'):
        """
        Args:
            data_dir: FER2013 data root directory (e.g.: ./data/FER2013/data)
            usage: dataset split ('Training' or 'Test')
            class_a: first class index (mapped to label 0)
            class_b: second class index (mapped to label 1)
            max_samples_per_class: max samples per class (default 2000 train/1000 test)
            img_size: target image size (48=native, other values will resize)
            in_channels: output channels (1=grayscale, 3=replicate to RGB, default=1)
            augmentation: augmentation strength ('none', 'weak', 'medium', 'strong')
        """
        self.class_a = class_a
        self.class_b = class_b
        self.img_size = img_size
        self.in_channels = in_channels
        self.augmentation = augmentation
        self.usage = usage  # Save usage; test set should not apply augmentation

        # Check directory existence
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"FER2013 data directory not found: {data_dir}\n"
                                    f"Please download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013")

        # Determine subdirectory name (train or test)
        subdir = "train" if usage == "Training" else "test"
        usage_dir = os.path.join(data_dir, subdir)

        if not os.path.exists(usage_dir):
            raise FileNotFoundError(f"Data directory not found: {usage_dir}")

        # Get names of the two target classes
        class_a_name = self._get_emotion_name_by_idx(class_a)
        class_b_name = self._get_emotion_name_by_idx(class_b)

        # Collect all image paths and labels
        all_samples = []

        for class_idx, emotion_name in [(class_a, class_a_name), (class_b, class_b_name)]:
            emotion_dir = os.path.join(usage_dir, emotion_name)
            if not os.path.exists(emotion_dir):
                print(f"Warning: class directory not found: {emotion_dir}")
                continue

            # Get all images for this class
            for filename in os.listdir(emotion_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_dir, filename)
                    # Map label: class_a->0, class_b->1
                    label = 0 if class_idx == class_a else 1
                    all_samples.append((img_path, label))

        # Bugfix: use random.sample instead of head to avoid biased sampling
        if max_samples_per_class is not None:
            # Group by class first
            samples_a = [s for s in all_samples if s[1] == 0]
            samples_b = [s for s in all_samples if s[1] == 1]

            # Random sampling
            if len(samples_a) > max_samples_per_class:
                import random
                random.seed(42)
                samples_a = random.sample(samples_a, max_samples_per_class)
            else:
                samples_a = samples_a[:max_samples_per_class]

            if len(samples_b) > max_samples_per_class:
                import random
                random.seed(42)
                samples_b = random.sample(samples_b, max_samples_per_class)
            else:
                samples_b = samples_b[:max_samples_per_class]

            all_samples = samples_a + samples_b

        # Shuffle
        import random
        random.seed(42)
        random.shuffle(all_samples)

        # Save data
        self.image_paths = [s[0] for s in all_samples]
        self.labels = np.array([s[1] for s in all_samples])

        # Bugfix: pre-build Resize transform to avoid repeated instantiation
        self.resize_transform = None
        if img_size != 48:
            self.resize_transform = transforms.Resize((img_size, img_size))

        # Print dataset info
        print(f"=" * 60)
        print(f"FER2013 {usage} Dataset (Binary)")
        print(f"Data directory: {data_dir}")
        print(f"Selected classes: {class_a} -> {self.get_emotion_name(class_a)} "
              f"vs {class_b} -> {self.get_emotion_name(class_b)}")
        print(f"Total samples: {len(self.image_paths)}")
        print(f"Class {class_a} ({self.get_emotion_name(class_a)}): {np.sum(self.labels == 0)}")
        print(f"Class {class_b} ({self.get_emotion_name(class_b)}): {np.sum(self.labels == 1)}")
        print(f"Image size: {img_size}x{img_size}, Channels: {in_channels}")
        print(f"=" * 60)

    def _get_emotion_name_by_idx(self, idx):
        """Get emotion name by index (lowercase, for folder matching)"""
        for name, i in self.EMOTION_TO_IDX.items():
            if i == idx:
                return name  # Return lowercase; folder names are lowercase
        return f"Unknown({idx})"

    @staticmethod
    def get_emotion_name(emotion_idx):
        """Get emotion class name"""
        return FER2013Binary.EMOTION_NAMES.get(emotion_idx, f"Unknown({emotion_idx})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        # Convert to grayscale
        img = img.convert('L')  # 'L' = grayscale

        # Data augmentation: only apply when training and augmentation != 'none'
        if self.usage == 'Training' and self.augmentation != 'none':
            img = self._apply_augmentation(img)

        # Convert to Tensor [1, H, W]
        img = transforms.ToTensor()(img)  # Auto-normalize to [0, 1]

        # Replicate channel based on in_channels parameter
        if self.in_channels == 3:
            # Expand single channel to 3 channels (replicate 3 times)
            img = img.repeat(3, 1, 1)
        # When in_channels == 1, keep single channel, no processing

        # Bugfix: use pre-built Resize transform
        if self.resize_transform is not None:
            img = self.resize_transform(img)

        # FER2013 standard normalization (pixel 0-1, mapped to [-1, 1])
        img = (img - 0.5) / 0.5  # [0, 1] -> [-1, 1]

        # Bugfix: convert label to int to ensure correct DataLoader collation
        label = int(self.labels[idx])

        return img, label

    def _apply_augmentation(self, img):
        """
        Apply data augmentation

        Augmentation strength:
        - 'none': no augmentation
        - 'weak': random horizontal flip (p=0.5)
        - 'medium': flip + slight rotation (+-10 degrees)
        - 'strong': flip + rotation + brightness/contrast + random crop
        """
        if self.augmentation == 'weak':
            # Weak augmentation: horizontal flip only
            if torch.rand(1).item() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        elif self.augmentation == 'medium':
            # Medium augmentation: flip + slight rotation
            if torch.rand(1).item() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Random rotation +-10 degrees
            angle = torch.empty(1).uniform_(-10, 10).item()
            img = img.rotate(angle)

        elif self.augmentation == 'strong':
            # Strong augmentation: flip + rotation + brightness/contrast + random crop
            if torch.rand(1).item() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Random rotation +-15 degrees
            angle = torch.empty(1).uniform_(-15, 15).item()
            img = img.rotate(angle)

            # Brightness adjustment (0.8 - 1.2x)
            from PIL import ImageEnhance
            brightness_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)

            # Contrast adjustment (0.8 - 1.2x)
            contrast_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)

            # Random crop (pad then crop)
            if torch.rand(1).item() < 0.5:
                # Original size 48, pad to 56 then crop back to 48
                padding = 4
                # Convert PIL image to tensor for padding, then back to PIL
                img_tensor = transforms.ToTensor()(img)
                img_tensor = transforms.functional.pad(img_tensor, padding, fill=0)
                img = transforms.ToPILImage()(img_tensor)
                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(img, (48, 48))
                img = transforms.functional.crop(img, i, j, h, w)

        return img


def load_fer2013_binary(data_dir="./data/FER2013/data", usage='Training',
                         class_a=3, class_b=5, max_samples_per_class=2000,
                         img_size=48, in_channels=1, augmentation='none'):
    """
    Load FER2013 binary classification dataset

    Args:
        data_dir: FER2013 data root directory (containing train and test subdirectories)
        usage: dataset split ('Training' or 'Test')
        class_a: first class index (mapped to label 0)
        class_b: second class index (mapped to label 1)
        max_samples_per_class: max samples per class (default 2000 train/1000 test)
        img_size: target image size (48=native, other values will resize)
        in_channels: output channels (1=grayscale, 3=replicate to RGB, default=1)
        augmentation: augmentation strength ('none', 'weak')

    Returns:
        FER2013Binary dataset object

    Usage example:
        # Load Happy vs Sad binary dataset (default 500 train + 200 test)
        train_dataset = load_fer2013_binary(
            data_dir="./data/FER2013/data",
            usage='Training',
            class_a=3,  # Happy
            class_b=4,  # Sad
            # max_samples_per_class defaults to 250
            img_size=48,
            in_channels=1
        )
    """
    return FER2013Binary(data_dir=data_dir, usage=usage,
                         class_a=class_a, class_b=class_b,
                         max_samples_per_class=max_samples_per_class,
                         img_size=img_size, in_channels=in_channels,
                         augmentation=augmentation)


def get_fer2013_transforms(train=True, img_size=48, in_channels=1):
    """
    Get FER2013 data augmentation/preprocessing transforms

    Note: FER2013Binary class already handles normalization and resize internally;
          this function is mainly for additional augmentation scenarios

    Args:
        train: whether in training mode (training mode adds augmentation)
        img_size: target image size
        in_channels: output channels

    Returns:
        transforms.Compose object
    """
    transform_list = []

    # Add data augmentation in training mode
    if train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.RandomRotation(degrees=10),    # Random rotation +-10 degrees
        ])

    transform_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels)
    ])

    return transforms.Compose(transform_list)


if __name__ == "__main__":
    """Test the data loader"""
    print("Testing FER2013 data loader\n")

    data_dir = "./data/FER2013/data"

    if os.path.exists(data_dir):
        # Test training set loading
        train_dataset = load_fer2013_binary(
            data_dir=data_dir,
            usage='Training',
            class_a=3,  # Happy
            class_b=4,  # Sad
            max_samples_per_class=100,
            img_size=48,
            in_channels=1
        )

        print(f"\nDataset size: {len(train_dataset)}")

        # Test getting a single sample
        img, label = train_dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Label: {label}")
        print(f"Image value range: [{img.min():.3f}, {img.max():.3f}]")

        # Test batch loading
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        batch_img, batch_label = next(iter(train_loader))
        print(f"\nBatch image shape: {batch_img.shape}")
        print(f"Batch label shape: {batch_label.shape}")
        print(f"Batch label values: {batch_label.tolist()}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please download FER2013 dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
