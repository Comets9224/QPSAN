# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_tiny_config():
    """Returns a tiny ViT configuration for small datasets."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_tiny_1layer_config():
    """Returns a tiny ViT configuration with only 1 transformer layer."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_fashion_mnist_config():
    """Returns a ViT configuration for FashionMNIST (28x28 grayscale images)."""
    config = ml_collections.ConfigDict()
    # patch size 7x7, 28/7=4, yields 4x4=16 patches
    config.patches = ml_collections.ConfigDict({'size': (7, 7)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_fashion_mnist_schemeA_config():
    """
    Returns a ViT configuration for Scheme A (seq_len=4).

    Uses 14x14 images (downsampled from 28x28) with 7x7 patches, yielding 2x2=4 patches + 1 CLS = 5 tokens.
    Scheme A's quantum circuit has a fixed seq_len=4 (2-bit index register), so only 4-patch attention is computed.

    Note: img_size must be 14, not 28, to produce 4 patches.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (7, 7)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_classical_fashion_mnist_config():
    """
    Returns a Classical ViT configuration for FashionMNIST (28x28 images).

    Note: uses 7x7 patch_size to match quantum model for fair comparison.
    Sequence length: (28/7)^2 = 16 patches + 1 CLS = 17 tokens
    """
    config = ml_collections.ConfigDict()
    # patch size 7x7, matching quantum model
    config.patches = ml_collections.ConfigDict({'size': (7, 7)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_cifar10_tiny_config():
    """
    Returns an optimized ViT configuration for CIFAR-10 binary classification.

    Optimized over the original config:
    - num_layers: 1 -> 4 (deeper network)
    - patch_size: 8x8 (balance compute efficiency and feature granularity)
    - hidden_size: 192 -> 256 (stronger expressiveness)
    - num_heads: 3 -> 4 (match hidden_size, 256/4=64)
    - mlp_dim: 768 -> 1024 (stronger MLP)

    CIFAR-10: 32x32 RGB, binary classification
    patch_size: 8x8, yields (32/8)^2 = 16 patches
    Sequence length: 16 patches + 1 CLS = 17 tokens

    Args:
        None (fixed config for binary classification)

    Returns:
        ml_collections.ConfigDict
    """
    config = ml_collections.ConfigDict()
    # patch size 8x8, 32/8=4, yields 4x4=16 patches
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 4  # 256/4=64, evenly divisible
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_cifar10_4x4_config():
    """
    Returns the LEGACY ViT configuration for CIFAR-10 with 4x4 patches.

    For loading checkpoints trained before 2026-03-14 (when patch_size changed from 4x4 to 8x8).
    Matches the original get_cifar10_tiny_config() at that time:
    - patch_size: 4x4, yields (32/4)^2 = 64 patches
    - Sequence length: 64 patches + 1 CLS = 65 tokens
    - hidden_size: 256, num_heads: 4, num_layers: 4

    Note: for new experiments, use get_cifar10_tiny_config() (8x8 patches)

    Returns:
        ml_collections.ConfigDict
    """
    config = ml_collections.ConfigDict()
    # legacy: 4x4 patch, 32/4=8, yields 8x8=64 patches
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_jaffe_config():
    """
    Returns a ViT configuration for JAFFE 7-class facial expression recognition.

    JAFFE: 32x32 grayscale images, 7 expression classes
    - 7 expressions: neutral, happy, sad, surprise, anger, disgust, fear
    - patch_size: 8x8, yields (32/8)^2 = 16 patches
    - Sequence length: 16 patches + 1 CLS = 17 tokens

    Optimization notes (2026-02-17 - small dataset, low lr):
    - patch_size: 8x8 (balance capacity and overfitting)
    - num_layers: 1 (shallow network for small dataset)
    - mlp_dim: 768 (original)
    - dropout_rate: 0.1 (restored to original)
    - attention_dropout_rate: 0.0 (restored to original)
    - Recommended lr: 1e-4 (lower for gentler training)

    Args:
        None (fixed config for 7-class)

    Returns:
        ml_collections.ConfigDict
    """
    config = ml_collections.ConfigDict()
    # patch size 8x8, 32/8=4, yields 4x4=16 patches
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_fer2013_config():
    """
    Returns the optimized ViT configuration for FER2013 facial expression recognition.

    FER2013: 48x48 grayscale images, 7 expression classes
    - 7 expressions: angry, disgust, fear, happy, sad, surprise, neutral
    - patch_size: 8x8, yields (48/8)^2 = 36 patches
    - Sequence length: 36 patches + 1 CLS = 37 tokens

    Config notes (2026-02-19 - Medium config):
    - num_layers: 2 (increased depth, best balance)
    - num_heads: 6 (more attention heads)
    - hidden_size: 192 (divisible by 6, 192/6=32)
    - mlp_dim: 1024 (stronger MLP)
    - dropout_rate: 0.15 (increased regularization)

    Experiment results (Happy vs Surprise, 2000 samples):
    - Accuracy: ~80%
    - Parameters: ~1.6M

    Args:
        None (fixed config for FER2013)

    Returns:
        ml_collections.ConfigDict
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 192  # 192/6=32
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 6  # 192/6 = 32
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.15
    config.classifier = "token"
    config.representation_size = None
    return config


def get_fer2013_7class_config():
    """
    Returns the ViT configuration for FER2013 7-class balanced classification.

    Same architecture as get_fer2013_config(), only num_classes=7 (passed in training script).
    Separate function for documentation and future multi-class optimization.

    FER2013 7-class: 48x48 grayscale, 7 expressions (balanced 250 per class)
    - patch_size: 8x8, (48/8)^2 = 36 patches + 1 CLS = 37 tokens
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 6
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.15
    config.classifier = "token"
    config.representation_size = None
    return config


def get_fer2013_3class_config():
    """
    Returns the ViT configuration for FER2013 3-class balanced classification.

    Same architecture as get_fer2013_config(), only num_classes=3 (passed in training script).
    Separate function for documentation and future multi-class optimization.

    FER2013 3-class: 48x48 grayscale, 3 expressions (Happy, Surprise, Disgust)
    - patch_size: 8x8, (48/8)^2 = 36 patches + 1 CLS = 37 tokens
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 6
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.15
    config.classifier = "token"
    config.representation_size = None
    return config


def get_dtd_config(img_size=64):
    """
    Returns a ViT configuration for DTD (Describable Textures Dataset).

    DTD: RGB texture images, resized to target dimensions
    - Default: 64x64 (development), optional: 224x224 (full experiment)
    - Binary classification: cracked(7) vs bubbly(3) or other texture pairs
    - 3-channel RGB input

    Config notes:
    - patch_size: fixed 12x12
      - 64x64: 5x5=25 patches (last patch with padding)
      - 96x96: 8x8=64 patches
      - 224x224: 19x19=361 patches (last patch with padding)
    - num_layers: 2 (balance capacity and compute)
    - num_heads: 4
    - hidden_size: 256
    - mlp_dim: 1024

    Args:
        img_size: image size, 64/96/224 etc (default 64), patch_size fixed at 12

    Returns:
        ml_collections.ConfigDict
    """
    config = ml_collections.ConfigDict()

    # fixed patch_size=12
    config.patches = ml_collections.ConfigDict({'size': (12, 12)})
    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 4  # 256/4=64
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_fer2013_gs_config():
    """
    Returns a ViT configuration for Gaussian Fuzzy Attention (ablation variant).

    Gaussian fuzzy attention vs quantum fuzzy attention ablation:
    - Quantum: 5 trainable params (enc_scale, gamma_diff, gamma_sum, alpha, beta)
    - Gaussian: 1 trainable param (sigma)
    - Both share the same model architecture config

    Config notes:
    - Identical architecture to get_fer2013_config()
    - patch_size: 8x8, 36 patches, 37 tokens
    - hidden_size: 192, num_heads: 6, num_layers: 2
    - mlp_dim: 1024, dropout_rate: 0.15

    Returns:
        ml_collections.ConfigDict
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 6
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.15
    config.classifier = "token"
    config.representation_size = None
    return config
