# QPSAN: Quantum Parameterized Self-Attention Network

PyTorch implementation of QPSAN, a Vision Transformer that replaces dot-product attention with a parameterized quantum circuit as the attention scoring function.

## Repository Structure

```
QPSAN/
├── models/                     # Model architectures
│   ├── VIT.py                  # Classical ViT (dot-product self-attention)
│   ├── QVIT.py                 # Quantum ViT (QPSAN, quantum self-attention)
│   ├── MLPVIT.py               # MLP-attention ViT (49-param & 585-param ablation)
│   ├── COSVIT.py               # Cosine-attention ViT
│   ├── LINERVIT.py             # Linear-attention ViT
│   ├── quantum_attention.py    # QAOACircuit + QuantumAttention (core quantum circuit)
│   ├── mlp_attention.py        # MLP scoring functions for ablation
│   └── configs.py              # Model configuration for each dataset
│
├── TRAIN/                      # Training scripts (organized by dataset)
│   ├── Cifar10/                #   CIFAR-10: ViT + QViT
│   ├── DirtyMNIST/             #   DirtyMNIST: ViT + QViT
│   ├── FashionMNIST/           #   FashionMNIST: ViT + QViT
│   └── FER2013/                #   FER2013: ViT + QViT + CosViT + LinearViT + MLP-49 + MLP-585
│
├── data/                       # Dataset dataloaders (data not included)
│   ├── Cifar10/dataloader.py
│   ├── DirtyMNIST/dataloader.py
│   ├── FashionMNIST/dataloader.py
│   └── FER2013/dataloader.py
│
├── utils/                      # Training utilities
│   ├── path_utils.py           # Output path management
│   ├── early_stopping.py       # Early stopping callback
│   ├── scheduler.py            # Learning rate scheduler
│   ├── training_plotter.py     # Loss/accuracy curve plotting
│   ├── statistical_analysis.py # Statistical analysis tools
│   └── dist_util.py            # Distributed training utilities
│
├── requirements.txt
└── LICENSE
```

## Quick Start

### 1. Create conda environment

```bash
conda create -n vit python=3.10 -y
conda activate vit
```

### 2. Install qiskit first (must use --no-deps)

Qiskit packages have dependency conflicts with other packages, so install them separately first.

```bash
pip install --no-deps \
  qiskit==0.38.0 \
  qiskit-ibm-runtime==0.20.0 \
  qiskit-ibmq-provider==0.19.2 \
  qiskit-terra==0.21.2 \
  qiskit-aer==0.11.0 \
  retworkx==0.17.1 \
  rustworkx==0.17.1 \
  tweedledum==1.1.1
```

### 3. Install PyTorch (CUDA 12.6)

```bash
pip install torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU access:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.10.0+cu126 / True
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Install torchquantum

torchquantum is not in requirements.txt and must be installed separately.

```bash
pip install torchquantum
```

### Training

```bash
# Classical ViT on FashionMNIST
python TRAIN/FashionMNIST/TRAIN_ViT_FashionMNIST.py --name vit_fashion --num_runs 20

# QPSAN (Quantum ViT) on FashionMNIST
python TRAIN/FashionMNIST/TRAIN_QVIT_FashionMNIST.py --name qvit_fashion --num_runs 20

# Full FER2013 comparison (6 models)
python TRAIN/FER2013/TRAIN_ViT_FER2013.py --name vit_fer --num_runs 20
python TRAIN/FER2013/TRAIN_QVIT_FER2013.py --name qvit_fer --num_runs 20
python TRAIN/FER2013/TRAIN_COSViT_FER2013.py --name cosvit_fer --num_runs 20
python TRAIN/FER2013/TRAIN_LINERViT_FER2013.py --name linearvit_fer --num_runs 20
python TRAIN/FER2013/TRAIN_ViT_FER2013_MLP.py --name mlp49_fer --num_runs 20
python TRAIN/FER2013/TRAIN_ViT_FER2013_MLP585.py --name mlp585_fer --num_runs 20
```

## Core Idea

Classical: `attention_scores = Q @ K^T / sqrt(head_dim)`

QPSAN: RY gates encode (q-k difference, q+k sum) -> bidirectional CNOT entanglement -> joint measurement P(|00>) + P(|11>) as attention score, naturally bounded in [0, 1].

## Citation

<!-- Add your paper BibTeX here once published -->
```bibtex
@article{TODO,
  title={TODO},
  author={TODO},
  journal={TODO},
  year={TODO}
}
```

## References

<!-- Add your reference bibliography here -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
