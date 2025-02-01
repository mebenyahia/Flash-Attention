# Flash-Transformer

![Description](https://tfwiki.net/mediawiki/images2/a/aa/FlashG1-card.jpg)

Flash-Transformer is a repository for training a Transformer-based sequence-to-sequence model for machine translation for the WMT14 dataset. The implementation builds upon the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper and uses **FlashAttention** for fast, memory-efficient attention computation. The model is designed to efficiently handle long sequences and large batch sizes using GPU-optimized kernels and mixed-precision training.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running on Colab via Notebook](#running-on-colab-via-notebook)
  - [Running via Shell Scripts](#running-via-shell-scripts)
- [Methodology](#methodology)
- [References](#references)
- [License](#license)
- [Contact](#contact)

## Features

- **FlashAttention**: GPU-optimized, memory-efficient replacement for vanilla scaled dot-product attention.
- **Mixed-Precision Training**: Uses PyTorch AMP for faster training and lower memory consumption.
- **Modular and Scalable Code**: Separate data processing, model definition, training, evaluation, and visualization.
- **Evaluation & Visualization**: Scripts for BLEU score computation and attention visualization.
- **Flexible Configuration**: Centralized management of hyperparameters and settings in `config/config.yaml`. This is the most important file.

## Repository Structure

```
flash-transformer/
├── README.md                # Overview and usage instructions
├── requirements.txt         # Dependencies
├── config/
│   └── config.yaml          # >> To configure pretty much everything
├── data/
│   ├── preprocess.py        # Tokenization & BPE
│   ├── download_dataset.py  # Download & preprocess WMT 2014 dataset
│   └── tokenized/           # Folder for tokenized datasets
├── src/
│   ├── model/
│   │   ├── transformer.py   # Transformer model
│   │   ├── flash_attention.py  # FlashAttention implementation
│   │   ├── encoder.py       # Encoder block
│   │   ├── decoder.py       # Decoder block
│   │   ├── embeddings.py    # Token & positional embeddings
│   │   └── utils.py         # Helper functions
│   ├── training/
│   │   ├── train.py         # Training loop
│   │   ├── optimizer.py     # Optimizer & scheduler
│   │   └── regularization.py # Dropout & label smoothing
│   ├── evaluation/
│   │   ├── evaluate.py      # BLEU score computation >> uses the max checkpoint defined in the config
│   │   ├── inference.py     # Translation generation
│   │   └── metrics.py       # Accuracy metrics
│   ├── visualization/
│   │   ├── plot_attention.py  # Attention map visualization
│   │   ├── training_curves.py # Training loss/accuracy plots
│   │   └── embedding_visuals.py # Token embedding visualization
├── notebooks/
│   ├── demo.ipynb           # >> Demo notebook
│   ├── exploration.ipynb    # Analysis & visualization notebook
├── scripts/
│   ├── train.sh             # Training script
│   ├── evaluate.sh          # Evaluation script
│   └── preprocess.sh        # Preprocessing script
├── tests/
│   ├── test_model.py        # Transformer & FlashAttention tests
│   ├── test_training.py     # Training pipeline tests
│   └── test_evaluation.py   # Evaluation function tests
└── checkpoints/             # Saved model checkpoints
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mebenyahia/flash-attention.git
   cd flash-transformer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running on Colab via Notebook

A demonstration notebook is provided in `notebooks/demo.ipynb`. This notebook:

- Downloads and preprocesses the dataset.
- Trains the Transformer model using FlashAttention.
- Evaluates the model and visualizes BLEU scores and token embeddings.
- Runs the full pipeline end-to-end.

**To run in Colab:**

1. Upload the repository to Colab or mount Google Drive.
2. Run `notebooks/demo.ipynb` after editing the paths.

### Running via Shell Scripts

Alternatively, locally, you can run the code using shell scripts:

- **Preprocess data:**
  
  ```bash
  bash scripts/preprocess.sh
  ```

- **Train the model (e.g., for 50 epochs):**
  
  ```bash
  bash scripts/train.sh
  ```

- **Evaluate the model and generate visualizations:**
  
  ```bash
  bash scripts/evaluate.sh
  ```

## Methodology

### FlashAttention
Instead of computing the full  N × N attention matrix, FlashAttention computes attention in a block-wise, memory-efficient manner, especially for long sequences. More details can be found in [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).

### Mixed-Precision Training
Uses PyTorch AMP for FP16 computations.

### Evaluation and Visualization
Evaluation scripts compute BLEU scores at both corpus and sentence levels. Visualizations, including PCA projections of token embeddings.

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *arXiv preprint arXiv:2205.14135*.

## License

This repository is released under the MIT License.

## Contact

For any inquiries, please reach out to me, or open an issue on Github.

