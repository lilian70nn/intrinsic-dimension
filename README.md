# Intrinsic Dimension Analysis in Neural Networks

This repository contains code for reproducing and analyzing the dynamics of intrinsic dimension in various neural network architectures, including convolutional neural networks (CNNs), tabular models, and transformer-based language models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Key Components](#key-components)
- [References](#references)

## Overview

Intrinsic dimension (ID) refers to the minimal number of coordinates needed to represent a dataset on its underlying manifold. This project explores how intrinsic dimension evolves across different layers and during training in various model architectures.

The codebase reproduces experiments from research on intrinsic dimension dynamics, providing implementations for:

- **Task 1**: Computing intrinsic dimension across layers of pretrained CNN models (VGG, ResNet, AlexNet) on ImageNet data, and analyzing intrinsic dimension dynamics during training on CIFAR-10
- **Task 2**: Analyzing intrinsic dimension in tabular models (MLPs) trained on datasets like Adult
- **Task 3**: Investigating intrinsic dimension in transformer models (RoBERTa) on text classification tasks and synthetic data experiments

## Features

- Multiple intrinsic dimension estimators (TwoNN, MLE)
- Support for various model architectures (CNNs, MLPs, Transformers)
- Comprehensive plotting functions for reproducing paper figures
- Modular codebase with separate modules for models, datasets, and analysis
- Jupyter notebooks for interactive exploration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lilian70nn/intrinsic-dimension.git
cd intrinsic-dimension
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Task 3_2, download required NLTK data:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Usage

### Running Tasks

Each task can be run independently using the provided Python scripts:

```bash
# Task 1: CNN intrinsic dimension analysis
python tasks/task1.py

# Task 2: Tabular model intrinsic dimension analysis
python tasks/task2.py

# Task 3: NLP model intrinsic dimension analysis, synthetic data experiments
python tasks/task3.py
```

### Using Jupyter Notebooks

Interactive notebooks are available for each task in the `notebooks/` directory, These notebooks provide a structured, end-to-end view of each experiment, including
code execution, visualizations, and qualitative observations of the results

- `Task1.ipynb`: CNN analysis
- `Task2.ipynb`: Tabular model analysis
- `Task3.ipynb`: NLP analysis, synthetic data experiments
- `utils.ipynb`: Utility functions

### Command Line Arguments

Tasks support various command line arguments. For example:

```bash
python tasks/task1.py --task1_1 --savepath figures/task1/
```

Use `--help` to see available options for each task.

## Project Structure

```
├── README.md
├── requirements.txt
├── figures/                 # Output directory for plots
│   ├── task1/
│   ├── task2/
│   └── task3/
├── notebooks/               # Jupyter notebooks
│   ├── Task1.ipynb
│   ├── Task2.ipynb
│   ├── Task3.ipynb
│   └── utils.ipynb
├── src/                     # Source code
│   ├── __init__.py
│   ├── cnn_models.py        # CNN model definitions (CIFAR-10–adapted variants)
│   ├── datasets.py          # Dataset utilities
│   ├── depths.py            # Layer depth extraction
│   ├── plots.py             # Plotting functions
│   ├── postprocess.py       # Postprocessing utilities
│   ├── tabular_depths.py    # Tabular-model–specific layer depth extraction
│   ├── tabular_models.py    # Tabular model definitions
│   ├── tabular_preprocessing.py
│   ├── train_and_test.py    # Training utilities
│   └── id/                  # Intrinsic dimension computation
│       ├── __init__.py
│       ├── compute.py
│       └── estimator.py
└── tasks/                   # Task scripts
    ├── __init__.py
    ├── task1.py
    ├── task2.py
    └── task3.py
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- numpy
- scipy
- matplotlib
- tqdm
- transformers
- datasets
- scikit-learn
- pillow
- pytorch-pretrained-biggan
- tabm
- nltk

## Key Components

### Intrinsic Dimension Estimators
- **TwoNN**: Two Nearest Neighbors estimator
- **MLE**: Maximum Likelihood Estimator

### Model Architectures
- **CNNs**: VGG, ResNet, AlexNet variants
- **Tabular**: RealMLP, StandardMLP, TabM
- **NLP**: RoBERTa-based models

### Datasets
- ImageNet, CIFAR-10
- Adult, Covertype, etc. (for tabular tasks)
- TweetEval, CNN/DailyMail (for NLP tasks), synthetic data


## References

This project reproduces and extends results from prior work on intrinsic dimension and MLPs

- Ansuini A, Laio A, Macke J H, et al. Intrinsic dimension of data representations in deep neural networks[J]. Advances in Neural Information Processing Systems, 2019, 32.

- Holzmüller D, Grinsztajn L, Steinwart I. Better by default: Strong pre-tuned mlps and boosted trees on tabular data[J]. Advances in Neural Information Processing Systems, 2024, 37: 26577-26658.

- Gorishniy Y, Kotelnikov A, Babenko A. TabM: Advancing tabular deep learning with parameter-efficient ensembling, 2025[J]. URL https://arxiv. org/abs/2410.24210.