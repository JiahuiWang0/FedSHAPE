# FedSHAPE: Sharpness-Guided Trajectory Harmonization and Parameter Equalization

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This is the official PyTorch implementation for the paper: **"FedSHAPE: Sharpness-Guided Trajectory Harmonization and Parameter Equalization for Federated Learning Under Data Scarcity and Domain Skew"** (FGCS).

## Abstract

Federated learning (FL) has emerged as a promising paradigm for privacy-preserving collaborative training. However, under the intertwined challenges of severe data scarcity (e.g., 1% sampling rate) and non-IID domain skew, existing FL approaches often experience substantial performance degradation. To address this issue, we introduce FedSHAPE (Federated Sharpness-aware Harmonized Aggregation and Parameter Equalization), a principled framework combining SPO and HAE. By systematically investigating the performance bottlenecks in such restrictive settings, we identify two coupled factors: standard local optimizers rapidly converge into domain-specific sharp minima (severe overfitting), rendering subsequent global feature alignment biased and suboptimal. Accordingly, FedSHAPE formulates a complete spatial-temporal optimization pipeline: it first executes spatial geometry purification at the client level to suppress local overfitting noise, establishing the strict mathematical prerequisite for the subsequent temporal trajectory harmonization and diversity-aware global feature alignment. Extensive experiments demonstrate that FedSHAPE consistently improves both generalization robustness and cross-domain fairness under highly restricted data conditions.

##  Overview

Federated Learning (FL) faces catastrophic generalization gaps when confronted with the intertwined challenges of **extreme data scarcity** (e.g., 1% sampling rate) and **domain skew**. Under these constraints, standard local optimizers rapidly descend into domain-specific sharp minima, causing global feature alignment to be hijacked by persistent, highly-biased noise.


**Core Requirements:**
- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- pandas
- numpy
- scikit-learn

##  Repository Structure

```text
├── backbone/               # Network architectures (CNN for Digits, ResNet-10 for Office-Caltech)
├── datasets/               # Dataloaders and non-IID partitioning logic
├── models/                 # FL algorithms (FedAvg, FedProx, FedDyn, MOON, FedSHAPE, etc.)
├── utils/                  # Helper functions, hyperparameter configs, and training loop
├── main.py                 # Main entry point for training
```

##  Quick Start

To reproduce the main experiments from the paper, use `main.py`. The framework supports multiple state-of-the-art baselines.

### 1. Run on Digits Dataset (1% Data Scarcity)
To train **FedSHAPE** on the Digits dataset (MNIST, USPS, SVHN, SYN) across 20 clients with 200 communication rounds:

```bash
python main.py \
    --dataset fl_digits \
    --model fedavgshape \
    --parti_num 20 \
    --communication_epoch 200 \
    --local_epoch 5 \
    --beta 0.6 \
    --threshold 0.5 \
    --seed 0
```

### 2. Run on Office-Caltech Dataset (10% Data Scarcity)
To train **FedSHAPE** on the more complex Office-Caltech dataset:

```bash
python main.py \
    --dataset fl_officecaltech \
    --model fedavgshape \
    --parti_num 20 \
    --communication_epoch 200 \
    --local_epoch 5 \
    --beta 0.6 \
    --threshold 0.4 \
    --seed 0
```

### 3. Run Baselines
You can easily switch the `--model` argument to evaluate other baselines included in our study:
* `fedavg` (FedAvg)
* `fedprox` (FedProx)
* `feddyn` (FedDyn)
* `moon` (MOON)
* `fedproto` (FedProto)
* `afl` (AFL)
* `qffl` (q-FFL)
* `ditto` (Ditto)
* `fedfa` (FedFA)

Example for running FedProx:
```bash
python main.py --dataset fl_digits --model fedprox --mu 0.01 --seed 0
```

##  Hyperparameter Configuration

To ensure strict reproducibility, optimal hyperparameters for all methods are pre-configured in `utils/best_args.py`. 
Key hyperparameters for **FedSHAPE** include:
* `--beta`: Temporal momentum retention for the EMA mask (Optimal: `0.6`).
* `--threshold` (or $\tau$): Temperature scaling factor for Parameter Equalization (Optimal: `0.5` for Digits, `0.4` for Office-Caltech).
* `wp_alpha` (or $\rho$): SPO exploration radius (Optimal: `0.05` configured via `best_args.py`).

##  Citation

If you find this code or our paper useful in your research, please consider citing our work:

```bibtex
@article{wang2026fedshape,
  title={FedSHAPE: Sharpness-Guided Trajectory Harmonization and Parameter Equalization for Federated Learning Under Data Scarcity and Domain Skew},
  author={Wang, Jiahui and Chen, Zheyi and Feng, Hailin and Liu, Xing and Hua, Kun and Lu, Jia},
  journal={Future Generation Computer Systems},
  year={2026}
}
```

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
