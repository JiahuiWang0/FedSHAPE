# FedSHAPE: Sharpness-Guided Trajectory Harmonization and Parameter Equalization

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository provides the PyTorch implementation and experiment records for **“FedSHAPE: Sharpness-Guided Trajectory Harmonization and Parameter Equalization for Federated Learning Under Data Scarcity and Domain Skew”**, submitted to *Future Generation Computer Systems*.

## Overview

FedSHAPE studies federated learning under data scarcity and non-IID domain skew. It combines Stable Parameter Optimization (SPO) during local training with Harmonized Aggregation and Equalization (HAE) during server aggregation. SPO produces local model updates, and HAE uses their temporal consistency to guide aggregation and parameter equalization. This is the algorithmic sequence evaluated in the paper; component comparisons report the empirical behavior of its modules.

## Reported Results

The paper evaluates FedSHAPE on Digits with a 1% sampling rate and Office-Caltech with a 10% sampling rate. In the reported comparisons, FedSHAPE achieves the highest average accuracy among the listed methods on both datasets: 84.98% on Digits and 66.79% on Office-Caltech. Its cross-domain standard deviations are 16.29 and 4.59, respectively.

The repository includes the implementation, experiment configurations, and experiment logs used to produce the reported results. The `Logs/` directory contains the recorded run outputs, including round-level results and system metrics used in the runtime and memory analyses.

The overhead comparison reports both runtime and mean peak GPU-memory allocation. FedSHAPE uses 4089.04 MiB on Digits and 4858.26 MiB on Office-Caltech; the corresponding FedHEAL measurements are 4079.48 MiB and 4843.14 MiB. FedSHAPE reaches the reported accuracy thresholds in less time than the baseline averages on both datasets, while its per-round runtime is lower than the baseline average on Digits and higher on Office-Caltech. These measurements describe the observed memory, runtime, and accuracy trade-off under the evaluated configurations.

## Requirements

- Python 3.8 or later
- PyTorch 2.0 or later
- torchvision
- pandas
- numpy
- scikit-learn

## Repository Structure

```text
├── backbone/               # Network architectures for Digits and Office-Caltech
├── datasets/               # Dataset loaders and non-IID partitioning logic
├── models/                 # Federated learning algorithms, including FedSHAPE
├── utils/                  # Training utilities and experiment configurations
├── Logs/                   # Experiment outputs and system metrics
├── main.py                 # Main entry point
└── README.md
```

## Quick Start

The commands below run FedSHAPE with 20 clients, 200 communication rounds, five local epochs, and seed 0.

### Digits (1% sampling rate)

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

### Office-Caltech (10% sampling rate)

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

### Baselines

The following baselines are available through the `--model` argument:

- `fedavg` (FedAvg)
- `fedprox` (FedProx)
- `feddyn` (FedDyn)
- `moon` (MOON)
- `fedproto` (FedProto)
- `afl` (AFL)
- `qffl` (q-FFL)
- `ditto` (Ditto)
- `fedfa` (FedFA)

For example, run FedProx on Digits with:

```bash
python main.py --dataset fl_digits --model fedprox --mu 0.01 --seed 0
```

## Experiment Configuration

Recommended settings for reproducing the paper experiments are provided in `utils/best_args.py`. The principal FedSHAPE parameters are:

- `--beta`: EMA momentum for the temporal trajectory (0.6)
- `--threshold` (or $\tau$): equalization temperature (0.5 for Digits and 0.4 for Office-Caltech)
- `wp_alpha` (or $\rho$): SPO exploration radius (0.05)

The experiment logs record the configurations and outcomes used in the paper's tables. They can be consulted alongside the training scripts to trace reported values to their runs.

## Citation

```bibtex
@article{wang2026fedshape,
  title={FedSHAPE: Sharpness-Guided Trajectory Harmonization and Parameter Equalization for Federated Learning Under Data Scarcity and Domain Skew},
  author={Wang, Jiahui and Chen, Zheyi and Feng, Hailin and Liu, Xing and Hua, Kun and Lu, Jia},
  journal={Future Generation Computer Systems},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
