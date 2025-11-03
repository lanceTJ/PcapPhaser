# Network Traffic Dataset Generation System

## Overview
This project implements a high-performance dataset generation system for network traffic based on phase segmentation using the PSS algorithm. It processes PCAP files, performs phase division, feature extraction, fusion, and labeling to output CSV datasets for machine learning tasks, following the provided documentation.

## Features
- Modular design with decoupled stages for reusability and fault tolerance.
- Supports multi-phase segmentation (p values) and feature fusion.
- Persistent intermediate results in a hierarchical directory structure.
- GPU/Numba acceleration for PSS matrix computations.
- Automated labeling based on CIC-IDS2018 rules.

## Installation
1. Clone the repository: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Manually install CicFlowMeter (external tool) and set path in config.ini.
4. Prepare data directory with PCAP files under data/day1/, etc.

## Usage
- Configure settings in `configs/config.ini`.
- Run the pipeline: `python run.py --config configs/config.ini --stages all`
- For specific steps: `python run.py --stage feature_extractor --day day1`
- Run tests: `pytest tests/`

## Directory Structure
- `src/`: Source code modules and utilities.
- `tests/`: Unit tests for each module.
- `configs/`: Configuration files.
- `data/`: Runtime data storage (e.g., feature_matrix, datasets).
- `docs/`: Project documentation.

## License
MIT License