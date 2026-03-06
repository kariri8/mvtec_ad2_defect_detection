#!/usr/bin/env python3
"""
Train a single experiment.

Usage:
    python scripts/train.py --exp 1 --config configs/experiment1.yaml
    python scripts/train.py --exp 2 --config configs/experiment2.yaml
    python scripts/train.py --exp 3 --config configs/experiment3.yaml
    python scripts/train.py --exp all --config configs/final_run.yaml
"""

import argparse
import logging
import random

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly detection experiments.")
    parser.add_argument("--exp", required=True, choices=["1", "2", "3", "all"],
                        help="Which experiment to run.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    if args.exp == "1":
        from src import experiment1
        for cat in cfg["categories"]:
            experiment1.train(cat, cfg)

    elif args.exp == "2":
        from src import experiment2
        for cat in cfg["categories"]:
            experiment2.train(cat, cfg)

    elif args.exp == "3":
        from src import experiment3
        for cat in cfg["categories"]:
            experiment3.train(cat, cfg)

    elif args.exp == "all":
        from src.train_all import main as run_all
        run_all(cfg)


if __name__ == "__main__":
    main()
