#!/usr/bin/env python3
"""
Evaluate a trained experiment and print results.

Usage:
    python scripts/evaluate.py --exp 1 --config configs/experiment1.yaml
    python scripts/evaluate.py --exp 2 --config configs/experiment2.yaml
    python scripts/evaluate.py --exp 3 --config configs/experiment3.yaml
"""

import argparse
import json
import logging
import os

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection experiments.")
    parser.add_argument("--exp", required=True, choices=["1", "2", "3"],
                        help="Which experiment to evaluate.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    all_results = []

    if args.exp == "1":
        from src import experiment1
        from transformers import ViTMAEForPreTraining
        for cat in cfg["categories"]:
            final_dir = os.path.join(cfg["checkpoint_dir"], f"exp1_{cat}", "final")
            model = ViTMAEForPreTraining.from_pretrained(final_dir).to(cfg["device"])
            res = experiment1.evaluate(model, cat, cfg)
            res["category"] = cat
            all_results.append(res)

    elif args.exp == "2":
        from src import experiment2
        for cat in cfg["categories"]:
            backbone, predictor = experiment2.train(cat, cfg)  # loads existing ckpt
            res = experiment2.evaluate(backbone, predictor, cat, cfg)
            res["category"] = cat
            all_results.append(res)

    elif args.exp == "3":
        from src import experiment3
        for cat in cfg["categories"]:
            backbone, predictor = experiment3.train(cat, cfg)  # loads existing ckpt
            res = experiment3.evaluate(backbone, predictor, cat, cfg)
            res["category"] = cat
            all_results.append(res)

    # Print summary table
    print(f"\n{'Category':<15} | {'AU-PRO':>8} | {'SegF1':>8} | {'k':>4}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['category']:<15} | {r['au_pro']:>8.4f} | {r['seg_f1']:>8.4f} | {r['best_k']:>4}")

    # Save to results/
    os.makedirs("results", exist_ok=True)
    out_path = f"results/exp{args.exp}_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
