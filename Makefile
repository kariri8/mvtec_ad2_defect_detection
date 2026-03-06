# ── MVTec AD2 Anomaly Detection ───────────────────────────────────────────────
# One-command shortcuts for training, evaluation, testing and linting.
#
# Prerequisites: pip install -r requirements.txt
# Data:          python scripts/download_data.py --archive <archive.zip> --dest data/

.PHONY: help install download train1 train2 train3 train-all eval1 eval2 eval3 test lint

help:
	@echo ""
	@echo "MVTec AD2 — available targets:"
	@echo "  make install        Install Python dependencies"
	@echo "  make download       Print data download instructions"
	@echo "  make train1         Train Experiment 1 (ViT-MAE pixel)"
	@echo "  make train2         Train Experiment 2 (DINOv3 + CNN)"
	@echo "  make train3         Train Experiment 3 (DINOv3 + Transformer)"
	@echo "  make train-all      Full-scale run across all 8 categories (2 GPUs)"
	@echo "  make eval1          Evaluate Experiment 1"
	@echo "  make eval2          Evaluate Experiment 2"
	@echo "  make eval3          Evaluate Experiment 3"
	@echo "  make test           Run unit tests"
	@echo "  make lint           Run ruff linter"
	@echo ""

install:
	pip install -r requirements.txt

download:
	python scripts/download_data.py

train1:
	python scripts/train.py --exp 1 --config configs/experiment1.yaml

train2:
	python scripts/train.py --exp 2 --config configs/experiment2.yaml

train3:
	python scripts/train.py --exp 3 --config configs/experiment3.yaml

train-all:
	python scripts/train.py --exp all --config configs/final_run.yaml

eval1:
	python scripts/evaluate.py --exp 1 --config configs/experiment1.yaml

eval2:
	python scripts/evaluate.py --exp 2 --config configs/experiment2.yaml

eval3:
	python scripts/evaluate.py --exp 3 --config configs/experiment3.yaml

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
