# GalaxyCBM-SR — per-stage task runner.
#
# Each target is a thin wrapper over a script in scripts/. Configs live in
# configs/ and are passed via Hydra-style overrides; no magic numbers in code.

PYTHON := uv run python
PYTEST := uv run pytest

CONFIG_DIR := configs

.PHONY: help sync test lint clean \
        data.raw data.concepts data.labels \
        stage1 stage2 stage3 \
        baselines robustness eval all

help:
	@grep -E '^[a-zA-Z0-9._-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

sync:  ## uv sync with dev extras (base install — no torch/zoobot)
	uv sync --extra dev

sync.stage1:  ## uv sync with Stage-1 modelling extras (torch, zoobot; requires macOS-arm64 or Linux)
	uv sync --extra dev --extra stage1

sync.baselines:  ## uv sync with P7 baseline extras (xgboost, shap → numba)
	uv sync --extra dev --extra baselines

sync.all:  ## uv sync with every extra
	uv sync --extra dev --extra stage1 --extra baselines

test:  ## Run the pytest suite
	$(PYTEST) -q

lint:  ## Ruff lint
	uv run ruff check src tests scripts

# ---------------- Data stages ----------------
data.raw:  ## P1 — download 3 GZ Evo primary shards (skip if already present)
	@ls data/raw/gz_evo/data/*.parquet >/dev/null 2>&1 && \
	  echo "[data.raw] shards already present under data/raw/gz_evo — skipping download." || \
	  $(PYTHON) scripts/download_gz_evo.py --config default --split train --n-files 3

data.concepts:  ## P2 — compute statmorph physical concepts
	$(PYTHON) scripts/build_dataset.py stage=concepts

data.labels:  ## P3 — build concept-label table + leakage-safe splits
	$(PYTHON) scripts/build_dataset.py stage=labels

# ---------------- Modelling stages ----------------
stage1:  ## P4 — Stage-1 concept predictor (Zoobot fine-tune)
	$(PYTHON) scripts/train_concepts.py

stage2:  ## P5 — Stage-2 symbolic decision head (PySR)
	$(PYTHON) scripts/train_symbolic.py

stage3:  ## P6 — Stage-3 split-conformal wrapper
	$(PYTHON) scripts/calibrate_conformal.py

# ---------------- Evaluation & robustness ----------------
baselines:  ## P7 — baselines + ablations
	$(PYTHON) scripts/run_baselines.py

robustness:  ## P8 — cross-survey shift experiments
	$(PYTHON) scripts/run_robustness.py

eval:  ## P9 — regenerate every figure and results/metrics.json
	$(PYTHON) scripts/regenerate_paper.py

all: data.raw data.concepts data.labels stage1 stage2 stage3 baselines robustness eval  ## Full pipeline

all.local: data.concepts data.labels stage2 stage3 baselines robustness eval  ## Pipeline minus torch-only stages (macOS x86_64 reproducer)

clean:  ## Remove caches (keeps data/ and models/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
