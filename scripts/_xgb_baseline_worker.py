"""Isolated subprocess for the XGBoost + SHAP baseline.

Why a subprocess: numba (a SHAP/XGBoost dependency) has hit native
segfaults on some macOS/Apple-Silicon setups (missing libomp linkage). A
segfault kills the whole interpreter — no Python try/except can catch it.
Running this baseline in its own process means a crash here costs one row
in the comparison table, not the entire `make baselines` run.

Writes --out as JSON: {"metrics": {...}, "interpretability_cost": int,
"importance": {feature: value}, "shap_importance": {feature: value} | null}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from galaxycbm.baselines.data import load_baseline_data
from galaxycbm.baselines.xgb_concepts import (
    concept_importance as xgb_importance,
)
from galaxycbm.baselines.xgb_concepts import (
    interpretability_cost as xgb_cost,
)
from galaxycbm.baselines.xgb_concepts import (
    shap_importance as xgb_shap_importance,
)
from galaxycbm.baselines.xgb_concepts import train_xgb
from galaxycbm.utils import load_config


def _to_json_safe_dict(series: pd.Series) -> dict[str, float]:
    """numpy float32/float64 scalars aren't JSON-serializable by default —
    coerce explicitly rather than trust pandas/numpy version behaviour.
    """
    return {str(k): float(v) for k, v in series.to_dict().items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    concepts_cfg = load_config("concepts")
    X_train, y_train, X_val, y_val, *_ = load_baseline_data(concepts_cfg)

    xgb_est, metrics, _le = train_xgb(X_train, y_train, X_val, y_val, seed=0)

    result = {
        "metrics": metrics,
        "interpretability_cost": int(xgb_cost(xgb_est)),
        "importance": _to_json_safe_dict(xgb_importance(xgb_est)),
        "shap_importance": None,
    }
    try:
        result["shap_importance"] = _to_json_safe_dict(xgb_shap_importance(xgb_est, X_val))
    except Exception as e:
        print(f"[xgb-worker] SHAP unavailable: {e}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
