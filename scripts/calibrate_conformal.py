"""Stage 3 entry point: split-conformal wrapper around the Stage-2 symbolic head.

Reads:
    src/galaxycbm/symbolic/exported_rules.py   (Stage-2 output)
    results/concepts/preds.parquet
    data/processed/dataset.parquet, splits.parquet
    configs/conformal.yaml
Emits:
    results/uncertainty/metrics.json            coverage + set-size summary
    results/uncertainty/per_class_coverage.csv
    results/uncertainty/selective_curve.csv
    results/uncertainty/selective_risk.png
    results/uncertainty/run.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.concepts import build_head_specs
from galaxycbm.symbolic import ClassRule, build_features
from galaxycbm.uncertainty import (
    SymbolicRuleClassifier,
    conformalize,
    coverage_and_set_size,
    per_class_coverage,
    predict_sets,
    selective_curve,
    selective_figure,
)
from galaxycbm.utils import load_config, seed_everything, write_run_json
from galaxycbm.utils.io import ensure_dir, write_json

STAGE = "uncertainty"


def _require(p: Path) -> None:
    if not p.exists():
        print(f"[stage3] missing {p}", file=sys.stderr)
        raise SystemExit(2)


def _load_exported_rules_module() -> object:
    """Import the auto-generated src/galaxycbm/symbolic/exported_rules.py."""
    path = Path("src/galaxycbm/symbolic/exported_rules.py")
    _require(path)
    spec = importlib.util.spec_from_file_location("_gc_exported_rules", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rules_from_stage2() -> tuple[list[ClassRule], list[str], list[str]]:
    """Rebuild ClassRule instances from the exported module + rule_table.csv."""
    mod = _load_exported_rules_module()
    classes = list(mod.CLASSES)
    feature_cols = list(mod.FEATURE_COLUMNS)
    table_path = Path("results/symbolic/rule_table.csv")
    if table_path.exists():
        tbl = pd.read_csv(table_path)
        tbl = tbl.set_index("hubble_class")
    else:
        tbl = None
    rules: list[ClassRule] = []
    for cls in classes:
        eqn = mod.EXPRESSIONS[cls]
        row = tbl.loc[cls] if tbl is not None and cls in tbl.index else None
        rules.append(ClassRule(
            hubble_class=cls,
            equation_str=eqn,
            latex=str(row["latex"]) if row is not None else "",
            complexity=int(row["complexity"]) if row is not None else 0,
            pysr_score=float(row["pysr_score"]) if row is not None else 0.0,
            cv_accuracy=float(row["cv_accuracy"]) if row is not None else float("nan"),
        ))
    return rules, classes, feature_cols


def main() -> None:
    concepts_cfg = load_config("concepts")
    conformal_cfg = load_config("conformal")
    seed_everything(int(conformal_cfg.seed))

    preds_path = Path("results/concepts/preds.parquet")
    dataset_path = Path("data/processed/dataset.parquet")
    splits_path = Path("data/processed/splits.parquet")
    for p in (preds_path, dataset_path, splits_path):
        _require(p)

    preds = pd.read_parquet(preds_path)
    dataset = pd.read_parquet(dataset_path)
    splits = pd.read_parquet(splits_path)

    heads = build_head_specs(concepts_cfg)
    X_all, feat_spec = build_features(preds, heads)
    X_all["id_str"] = preds["id_str"].astype(str).to_numpy()

    rules, classes, feature_cols = _rules_from_stage2()
    if set(feature_cols) - set(feat_spec.columns):
        missing = sorted(set(feature_cols) - set(feat_spec.columns))
        raise KeyError(f"exported_rules references features missing from preds: {missing}")

    # NaN policy — mirror Stage-2 (train medians live in calibration set here).
    labels = dataset[["id_str", "hubble_type"]].astype({"id_str": str})
    joined = X_all.astype({"id_str": str}).merge(labels, on="id_str", how="inner")

    def split_frame(name: str) -> pd.DataFrame:
        ids = set(
            dataset.iloc[splits.loc[splits["split"] == name, "row_index"].to_numpy()]["id_str"]
                   .astype(str)
        )
        return joined[joined["id_str"].isin(ids)].reset_index(drop=True)

    cal = split_frame("calibration")
    tst = split_frame("test")
    if cal.empty or tst.empty:
        raise RuntimeError(
            f"empty calibration or test split (cal={len(cal)}, test={len(tst)}). "
            "Adjust configs/data.yaml → labels.splits to allocate rows to both."
        )

    X_cal = cal[feature_cols].copy()
    X_tst = tst[feature_cols].copy()
    medians = X_cal.median(numeric_only=True).fillna(0.0)
    X_cal = X_cal.fillna(medians).fillna(0.0)
    X_tst = X_tst.fillna(medians).fillna(0.0)
    y_cal = cal["hubble_type"]
    y_tst = tst["hubble_type"]

    # Drop calibration / test rows whose true class is unseen by Stage 2 —
    # LAC's nonconformity is defined only for classes the estimator scores.
    known = set(classes)
    cal_keep = y_cal.astype(str).isin(known)
    tst_keep = y_tst.astype(str).isin(known)
    X_cal, y_cal = X_cal[cal_keep].reset_index(drop=True), y_cal[cal_keep].reset_index(drop=True)
    X_tst, y_tst = X_tst[tst_keep].reset_index(drop=True), y_tst[tst_keep].reset_index(drop=True)
    if X_cal.empty or X_tst.empty:
        raise RuntimeError("no calibration/test rows survived class-filter to Stage 2 classes")

    # Conformal fit + prediction sets.
    estimator = SymbolicRuleClassifier(rules=rules, feature_columns=feature_cols)
    head = conformalize(
        estimator, X_cal, y_cal,
        alpha=float(conformal_cfg.alpha),
        method=str(conformal_cfg.method),
        random_state=int(conformal_cfg.seed),
    )
    _, set_mask = predict_sets(head, X_tst)

    summary = coverage_and_set_size(head, set_mask, y_tst)
    per_class = per_class_coverage(head, set_mask, y_tst)
    curve = selective_curve(
        estimator, X_tst, y_tst, set_mask,
        abstain_when_set_size_ge=int(conformal_cfg.selective.abstain_when_set_size_ge),
    )

    out_dir = ensure_dir(Path("results/uncertainty"))
    metrics_path = out_dir / "metrics.json"
    per_class_path = out_dir / "per_class_coverage.csv"
    curve_path = out_dir / "selective_curve.csv"
    fig_path = out_dir / "selective_risk.png"

    write_json(metrics_path, summary)
    per_class.to_csv(per_class_path, index=False)
    curve.to_csv(curve_path, index=False)
    selective_figure(curve, fig_path)

    # Coverage tolerance check per conformal.yaml (default: |empirical - nominal| ≤ 3 * √(1/n_test))
    tol_cfg = conformal_cfg.get("coverage_tolerance", 0.05)
    tolerance = 3.0 / np.sqrt(len(y_tst)) if tol_cfg == "auto" else float(tol_cfg)
    within = abs(summary["coverage_gap"]) <= tolerance

    write_run_json(
        STAGE,
        seed=int(conformal_cfg.seed),
        config={"conformal": conformal_cfg},
        extra={
            "n_calibration": int(len(X_cal)),
            "n_test": int(len(X_tst)),
            "coverage_tolerance": float(tolerance),
            "coverage_within_tolerance": bool(within),
            "summary": summary,
            "metrics_json": str(metrics_path),
            "per_class_csv": str(per_class_path),
            "selective_curve_csv": str(curve_path),
            "selective_risk_png": str(fig_path),
        },
    )

    print(f"[stage3] target {1.0 - head.alpha:.2%} coverage → "
          f"empirical {summary['empirical_coverage']:.3f}  "
          f"gap {summary['coverage_gap']:+.3f}  (tol {tolerance:.3f})")
    print(f"[stage3] mean set size {summary['mean_set_size']:.2f}  "
          f"singletons {summary['singleton_fraction']:.2f}  "
          f"empty {summary['empty_fraction']:.2f}")
    print(f"[stage3] per-class coverage: {per_class_path}")
    print(f"[stage3] selective risk: {fig_path}")
    if not within:
        print(f"[stage3] warning: |gap| exceeds tolerance {tolerance:.3f} — "
              "expected on small calibration sets (finite-sample noise dominates).",
              file=sys.stderr)


if __name__ == "__main__":
    main()
