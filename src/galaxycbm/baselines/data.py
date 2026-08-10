"""Shared baseline-data loader.

Used by both the main driver (scripts/run_baselines.py, in-process) and the
xgboost worker (scripts/_xgb_baseline_worker.py, subprocess) so the two
never see different data — same rules, same split, same feature order.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from galaxycbm.concepts import build_head_specs
from galaxycbm.symbolic import ClassRule, build_features


def require(p: Path) -> None:
    if not p.exists():
        print(f"[baselines] missing {p}", file=sys.stderr)
        raise SystemExit(2)


def load_exported_rules() -> tuple[list[ClassRule], list[str], list[str]]:
    path = Path("src/galaxycbm/symbolic/exported_rules.py")
    require(path)
    spec = importlib.util.spec_from_file_location("_gc_exported_rules", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tbl_path = Path("results/symbolic/rule_table.csv")
    tbl = pd.read_csv(tbl_path).set_index("hubble_class") if tbl_path.exists() else None
    rules: list[ClassRule] = []
    for cls in list(mod.CLASSES):
        row = tbl.loc[cls] if (tbl is not None and cls in tbl.index) else None
        rules.append(ClassRule(
            hubble_class=cls,
            equation_str=mod.EXPRESSIONS[cls],
            latex=str(row["latex"]) if row is not None else "",
            complexity=int(row["complexity"]) if row is not None else 0,
            pysr_score=float(row["pysr_score"]) if row is not None else 0.0,
            cv_accuracy=float(row["cv_accuracy"]) if row is not None else float("nan"),
        ))
    return rules, list(mod.CLASSES), list(mod.FEATURE_COLUMNS)


def _split_frame(dataset: pd.DataFrame, splits: pd.DataFrame, X_all: pd.DataFrame, name: str) -> pd.DataFrame:
    ids = set(dataset.iloc[splits.loc[splits["split"] == name, "row_index"].to_numpy()]["id_str"].astype(str))
    return X_all[X_all["id_str"].isin(ids)].reset_index(drop=True)


def load_baseline_data(concepts_cfg) -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
    list[str], list[ClassRule], list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    """Returns (X_train, y_train, X_val, y_val, feat_cols, rules, classes, tr, va, dataset)."""
    preds_path = Path("results/concepts/preds.parquet")
    dataset_path = Path("data/processed/dataset.parquet")
    splits_path = Path("data/processed/splits.parquet")
    for p in (preds_path, dataset_path, splits_path):
        require(p)

    preds = pd.read_parquet(preds_path)
    dataset = pd.read_parquet(dataset_path)
    splits = pd.read_parquet(splits_path)

    heads = build_head_specs(concepts_cfg)
    X_all, _feat_spec = build_features(preds, heads)
    X_all["id_str"] = preds["id_str"].astype(str).to_numpy()
    labels = dataset[["id_str", "hubble_type"]].astype({"id_str": str})
    X_all = X_all.merge(labels, on="id_str", how="inner")

    rules, classes, feat_cols = load_exported_rules()
    for c in feat_cols:
        if c not in X_all.columns:
            X_all[c] = 0.0
    tr = _split_frame(dataset, splits, X_all, "train")
    va = _split_frame(dataset, splits, X_all, "val")
    if tr.empty or va.empty:
        raise RuntimeError(f"empty train/val split: train={len(tr)}, val={len(va)}")

    X_train = tr[feat_cols].copy()
    X_val = va[feat_cols].copy()
    medians = X_train.median(numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_val = X_val.fillna(medians).fillna(0.0)
    y_train = tr["hubble_type"]
    y_val = va["hubble_type"]

    return X_train, y_train, X_val, y_val, feat_cols, rules, classes, tr, va, dataset
