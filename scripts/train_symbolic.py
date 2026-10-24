"""Stage 2 entry point: PySR one-vs-rest symbolic decision head.

Consumes:
    results/concepts/preds.parquet       (predicted concepts per split — Stage 1)
    data/processed/dataset.parquet       (ground-truth Hubble labels)
    data/processed/splits.parquet        (train / val / calibration / test)

Emits:
    results/symbolic/rules.tex           LaTeX equations
    results/symbolic/rules.txt           plain-text equations
    src/galaxycbm/symbolic/exported_rules.py     importable Python predictor
    results/symbolic/rule_table.csv      human-readable rule table
    results/symbolic/pareto.png          accuracy vs complexity per class
    results/symbolic/metrics.json        acc, macro-F1, κ, confusion matrix
    results/symbolic/run.json            git SHA + versions + config hash
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from galaxycbm.concepts import build_head_specs
from galaxycbm.symbolic import (
    build_features,
    compute_metrics,
    export_callable,
    export_latex,
    export_plain,
    fit_symbolic,
    pareto_figure,
    predict_labels,
    rules_dataframe,
)
from galaxycbm.utils import load_config, seed_everything, write_run_json
from galaxycbm.utils.io import ensure_dir, write_json


def _require(path: Path) -> None:
    if not path.exists():
        print(f"[stage2] missing {path}", file=sys.stderr)
        raise SystemExit(2)


def main() -> None:
    concepts_cfg = load_config("concepts")
    symbolic_cfg = load_config("symbolic")
    seed_everything(int(symbolic_cfg.seed))

    preds_path = Path("results/concepts/preds.parquet")
    dataset_path = Path("data/processed/dataset.parquet")
    splits_path = Path("data/processed/splits.parquet")
    for p in (preds_path, dataset_path, splits_path):
        _require(p)

    # ---- Load & align ----------------------------------------------------
    preds = pd.read_parquet(preds_path)
    dataset = pd.read_parquet(dataset_path)
    splits = pd.read_parquet(splits_path)

    if "id_str" not in preds.columns or "id_str" not in dataset.columns:
        raise KeyError("both preds.parquet and dataset.parquet must have id_str")

    heads = build_head_specs(concepts_cfg)
    X_all, feat_spec = build_features(preds, heads)
    X_all = X_all.copy()
    X_all["id_str"] = preds["id_str"].to_numpy()
    X_all["split"] = preds["split"].to_numpy() if "split" in preds.columns else None

    labels = dataset[["id_str", "hubble_type"]].astype({"id_str": str})
    joined = X_all.astype({"id_str": str}).merge(labels, on="id_str", how="inner")

    train_ids = set(
        dataset.iloc[splits.loc[splits["split"] == "train", "row_index"].to_numpy()]["id_str"].astype(str)
    )
    val_ids = set(
        dataset.iloc[splits.loc[splits["split"] == "val", "row_index"].to_numpy()]["id_str"].astype(str)
    )
    tr = joined[joined["id_str"].isin(train_ids)].reset_index(drop=True)
    va = joined[joined["id_str"].isin(val_ids)].reset_index(drop=True)
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(
            f"empty split after join — train={len(tr)}, val={len(va)}. "
            "Check that preds.parquet covers the same id_str values as dataset.parquet."
        )

    X_train = tr[feat_spec.columns].copy()
    y_train = tr["hubble_type"]
    X_val = va[feat_spec.columns].copy()
    y_val = va["hubble_type"]

    # PySR won't accept NaN. Fit train medians (fall back to 0 for all-NaN
    # columns) and reuse them on val so leakage stays impossible.
    train_medians = X_train.median(numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(train_medians).fillna(0.0)
    X_val = X_val.fillna(train_medians).fillna(0.0)

    # ---- Fit + CV expression selection ----------------------------------
    print(f"[stage2] fitting PySR one-vs-rest on {len(X_train)} training rows, "
          f"{len(feat_spec.columns)} features, "
          f"{y_train.nunique()} classes.")
    result = fit_symbolic(X_train, y_train, symbolic_cfg)

    # ---- Exports ---------------------------------------------------------
    latex_path = Path(str(symbolic_cfg.export.latex_path))
    plain_path = Path(str(symbolic_cfg.export.plain_path))
    python_path = Path(str(symbolic_cfg.export.python_path))
    export_latex(result, latex_path)
    export_plain(result, plain_path)
    export_callable(result, python_path)

    rule_table_path = Path("results/symbolic/rule_table.csv")
    ensure_dir(rule_table_path.parent)
    rules_df = rules_dataframe(result)
    rules_df.to_csv(rule_table_path, index=False)

    pareto_path = pareto_figure(result, "results/symbolic/pareto.png")

    # ---- Evaluate on val -------------------------------------------------
    y_pred = predict_labels(result.rules, X_val)
    metrics = compute_metrics(y_val, y_pred)
    metrics_path = Path("results/symbolic/metrics.json")
    write_json(metrics_path, metrics)

    max_concepts_used = int(rules_df["complexity"].max())
    write_run_json(
        "symbolic",
        seed=int(symbolic_cfg.seed),
        config={"symbolic": symbolic_cfg},
        extra={
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_features": int(len(feat_spec.columns)),
            "classes": result.classes,
            "max_complexity": max_concepts_used,
            "metrics": metrics,
            "rules_tex": str(latex_path),
            "rules_txt": str(plain_path),
            "rules_py": str(python_path),
            "rule_table_csv": str(rule_table_path),
            "pareto_png": str(pareto_path),
            "metrics_json": str(metrics_path),
        },
    )

    print(f"[stage2] val accuracy = {metrics['accuracy']:.3f}, "
          f"macro-F1 = {metrics['macro_f1']:.3f}, κ = {metrics['cohen_kappa']:.3f}")
    print(f"[stage2] max complexity across classes = {max_concepts_used}")
    print(f"[stage2] wrote:")
    for p in (latex_path, plain_path, python_path, rule_table_path, pareto_path, metrics_path):
        print(f"[stage2]   {p}")


if __name__ == "__main__":
    main()
