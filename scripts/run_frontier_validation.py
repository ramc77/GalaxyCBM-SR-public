"""Out-of-sample check of Stage-2 expression selection.

Scores every cached Pareto-front expression on the training pool and on the
validation split, which the symbolic search never saw, and compares the rule
set Stage 2 adopted with the rule set that selecting on validation accuracy
would give. Nothing is refitted and no rule used downstream is changed.

Reads:
    src/galaxycbm/symbolic/exported_rules.py, results/symbolic/rule_table.csv
    results/symbolic/fit_cache/*__pareto.parquet     (git-ignored; iMac only)
    results/concepts/preds.parquet, data/processed/{dataset,splits}.parquet
Emits:
    results/symbolic/frontier_accuracies.csv   one row per frontier expression
    results/symbolic/frontier_selection.json   adopted vs validation-selected
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_revision import _load_splits  # noqa: E402

from galaxycbm.symbolic import load_fronts  # noqa: E402
from galaxycbm.symbolic.frontier import (  # noqa: E402
    frontier_accuracy_table,
    prepare_splits,
    selection_report,
)
from galaxycbm.utils.io import write_json  # noqa: E402

SPLITS = ("train", "val")      # the frontier table covers only these two
OUT_CSV = Path("results/symbolic/frontier_accuracies.csv")
OUT_JSON = Path("results/symbolic/frontier_selection.json")


def main() -> None:
    frames, adopted, feature_cols = _load_splits()
    classes = [r.hubble_class for r in adopted]

    n_nan = {n: int(frames[n][feature_cols].isna().sum().sum()) for n in frames}
    print("[frontier] NaN feature values per split (imputed with train medians):", n_nan)
    X, y = prepare_splits(frames, feature_cols, classes)
    print("[frontier] rows per split:", {k: len(v) for k, v in X.items()})

    fronts = load_fronts("results/symbolic/fit_cache", adopted)
    if set(fronts) != set(classes):
        raise RuntimeError(
            f"missing Pareto fronts for {sorted(set(classes) - set(fronts))}; "
            "the *__pareto.parquet files exist only where Stage 2 ran.")

    table = frontier_accuracy_table(
        fronts, {s: X[s] for s in SPLITS}, {s: y[s] for s in SPLITS})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    report = selection_report(table, adopted, X, y)
    write_json(OUT_JSON, report)

    print(f"\n[frontier] stored cv_accuracy vs recomputed training accuracy: "
          f"max |difference| = {report['max_abs_cv_minus_train']:.2e} "
          f"over {len(table)} expressions")
    print("\nclass  adopted(n)  train   val  |  val-selected(n)  train   val   same")
    for r in report["per_class"]:
        print(f"{r['hubble_class']:>5}  {r['adopted_complexity']:>9}  "
              f"{r['adopted_train_acc']:.4f} {r['adopted_val_acc']:.4f} |  "
              f"{r['selected_complexity']:>13}  {r['selected_train_acc']:.4f} "
              f"{r['selected_val_acc']:.4f}  {r['same_expression']}")
    for label in ("adopted", "val_selected"):
        m = report["metrics"][label]
        v, t = m["val"], m["test"]
        print(f"{label:>13}: nodes {m['total_nodes']:>3} | "
              f"val acc {v['accuracy']:.3f} F1 {v['macro_f1']:.3f} k {v['cohen_kappa']:.3f} | "
              f"test acc {t['accuracy']:.3f} F1 {t['macro_f1']:.3f} k {t['cohen_kappa']:.3f}")
    print(f"\nwrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
