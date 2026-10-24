"""Calibration and parsimony diagnostics.

Two analyses that run on top of frozen Stage-1 and Stage-2 outputs, so nothing
upstream is retrained:

  1. Class-conditional (Mondrian) conformal calibration — a concrete mitigation
     for the rare-class coverage collapse reported in the submitted version,
     rather than a deferral to future work.
  2. A compact-rule ablation — the smallest expression per class within a
     tolerance of the cross-validated optimum, evaluated end to end, so the
     adopted 17-25 node expressions are justified against a parsimonious
     alternative instead of asserted.

Reads:
    src/galaxycbm/symbolic/exported_rules.py
    results/symbolic/rule_table.csv, results/symbolic/fit_cache/
    results/concepts/preds.parquet
    data/processed/dataset.parquet, splits.parquet
Emits:
    results/revision/mondrian_summary.json
    results/revision/mondrian_per_class.csv
    results/revision/compact_rules.csv
    results/revision/compact_comparison.json
    results/revision/run.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_conformal import _rules_from_stage2  # noqa: E402

from galaxycbm.concepts import build_head_specs  # noqa: E402
from galaxycbm.symbolic import (  # noqa: E402
    build_features,
    evaluate_rule_set,
    load_fronts,
    select_compact_rules,
)
from galaxycbm.uncertainty import (  # noqa: E402
    SymbolicRuleClassifier,
    conformalize,
    coverage_and_set_size,
    min_calibration_points,
    mondrian_conformalize,
    mondrian_predict_sets,
    mondrian_report,
    per_class_coverage,
    predict_sets,
)
from galaxycbm.utils import load_config, seed_everything, write_run_json  # noqa: E402
from galaxycbm.utils.io import ensure_dir, write_json  # noqa: E402

STAGE = "revision"
TAU = 0.005          # CV-accuracy tolerance for the compact selection


def _require(p: Path) -> None:
    if not p.exists():
        print(f"[revision] missing {p}", file=sys.stderr)
        raise SystemExit(2)


def _load_splits() -> tuple[pd.DataFrame, list, list[str]]:
    concepts_cfg = load_config("concepts")
    preds_path = Path("results/concepts/preds.parquet")
    dataset_path = Path("data/processed/dataset.parquet")
    splits_path = Path("data/processed/splits.parquet")
    for p in (preds_path, dataset_path, splits_path):
        _require(p)

    preds = pd.read_parquet(preds_path)
    dataset = pd.read_parquet(dataset_path)
    splits = pd.read_parquet(splits_path)

    heads = build_head_specs(concepts_cfg)
    X_all, _ = build_features(preds, heads)
    X_all["id_str"] = preds["id_str"].astype(str).to_numpy()

    rules, classes, feature_cols = _rules_from_stage2()
    labels = dataset[["id_str", "hubble_type"]].astype({"id_str": str})
    joined = X_all.astype({"id_str": str}).merge(labels, on="id_str", how="inner")

    # Split names are those written by Stage 3 (data/processed/splits.parquet):
    # "train", "val", "calibration", "test". Fail loudly on a rename rather
    # than handing an empty frame to sklearn several calls later.
    available = sorted(splits["split"].astype(str).unique())
    wanted = ("train", "val", "calibration", "test")
    missing = [n for n in wanted if n not in available]
    if missing:
        raise KeyError(
            f"splits.parquet has no split named {missing}; it contains {available}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for name in wanted:
        ids = set(
            dataset.iloc[splits.loc[splits["split"] == name, "row_index"].to_numpy()]["id_str"]
                   .astype(str)
        )
        frames[name] = joined[joined["id_str"].isin(ids)].reset_index(drop=True)
        if frames[name].empty:
            raise RuntimeError(
                f"split {name!r} is empty after joining predicted concepts to labels "
                f"({len(ids)} ids in the split). Stage 1 predictions and the dataset "
                "are out of sync."
            )

    return frames, rules, feature_cols


def main() -> None:
    conformal_cfg = load_config("conformal")
    alpha = float(conformal_cfg.alpha)
    seed_everything(int(conformal_cfg.seed))

    frames, rules, feature_cols = _load_splits()
    classes = [r.hubble_class for r in rules]
    out_dir = ensure_dir(Path("results/revision"))

    # ---- shared feature prep (identical NaN policy to Stage 3) -------------
    cal, tst, val = frames["calibration"], frames["test"], frames["val"]
    X_cal = cal[feature_cols].copy()
    medians = X_cal.median(numeric_only=True).fillna(0.0)
    X_cal = X_cal.fillna(medians).fillna(0.0)
    X_tst = tst[feature_cols].copy().fillna(medians).fillna(0.0)
    X_val = val[feature_cols].copy().fillna(medians).fillna(0.0)
    y_cal, y_tst, y_val = cal["hubble_type"], tst["hubble_type"], val["hubble_type"]

    # Drop rows whose true class the rule set does not score, exactly as Stage 3
    # does: LAC nonconformity is undefined for a class with no rule.
    known = set(classes)
    m_cal, m_tst = y_cal.astype(str).isin(known), y_tst.astype(str).isin(known)
    m_val = y_val.astype(str).isin(known)
    X_cal, y_cal = X_cal[m_cal].reset_index(drop=True), y_cal[m_cal].reset_index(drop=True)
    X_tst, y_tst = X_tst[m_tst].reset_index(drop=True), y_tst[m_tst].reset_index(drop=True)
    X_val, y_val = X_val[m_val].reset_index(drop=True), y_val[m_val].reset_index(drop=True)

    estimator = SymbolicRuleClassifier(rules=rules, feature_columns=feature_cols)

    # =======================================================================
    # 1. Mondrian conformal vs the marginal head, same calibration data
    # =======================================================================
    marg_head = conformalize(estimator, X_cal, y_cal, alpha=alpha,
                             method=str(conformal_cfg.method),
                             random_state=int(conformal_cfg.seed))
    _, marg_mask = predict_sets(marg_head, X_tst)
    marg_summary = coverage_and_set_size(marg_head, marg_mask, y_tst)
    marg_per_class = per_class_coverage(marg_head, marg_mask, y_tst)

    mond_head = mondrian_conformalize(estimator, X_cal, y_cal, alpha=alpha)
    _, mond_mask = mondrian_predict_sets(mond_head, estimator, X_tst)
    mond_summary, mond_per_class = mondrian_report(mond_head, mond_mask, y_tst)

    merged = mond_per_class.merge(
        marg_per_class[["class", "coverage", "mean_set_size"]]
        .rename(columns={"coverage": "marginal_coverage",
                         "mean_set_size": "marginal_mean_set_size"}),
        on="class", how="left",
    )
    merged = merged.rename(columns={"coverage": "mondrian_coverage",
                                    "mean_set_size": "mondrian_mean_set_size"})
    merged.to_csv(out_dir / "mondrian_per_class.csv", index=False)

    # Worst-case per-class coverage is the statistic conditional validity is
    # judged on; the marginal figure can hide an arbitrarily bad class.
    def _worst(df: pd.DataFrame, col: str) -> float:
        v = df.loc[df["n"] > 0, col]
        return float(v.min()) if len(v) else float("nan")

    comparison = {
        "alpha": alpha,
        "n_test": int(len(y_tst)),
        "n_calibration": int(len(y_cal)),
        "min_calibration_points_for_validity": min_calibration_points(alpha),
        "marginal": marg_summary,
        "mondrian": mond_summary,
        "worst_class_coverage_marginal": _worst(merged, "marginal_coverage"),
        "worst_class_coverage_mondrian": _worst(merged, "mondrian_coverage"),
        "set_size_cost": float(mond_summary["mean_set_size"] - marg_summary["mean_set_size"]),
        "degenerate_classes": mond_head.degenerate,
        "class_calibration_counts": mond_head.n_cal,
    }
    write_json(out_dir / "mondrian_summary.json", comparison)

    print(f"[revision] marginal : coverage {marg_summary['empirical_coverage']:.3f}  "
          f"|C| {marg_summary['mean_set_size']:.2f}  "
          f"worst-class {comparison['worst_class_coverage_marginal']:.3f}")
    print(f"[revision] mondrian : coverage {mond_summary['empirical_coverage']:.3f}  "
          f"|C| {mond_summary['mean_set_size']:.2f}  "
          f"worst-class {comparison['worst_class_coverage_mondrian']:.3f}")
    if mond_head.degenerate:
        print(f"[revision] degenerate (q=inf, always in set): {mond_head.degenerate} "
              f"— need n_cal >= {min_calibration_points(alpha)} per class")

    # =======================================================================
    # 2. Compact-rule ablation
    # =======================================================================
    fronts = load_fronts("results/symbolic/fit_cache", rules)
    missing = sorted(set(classes) - set(fronts))
    if not fronts:
        # With no fronts the "compact" set is just the adopted set and the
        # ablation answers nothing. Say so rather than reporting a null delta.
        raise RuntimeError(
            "no Pareto fronts matched the adopted rules under "
            "results/symbolic/fit_cache/. The *__pareto.parquet files are "
            "written by Stage 2 and are git-ignored, so they exist only where "
            "the fit ran. Re-run `make stage2` (cached classes are skipped) or "
            "copy the fit_cache from the machine that produced the rules."
        )
    if missing:
        print(f"[revision] WARNING: no matching Pareto front for {missing}; "
              "those classes keep their adopted rule.", file=sys.stderr)

    compact, table = select_compact_rules(rules, fronts, tau=TAU)
    table.to_csv(out_dir / "compact_rules.csv", index=False)

    adopted_val = evaluate_rule_set(rules, X_val, y_val)
    compact_val = evaluate_rule_set(compact, X_val, y_val)
    adopted_tst = evaluate_rule_set(rules, X_tst, y_tst)
    compact_tst = evaluate_rule_set(compact, X_tst, y_tst)

    ablation = {
        "tau": TAU,
        "classes_with_front": sorted(fronts),
        "classes_without_front": missing,
        "adopted": {"validation": adopted_val, "test": adopted_tst},
        "compact": {"validation": compact_val, "test": compact_tst},
        "delta_validation": {
            k: float(compact_val[k] - adopted_val[k])
            for k in ("accuracy", "macro_f1", "cohen_kappa")
        },
        "delta_test": {
            k: float(compact_tst[k] - adopted_tst[k])
            for k in ("accuracy", "macro_f1", "cohen_kappa")
        },
        "node_reduction": int(adopted_val["total_nodes"] - compact_val["total_nodes"]),
    }
    write_json(out_dir / "compact_comparison.json", ablation)

    print(f"[revision] adopted  : {adopted_val['total_nodes']:3d} nodes  "
          f"val acc {adopted_val['accuracy']:.3f}  kappa {adopted_val['cohen_kappa']:.3f}")
    print(f"[revision] compact  : {compact_val['total_nodes']:3d} nodes  "
          f"val acc {compact_val['accuracy']:.3f}  kappa {compact_val['cohen_kappa']:.3f}")

    write_run_json(
        STAGE,
        seed=int(conformal_cfg.seed),
        config={"conformal": conformal_cfg, "tau": TAU},
        extra={"mondrian": comparison, "compact": ablation},
    )


if __name__ == "__main__":
    main()
