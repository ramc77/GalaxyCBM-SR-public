"""Stage 9 entry point: regenerate every paper artefact from results/.

Produces:
    results/metrics.json                 — single source of truth
    paper/figures/*.pdf                  — publication-styled figures
    paper/tables/*.{csv,tex}             — publication tables + CSV
    paper/claims_report.json             — consistency check output
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from galaxycbm.eval import (
    aggregate_metrics,
    build_all_figures,
    check_claims,
    load_claims_manifest,
    write_table,
)
from galaxycbm.utils import write_run_json
from galaxycbm.utils.io import ensure_dir, write_json

RESULTS_ROOT = Path("results")
PAPER_ROOT = Path("paper")
STAGE = "eval"


def main() -> None:
    ensure_dir(RESULTS_ROOT)
    metrics = aggregate_metrics(RESULTS_ROOT)
    metrics_path = RESULTS_ROOT / "metrics.json"
    write_json(metrics_path, metrics)

    figs = build_all_figures(RESULTS_ROOT, PAPER_ROOT)

    tables_out = PAPER_ROOT / "tables"
    ensure_dir(tables_out)
    written: list[tuple[Path, Path]] = []

    rules_csv = RESULTS_ROOT / "symbolic" / "rule_table.csv"
    if rules_csv.exists():
        rules_df = pd.read_csv(rules_csv)[["hubble_class", "expression",
                                            "complexity", "cv_accuracy"]]
        written.append(write_table(
            rules_df,
            tables_out / "rules.csv", tables_out / "rules.tex",
            caption=("Symbolic decision head — one rule per Hubble class "
                     "(Stage 2). Rules were selected by stratified $k$-fold CV "
                     "accuracy on the training pool."),
            label="tab:rules",
            floatfmt=".3f",
        ))

    comparison_csv = RESULTS_ROOT / "tables" / "comparison.csv"
    if comparison_csv.exists():
        cmp_df = pd.read_csv(comparison_csv)
        written.append(write_table(
            cmp_df, tables_out / "comparison.csv", tables_out / "comparison.tex",
            caption=("Baselines and ablations on the validation split (Stage 7). "
                     "Interpretability cost is expression complexity for the "
                     "symbolic head, non-zero coefficient count for the linear "
                     "CBM, non-zero feature-importance count for XGBoost, and "
                     "trainable parameter count for the end-to-end CNN."),
            label="tab:baselines",
            floatfmt=".3f",
        ))

    robustness_csv = RESULTS_ROOT / "tables" / "robustness.csv"
    if robustness_csv.exists():
        rob_df = pd.read_csv(robustness_csv)
        written.append(write_table(
            rob_df, tables_out / "robustness.csv", tables_out / "robustness.tex",
            caption=("Cross-survey shift relative to the DECaLS reference "
                     "(Stage 8). Rule stability is per-class Jaccard on features "
                     "used; top-$k$ overlap is Jaccard on the top-5 dominant "
                     "concepts overall."),
            label="tab:robustness",
            floatfmt=".3f",
        ))

    claims_path = PAPER_ROOT / "claims.yaml"
    if claims_path.exists():
        report = check_claims(metrics, load_claims_manifest(claims_path))
        (PAPER_ROOT / "claims_report.json").write_text(
            (import_json := __import__("json")).dumps({
                "total": report.total,
                "present": report.present,
                "missing": report.missing,
                "missing_optional": report.missing_optional,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        if not report.ok:
            print(f"[eval] {len(report.missing)} orphan claim(s) — no supporting data:",
                  file=sys.stderr)
            for c in report.missing:
                print(f"[eval]   {c}", file=sys.stderr)
    else:
        report = None

    write_run_json(
        STAGE,
        seed=None,
        config=None,
        extra={
            "metrics_json": str(metrics_path),
            "figures":      [str(p) for p in figs],
            "tables":       [{"csv": str(csv), "tex": str(tex)} for csv, tex in written],
            "claims": {
                "total": report.total if report else 0,
                "present": report.present if report else 0,
                "missing": report.missing if report else [],
                "missing_optional": report.missing_optional if report else [],
            } if report else None,
        },
    )

    print(f"[eval] metrics.json → {metrics_path}")
    print(f"[eval] figures ({len(figs)}):")
    for p in figs:
        print(f"[eval]   {p}")
    print(f"[eval] tables ({len(written)}):")
    for csv, tex in written:
        print(f"[eval]   {csv}  {tex}")
    if report is not None:
        print(f"[eval] claims: {report.present}/{report.total} present, "
              f"{len(report.missing)} missing, {len(report.missing_optional)} optional-missing")
        if not report.ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
