"""Stage 8 entry point: cross-survey robustness.

Applies the frozen Stage-2 rules to Euclid Q1 and JWST-COSMOS-Web samples,
refits PySR on each shifted survey to quantify rule stability, and writes a
summary table + shift figure + findings note.

Access points are locked in `src/galaxycbm/robustness/sources.py` (verified
against mwalmsley's HF namespace); override via configs/data.yaml → robustness
if the URLs move.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.concepts import build_head_specs
from galaxycbm.data.labels import HUBBLE_UNCLASSIFIED, build_dataset
from galaxycbm.robustness import (
    DEFAULT_SOURCES,
    download_all_shards,
    expected_calibration_error,
    findings_note,
    rule_stability,
    run_shifted_pipeline,
    shift_delta_row,
    shift_figure,
)
from galaxycbm.symbolic import ClassRule, build_features, compute_metrics, predict_labels
from galaxycbm.uncertainty import (
    SymbolicRuleClassifier,
    conformalize,
    coverage_and_set_size,
    predict_sets,
)
from galaxycbm.utils import load_config, seed_everything, write_run_json
from galaxycbm.utils.io import ensure_dir

STAGE = "robustness"


def _require(p: Path) -> None:
    if not p.exists():
        print(f"[robustness] missing {p}", file=sys.stderr)
        raise SystemExit(2)


def _load_exported_rules() -> tuple[list[ClassRule], list[str], list[str]]:
    path = Path("src/galaxycbm/symbolic/exported_rules.py")
    _require(path)
    spec = importlib.util.spec_from_file_location("_gc_exported_rules", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tbl = Path("results/symbolic/rule_table.csv")
    tbl_df = pd.read_csv(tbl).set_index("hubble_class") if tbl.exists() else None
    rules: list[ClassRule] = []
    for cls in list(mod.CLASSES):
        row = tbl_df.loc[cls] if (tbl_df is not None and cls in tbl_df.index) else None
        rules.append(ClassRule(
            hubble_class=cls,
            equation_str=mod.EXPRESSIONS[cls],
            latex=str(row["latex"]) if row is not None else "",
            complexity=int(row["complexity"]) if row is not None else 0,
            pysr_score=float(row["pysr_score"]) if row is not None else 0.0,
            cv_accuracy=float(row["cv_accuracy"]) if row is not None else float("nan"),
        ))
    return rules, list(mod.CLASSES), list(mod.FEATURE_COLUMNS)


def _reference_metrics_and_calibration(cfg_data, cfg_concepts, cfg_conformal, rules, feat_cols):
    """Recompute the reference-survey Stage-2/3 numbers so shift is a delta on the same run."""
    preds = pd.read_parquet("results/concepts/preds.parquet")
    dataset = pd.read_parquet("data/processed/dataset.parquet")
    splits = pd.read_parquet("data/processed/splits.parquet")

    heads = build_head_specs(cfg_concepts)
    X, feat_spec = build_features(preds, heads)
    X["id_str"] = preds["id_str"].astype(str).to_numpy()
    labels = dataset[["id_str", "hubble_type"]].astype({"id_str": str})
    joined = X.merge(labels, on="id_str", how="inner")
    for c in feat_cols:
        if c not in joined.columns:
            joined[c] = 0.0

    def frame(name: str) -> pd.DataFrame:
        ids = set(dataset.iloc[splits.loc[splits["split"] == name, "row_index"].to_numpy()]["id_str"].astype(str))
        return joined[joined["id_str"].isin(ids)].reset_index(drop=True)

    cal, val = frame("calibration"), frame("val")
    X_cal, X_val = cal[feat_cols].copy(), val[feat_cols].copy()
    medians = X_cal.median(numeric_only=True).fillna(0.0)
    X_cal = X_cal.fillna(medians).fillna(0.0)
    X_val = X_val.fillna(medians).fillna(0.0)
    y_cal, y_val = cal["hubble_type"], val["hubble_type"]
    keep = y_cal.astype(str).isin([r.hubble_class for r in rules])
    X_cal, y_cal = X_cal[keep].reset_index(drop=True), y_cal[keep].reset_index(drop=True)
    keep_v = y_val.astype(str).isin([r.hubble_class for r in rules])
    X_val, y_val = X_val[keep_v].reset_index(drop=True), y_val[keep_v].reset_index(drop=True)

    est = SymbolicRuleClassifier(rules=rules, feature_columns=feat_cols)
    est.fit(X_cal, y_cal)
    probs_val = est.predict_proba(X_val)
    ece = expected_calibration_error(probs_val, y_val, list(est.classes_))
    metrics = compute_metrics(y_val, predict_labels(rules, X_val))

    head = conformalize(est, X_cal, y_cal,
                        alpha=float(cfg_conformal.alpha),
                        method=str(cfg_conformal.method),
                        random_state=int(cfg_conformal.seed))
    _, mask = predict_sets(head, X_val)
    ref_cov = coverage_and_set_size(head, mask, y_val)["empirical_coverage"]
    return {
        "metrics": metrics, "ece": ece, "coverage": ref_cov,
        "est": est, "head": head, "X_cal": X_cal, "y_cal": y_cal,
        "n_val": int(len(y_val)),
    }


def _shifted_conformal_coverage(ref_head, est, X_shift, y_shift):
    _, mask = predict_sets(ref_head, X_shift)
    # class_to_idx uses ref_head.classes
    class_to_idx = {c: i for i, c in enumerate(ref_head.classes)}
    y = y_shift.astype(str).to_numpy()
    covered = np.array([mask[i, class_to_idx[c]] if c in class_to_idx else False
                        for i, c in enumerate(y)])
    return float(covered.mean())


def _load_shifted_dataset(source, cfg_data) -> pd.DataFrame:
    """Download the shifted survey and derive Hubble labels — no splits, no
    ra/dec dependency (both are irrelevant for shift eval; the whole survey
    is one big test set).
    """
    from galaxycbm.data.labels import derive_hubble_type

    out_root = Path(cfg_data.download.root) / source.name
    shards = download_all_shards(source, out_root)
    if not shards:
        raise RuntimeError(f"[{source.name}] no parquet shards under {out_root}")
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    # id_str fallback — some shifted repos use `id` or `objid` instead.
    if "id_str" not in df.columns:
        for alt in ("id", "objid", "iauname"):
            if alt in df.columns:
                df["id_str"] = df[alt].astype(str)
                break
        else:
            df["id_str"] = df.index.astype(str)

    threshold = float(cfg_data.labels.clean_sample_threshold)
    df["hubble_type"] = derive_hubble_type(df, threshold=threshold, suffix=source.suffix)
    df = df[df["hubble_type"] != HUBBLE_UNCLASSIFIED].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            f"[{source.name}] no rows survived Hubble-type derivation. "
            f"Check that vote-fraction columns for suffix='{source.suffix}' exist."
        )
    return df


def main() -> None:
    cfg_data = load_config("data")
    cfg_concepts = load_config("concepts")
    cfg_symbolic = load_config("symbolic")
    cfg_conformal = load_config("conformal")
    seed_everything(int(cfg_symbolic.seed))

    rules, ref_classes, feat_cols = _load_exported_rules()
    ref = _reference_metrics_and_calibration(cfg_data, cfg_concepts, cfg_conformal,
                                              rules, feat_cols)

    delta_rows: list[dict] = []
    survey_dir = ensure_dir(Path("results/robustness"))

    for name, source in DEFAULT_SOURCES.items():
        print(f"[robustness] === {name} ({source.hf_repo}) ===")
        try:
            shifted_ds = _load_shifted_dataset(source, cfg_data)
        except Exception as e:
            hint = ""
            if "401" in str(e) or "Unauthorized" in str(e):
                hint = (f"\n[robustness]   → gated repo. Visit "
                        f"https://huggingface.co/datasets/{source.hf_repo} , "
                        "click 'Agree and access', then run "
                        "`uv run huggingface-cli login` (or set HF_TOKEN=hf_...).")
            print(f"[robustness] {name}: failed to load survey ({e}){hint}", file=sys.stderr)
            continue

        try:
            run = run_shifted_pipeline(
                survey_name=name,
                shifted_ds=shifted_ds,
                perceptual_source_suffix=source.suffix,
                reference_rules=rules,
                reference_features=feat_cols,
                calibration_probs=np.zeros((0, len(rules))),
                calibration_labels=pd.Series(dtype=str),
                concepts_cfg=cfg_concepts,
                symbolic_cfg=cfg_symbolic,
                conformal_cfg=cfg_conformal,
                mock_from_ground_truth=True,
            )
        except Exception as e:
            print(f"[robustness] {name}: pipeline failed ({e})", file=sys.stderr)
            continue

        # Cross-survey conformal coverage using the REFERENCE calibration head.
        cov_shift = _shifted_conformal_coverage(ref["head"], ref["est"], run.X_shift, run.y_shift)

        stability = rule_stability(rules, run.rules_refit, feat_cols)

        delta_rows.append(shift_delta_row(
            name, ref["metrics"], run.metrics,
            ece_ref=ref["ece"], ece_shift=run.ece,
            coverage_ref=ref["coverage"], coverage_shift=cov_shift,
            stability=stability,
        ))
        # Persist per-survey artefacts.
        stability.per_class.to_csv(survey_dir / f"{name}_rule_stability.csv", index=False)
        run.shifted_preds.to_parquet(survey_dir / f"{name}_preds.parquet", index=False)

    if not delta_rows:
        print("[robustness] no surveys completed — nothing to write.", file=sys.stderr)
        raise SystemExit(1)

    deltas = pd.DataFrame(delta_rows)
    table_path = Path("results/tables/robustness.csv")
    ensure_dir(table_path.parent)
    deltas.to_csv(table_path, index=False)
    fig_path = shift_figure(deltas, survey_dir / "shift.png")
    note_path = findings_note("DECaLS (gz_desi)", deltas,
                               path=survey_dir / "findings.md")

    write_run_json(
        STAGE,
        seed=int(cfg_symbolic.seed),
        config={"concepts": cfg_concepts, "symbolic": cfg_symbolic,
                "conformal": cfg_conformal},
        extra={
            "reference": {
                "n_val": ref["n_val"], "metrics": ref["metrics"],
                "ece": ref["ece"], "coverage": ref["coverage"],
            },
            "surveys": deltas.to_dict(orient="records"),
            "shift_table_csv": str(table_path),
            "shift_png": str(fig_path),
            "findings_md": str(note_path),
        },
    )
    print()
    print(deltas.to_string(index=False))
    print()
    print(f"[robustness] table: {table_path}")
    print(f"[robustness] figure: {fig_path}")
    print(f"[robustness] note: {note_path}")


if __name__ == "__main__":
    main()
