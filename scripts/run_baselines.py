"""Stage 7 entry point: run (a)-(d) baselines on the SAME splits/metrics.

Any of the heavy baselines skipped on a machine that lacks their extra get
a row in the comparison table marked `skipped: <reason>` so the audit trail
is honest about what actually ran.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from galaxycbm.baselines import (
    comparison_row,
    comparison_table,
    concept_fidelity,
    linear_concept_importance,
    linear_interpretability_cost,
    skipped_row,
    symbolic_concept_weights,
    symbolic_interpretability_cost,
    train_linear_cbm,
)
from galaxycbm.baselines.data import load_baseline_data
from galaxycbm.symbolic import predict_labels
from galaxycbm.utils import load_config, seed_everything, write_run_json
from galaxycbm.utils.io import ensure_dir, write_json

STAGE = "baselines"


def main() -> None:
    concepts_cfg = load_config("concepts")
    seed_everything(0)

    X_train, y_train, X_val, y_val, feat_cols, rules, classes, tr, va, dataset = \
        load_baseline_data(concepts_cfg)

    rows: list[dict] = []
    importances: dict[str, pd.Series] = {}

    # -- symbolic head (Stage 2) ---------------------------------------------
    sym_preds = predict_labels(rules, X_val)
    rows.append(comparison_row(
        "galaxycbm_symbolic", y_val, sym_preds,
        interpretability_cost=symbolic_interpretability_cost(rules),
        interpretability_kind="intrinsic",
    ))
    sym_w = symbolic_concept_weights(rules)
    importances["symbolic"] = sym_w

    # -- dense-linear CBM ----------------------------------------------------
    lin_est, _ = train_linear_cbm(X_train, y_train, X_val, y_val, seed=0)
    lin_pred = pd.Series(lin_est.predict(X_val), index=X_val.index)
    rows.append(comparison_row(
        "dense_linear_cbm", y_val, lin_pred,
        interpretability_cost=linear_interpretability_cost(lin_est),
        interpretability_kind="intrinsic",
    ))
    importances["linear"] = linear_concept_importance(lin_est)

    # -- XGBoost on concepts (baselines extra) -------------------------------
    # Isolated in a subprocess (scripts/_xgb_baseline_worker.py): numba (a
    # SHAP/XGBoost dependency) has hit native segfaults on some macOS/Apple
    # Silicon setups, which no Python try/except can catch since it kills
    # the whole interpreter. A crash in the child costs one row here, not
    # the symbolic/linear results already computed above.
    xgb_result_path = ensure_dir(Path("results/baselines")) / "_xgb_worker_result.json"
    xgb_result_path.unlink(missing_ok=True)
    worker = Path(__file__).parent / "_xgb_baseline_worker.py"
    proc = subprocess.run(
        [sys.executable, str(worker), "--out", str(xgb_result_path)],
        capture_output=True, text=True,
    )
    if proc.returncode == 0 and xgb_result_path.exists():
        xgb_result = json.loads(xgb_result_path.read_text())
        m = xgb_result["metrics"]
        rows.append({
            "model": "xgboost_concepts",
            "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "cohen_kappa": m["cohen_kappa"],
            "interpretability_cost": xgb_result["interpretability_cost"],
            "interpretability_kind": "post-hoc-required",
            "note": "",
        })
        importances["xgb"] = pd.Series(xgb_result["importance"])
        if xgb_result.get("shap_importance"):
            importances["xgb_shap"] = pd.Series(xgb_result["shap_importance"])
    else:
        crashed = proc.returncode is not None and proc.returncode < 0
        reason = f"subprocess exit {proc.returncode}" + (" (native crash)" if crashed else "")
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-5:]) if proc.stderr else ""
        print(f"[baselines] xgboost worker failed: {reason}", file=sys.stderr)
        if stderr_tail:
            print(stderr_tail, file=sys.stderr)
        rows.append(skipped_row("xgboost_concepts", reason))

    # -- End-to-end ConvNeXt + SmoothGrad (stage1 extra) --------------------
    try:
        import torch  # noqa: F401
        from galaxycbm.baselines.cnn import (
            interpretability_cost as cnn_cost,
            smoothgrad_saliency,
            train_endtoend_convnext,
        )
        from galaxycbm.data.cutout_cache import cache_cutouts

        data_cfg = load_config("data")
        model_cfg = load_config("model")
        raw_root = Path(data_cfg.download.root) / "gz_evo"
        shards = sorted(raw_root.rglob("*.parquet"))
        cutouts_root = Path("data/interim/cutouts")
        needed = set(pd.concat([tr, va])["id_str"].astype(str))
        cache_cutouts(shards, cutouts_root, size=224, ids=needed)

        splits_dict = {"train": tr, "val": va}
        module, cnn_metrics, cnn_preds = train_endtoend_convnext(
            dataset, splits_dict, cutouts_root, model_cfg, seed=0,
        )
        rows.append(comparison_row(
            "endtoend_convnext",
            va.set_index("id_str").loc[cnn_preds["id_str"], "hubble_type"].reset_index(drop=True),
            cnn_preds["hubble_pred"],
            interpretability_cost=cnn_cost(module),
            interpretability_kind="black-box",
        ))
        try:
            smoothgrad_saliency(module, va, cutouts_root,
                                out_path="results/baselines/smoothgrad.png")
        except Exception as e:
            print(f"[baselines] SmoothGrad skipped: {e}", file=sys.stderr)
    except ImportError as e:
        rows.append(skipped_row("endtoend_convnext", str(e)))
        rows.append(skipped_row("endtoend_smoothgrad", "requires stage1 extra"))

    # ------------------------------------------------------------------------
    tbl = comparison_table(rows)
    tables_dir = ensure_dir(Path("results/tables"))
    tbl.to_csv(tables_dir / "comparison.csv", index=False)

    # Fidelity: symbolic (intrinsic) vs everyone else's per-concept importance.
    fidelity_rows: list[dict] = []
    for name, imp in importances.items():
        if name == "symbolic":
            continue
        stats = concept_fidelity(sym_w, imp)
        fidelity_rows.append({"vs_symbolic": name, **stats})
    fidelity_df = pd.DataFrame(fidelity_rows)
    fidelity_path = Path("results/baselines/fidelity.csv")
    ensure_dir(fidelity_path.parent)
    fidelity_df.to_csv(fidelity_path, index=False)

    write_run_json(
        STAGE,
        seed=0,
        config={"concepts": concepts_cfg},
        extra={
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_features": int(len(feat_cols)),
            "comparison_csv": str(tables_dir / "comparison.csv"),
            "fidelity_csv": str(fidelity_path),
            "rows": tbl.to_dict(orient="records"),
        },
    )
    print(tbl.to_string(index=False))
    print()
    print("Fidelity vs symbolic (Spearman / Pearson):")
    print(fidelity_df.to_string(index=False) if not fidelity_df.empty else "  (no baselines produced importance vectors)")


if __name__ == "__main__":
    main()
