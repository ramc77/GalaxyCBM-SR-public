"""Apply the frozen Stage-1 → Stage-2 → Stage-3 pipeline on a shifted survey,
and refit PySR on the shifted survey to measure rule stability.

Stage-1 inference on the shifted images needs torch/zoobot; when they're
missing we fall back to `mock_from_ground_truth=True`, using the shifted
survey's own GZ vote fractions in place of predicted concepts. That gives a
ceiling for the shifted survey — the real experiment overwrites this on a
Linux/GPU box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.concepts import build_head_specs
from galaxycbm.symbolic import ClassRule, build_features, compute_metrics, predict_labels
from galaxycbm.uncertainty import (
    SymbolicRuleClassifier,
    conformalize,
    coverage_and_set_size,
    predict_sets,
)


@dataclass
class ShiftedRun:
    survey: str
    n: int
    metrics: dict                 # accuracy, macro_f1, cohen_kappa, ...
    ece: float
    conformal_coverage: float
    shifted_preds: pd.DataFrame   # rows aligned with X_shift, includes id_str
    X_shift: pd.DataFrame
    y_shift: pd.Series
    rules_refit: list[ClassRule]  # PySR refit on shifted data


def _mock_preds_from_shifted(
    shifted_ds: pd.DataFrame,
    heads,
    *,
    perceptual_source_suffix: str,
) -> pd.DataFrame:
    """Build a preds-parquet-shaped DataFrame from the shifted survey's own votes.

    For classification heads we translate `<task>-<suffix>_<answer>_fraction`
    columns into `<head>__<class>` columns; regression heads pass through.
    """
    from galaxycbm.data.labels import DR5_TASK_ANSWERS

    out = pd.DataFrame({"id_str": shifted_ds["id_str"].astype(str).to_numpy()})
    for h in heads:
        if h.kind == "classification":
            # Map the head's canonical answers into the shifted suffix's vote
            # fractions; falls back to NaN if the shifted survey lacks a task.
            for cls in h.classes or ():
                col_candidates = [
                    f"{h.name}-{perceptual_source_suffix}_{cls}_fraction",
                    f"{h.name}-{perceptual_source_suffix}_{cls}-shaped_fraction",
                ]
                found = next((c for c in col_candidates if c in shifted_ds.columns), None)
                out[f"{h.name}__{cls}"] = (
                    shifted_ds[found].astype(float).to_numpy()
                    if found else np.full(len(shifted_ds), np.nan)
                )
        else:
            out[h.name] = (
                shifted_ds[h.name].astype(float).to_numpy()
                if h.name in shifted_ds.columns else np.full(len(shifted_ds), np.nan)
            )
    return out


def _load_reference_features(
    ref_preds: pd.DataFrame,
    ref_dataset: pd.DataFrame,
    heads,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    X, feat_spec = build_features(ref_preds, heads)
    X["id_str"] = ref_preds["id_str"].astype(str).to_numpy()
    joined = X.merge(ref_dataset[["id_str", "hubble_type"]].astype({"id_str": str}),
                     on="id_str", how="inner")
    return joined[feat_spec.columns], joined["hubble_type"], feat_spec.columns


def _refit_pysr_on_shifted(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    symbolic_cfg,
) -> list[ClassRule]:
    """Refit PySR on a shifted survey. Subsample to keep wall time sane —
    rule stability is a structural metric, so a few thousand rows is enough
    to see whether PySR lands on the same features.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    from galaxycbm.symbolic import fit_symbolic

    cap = int(symbolic_cfg.get("robustness_max_train_rows", 3000))
    if len(X_train) > cap:
        rng_seed = int(symbolic_cfg.seed)
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=rng_seed)
        keep_idx, _ = next(splitter.split(X_train.values, y_train.astype(str).values))
        X_train = X_train.iloc[keep_idx].reset_index(drop=True)
        y_train = y_train.iloc[keep_idx].reset_index(drop=True)
    return fit_symbolic(X_train, y_train, symbolic_cfg).rules


def run_shifted_pipeline(
    survey_name: str,
    shifted_ds: pd.DataFrame,          # dataset.parquet-shaped (id_str + labels + concepts)
    perceptual_source_suffix: str,     # e.g. "euclid", "jwst"
    reference_rules: list[ClassRule],
    reference_features: list[str],
    calibration_probs: np.ndarray,     # for conformal reuse
    calibration_labels: pd.Series,     # y_calibration from reference survey
    concepts_cfg,
    symbolic_cfg,
    conformal_cfg,
    *,
    mock_from_ground_truth: bool = True,
) -> ShiftedRun:
    heads = build_head_specs(concepts_cfg)
    # 1. Turn the shifted survey into a preds-shaped DataFrame.
    if mock_from_ground_truth:
        preds = _mock_preds_from_shifted(
            shifted_ds, heads, perceptual_source_suffix=perceptual_source_suffix,
        )
    else:
        raise NotImplementedError(
            "Stage-1 inference on shifted images requires the `stage1` extra "
            "(torch/zoobot). Re-run from a Linux/GPU box with the trained CBM."
        )

    # 2. Build the same feature matrix Stage 2 saw.
    X, feat_spec = build_features(preds, heads)
    X["id_str"] = preds["id_str"].to_numpy()
    joined = X.merge(shifted_ds[["id_str", "hubble_type"]].astype({"id_str": str}),
                     on="id_str", how="inner")
    for c in reference_features:
        if c not in joined.columns:
            joined[c] = 0.0
    X_shift = joined[reference_features].copy()
    y_shift = joined["hubble_type"]

    medians = X_shift.median(numeric_only=True).fillna(0.0)
    X_shift = X_shift.fillna(medians).fillna(0.0)

    # Drop rows whose true class isn't in the frozen rules' vocabulary.
    ref_classes = [r.hubble_class for r in reference_rules]
    keep = y_shift.astype(str).isin(ref_classes)
    X_shift, y_shift = X_shift[keep].reset_index(drop=True), y_shift[keep].reset_index(drop=True)

    if X_shift.empty:
        raise RuntimeError(f"[{survey_name}] no rows survived filtering to reference classes.")

    # 3. Frozen Stage-2 prediction + metrics.
    y_pred = predict_labels(reference_rules, X_shift)
    metrics = compute_metrics(y_shift, y_pred)

    # 4. Frozen Stage-3 conformal head applied to shifted data.
    est = SymbolicRuleClassifier(rules=reference_rules, feature_columns=reference_features)
    est.fit(X_shift, y_shift)   # no-op fit (just sets classes_)
    from galaxycbm.robustness.metrics import expected_calibration_error

    probs_shift = est.predict_proba(X_shift)
    ece = expected_calibration_error(probs_shift, y_shift, list(est.classes_))

    head = conformalize(
        est,
        pd.DataFrame(np.zeros((len(calibration_labels), len(reference_features))),
                      columns=reference_features),
        calibration_labels,
        alpha=float(conformal_cfg.alpha),
        method=str(conformal_cfg.method),
        random_state=int(conformal_cfg.seed),
    ) if False else None  # placeholder — proper calibration handled outside

    # We conformalise on the actual reference calibration probabilities via the
    # coverage_and_set_size helper — take the shifted probs against the shifted
    # true labels using the reference q_hat threshold. Simpler: reuse the same
    # SplitConformalClassifier machinery but with the reference calibration data.
    # For readability, do it in the driver where reference X_cal / y_cal live.
    conformal_coverage = float("nan")  # filled by the driver

    # 5. Refit PySR on the shifted survey to measure rule stability.
    rules_refit: list[ClassRule] = _refit_pysr_on_shifted(X_shift, y_shift, symbolic_cfg)

    preds_df = pd.DataFrame({
        "id_str": joined.loc[keep, "id_str"].reset_index(drop=True),
        "hubble_true": y_shift.reset_index(drop=True),
        "hubble_pred": y_pred.reset_index(drop=True),
    })
    return ShiftedRun(
        survey=survey_name,
        n=int(len(X_shift)),
        metrics=metrics,
        ece=ece,
        conformal_coverage=conformal_coverage,
        shifted_preds=preds_df,
        X_shift=X_shift,
        y_shift=y_shift,
        rules_refit=rules_refit,
    )
