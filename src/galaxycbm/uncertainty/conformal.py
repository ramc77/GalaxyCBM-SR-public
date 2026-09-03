"""Split-conformal wrapper for the symbolic classifier.

Uses MAPIE 1.4 (`SplitConformalClassifier`, verified via `inspect.signature`):
    SplitConformalClassifier(estimator, confidence_level, conformity_score='lac',
                             prefit=True, ...).conformalize(X_cal, y_cal)
    → .predict_set(X) returns (predictions, prediction_sets_bool)

Everything downstream — coverage, mean set size, per-class coverage,
selective-accuracy vs abstention curve — is pure numpy so it stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.uncertainty.estimator import SymbolicRuleClassifier


# ---------------------------------------------------------------------------
# Conformalisation
# ---------------------------------------------------------------------------


@dataclass
class ConformalHead:
    mapie: object                       # SplitConformalClassifier
    classes: list[str]
    alpha: float
    method: str


def conformalize(
    estimator: SymbolicRuleClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    alpha: float,
    method: str = "lac",
    random_state: int = 0,
) -> ConformalHead:
    from mapie.classification import SplitConformalClassifier

    estimator = estimator.fit(X_cal, y_cal)  # no-op refit; just sets classes_
    mapie = SplitConformalClassifier(
        estimator=estimator,
        confidence_level=1.0 - alpha,
        conformity_score=method,
        prefit=True,
        random_state=random_state,
    )
    mapie.conformalize(X_cal, y_cal)
    return ConformalHead(mapie=mapie, classes=list(estimator.classes_),
                         alpha=alpha, method=method)


def predict_sets(head: ConformalHead, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (point_predictions, set_mask). `set_mask[i, k]` is True iff class k is in the set for row i."""
    preds, sets = head.mapie.predict_set(X)
    sets = np.asarray(sets)
    if sets.ndim == 3:
        # v1.x can return (n, n_classes, n_confidence_levels); we requested one level.
        sets = sets[..., 0]
    return np.asarray(preds), sets.astype(bool)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def coverage_and_set_size(
    head: ConformalHead,
    set_mask: np.ndarray,
    y_true: pd.Series,
) -> dict:
    y = y_true.astype(str).to_numpy()
    class_to_idx = {c: i for i, c in enumerate(head.classes)}
    covered = np.array([set_mask[i, class_to_idx[c]] if c in class_to_idx else False
                        for i, c in enumerate(y)])
    sizes = set_mask.sum(axis=1)
    return {
        "alpha": head.alpha,
        "nominal_coverage": 1.0 - head.alpha,
        "empirical_coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - (1.0 - head.alpha)),
        "n": int(len(y)),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_fraction": float((sizes == 1).mean()),
        "empty_fraction": float((sizes == 0).mean()),
        "method": head.method,
        "classes": head.classes,
    }


def per_class_coverage(
    head: ConformalHead,
    set_mask: np.ndarray,
    y_true: pd.Series,
) -> pd.DataFrame:
    y = y_true.astype(str).to_numpy()
    class_to_idx = {c: i for i, c in enumerate(head.classes)}
    rows: list[dict[str, object]] = []
    for c, idx in class_to_idx.items():
        in_class = y == c
        n = int(in_class.sum())
        cov = float(set_mask[in_class, idx].mean()) if n else float("nan")
        rows.append({
            "class": c, "n": n,
            "coverage": cov,
            "mean_set_size": float(set_mask[in_class].sum(axis=1).mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


def selective_curve(
    estimator: SymbolicRuleClassifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    set_mask: np.ndarray,
    *,
    abstain_when_set_size_ge: int = 2,
) -> pd.DataFrame:
    """Selective accuracy vs. abstention rate.

    Two curves are computed:
      - `by_confidence`: sort by max softmax prob; sweep the accept-fraction.
      - `by_set_size`:   accept when |set| < threshold, threshold = 1..n_classes.
    """
    y = y_true.astype(str).to_numpy()
    probs = estimator.predict_proba(X)
    preds = estimator.classes_[np.argmax(probs, axis=-1)]
    confidence = probs.max(axis=-1)

    rows: list[dict[str, float]] = []
    order = np.argsort(-confidence)  # high confidence first
    n = len(y)
    for k in range(1, n + 1):
        kept = order[:k]
        acc = float(np.mean(preds[kept] == y[kept]))
        rows.append({
            "policy": "by_confidence",
            "abstain_fraction": (n - k) / n,
            "kept_fraction": k / n,
            "accuracy": acc,
        })

    sizes = set_mask.sum(axis=1)
    for threshold in range(1, len(estimator.classes_) + 1):
        keep = sizes < threshold
        if keep.sum() == 0:
            continue
        acc = float(np.mean(preds[keep] == y[keep]))
        rows.append({
            "policy": "by_set_size",
            "abstain_fraction": float(1.0 - keep.mean()),
            "kept_fraction": float(keep.mean()),
            "accuracy": acc,
            "set_size_lt": threshold,
        })
    return pd.DataFrame(rows)


def selective_figure(curve: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    by_c = curve[curve["policy"] == "by_confidence"].sort_values("abstain_fraction")
    ax.plot(by_c["abstain_fraction"], by_c["accuracy"],
            label="rank by max softmax", color="C0")
    by_s = curve[curve["policy"] == "by_set_size"].sort_values("abstain_fraction")
    if not by_s.empty:
        ax.scatter(by_s["abstain_fraction"], by_s["accuracy"],
                   marker="s", label="accept if |set| < k", color="C1", zorder=3)
    ax.set_xlabel("abstention rate")
    ax.set_ylabel("selective accuracy")
    ax.set_title("Selective accuracy vs. abstention")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Class-conditional (Mondrian) conformal prediction
# ---------------------------------------------------------------------------
#
# Split conformal calibrated on a single pooled quantile guarantees only
# MARGINAL coverage. On an imbalanced sample the guarantee is satisfied by
# over-covering the dominant class while every rare class falls far below
# nominal (Section "Calibration" of the paper). The standard fix is Mondrian
# (class-conditional) conformal prediction: calibrate one quantile per true
# class, using only the calibration points of that class.
#
# For class k with n_k calibration points the finite-sample-valid level is
#     lvl_k = ceil((n_k + 1)(1 - alpha)) / n_k .
# When n_k < ceil(1/alpha) - 1 this exceeds 1: no finite quantile of n_k
# points can certify 1 - alpha coverage, and the only valid choice is
# q_k = +inf, i.e. class k enters EVERY prediction set. That degeneracy is
# not a defect of the method — it is the honest statement that eight
# calibration galaxies cannot underwrite a 90% guarantee.


@dataclass
class MondrianHead:
    """Per-class LAC thresholds. `q[k]` may be +inf (see module note)."""

    classes: list[str]
    quantiles: dict[str, float]
    n_cal: dict[str, int]
    levels: dict[str, float]
    degenerate: list[str]        # classes whose quantile had to be +inf
    alpha: float
    method: str = "lac"


def min_calibration_points(alpha: float) -> int:
    """Smallest n_k for which a finite class-conditional quantile exists."""
    return int(np.ceil(1.0 / float(alpha))) - 1


def mondrian_conformalize(
    estimator: SymbolicRuleClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    alpha: float,
) -> MondrianHead:
    """Calibrate one LAC threshold per true class."""
    estimator = estimator.fit(X_cal, y_cal)
    classes = [str(c) for c in estimator.classes_]
    probs = estimator.predict_proba(X_cal)
    y = y_cal.astype(str).to_numpy()

    quantiles: dict[str, float] = {}
    n_cal: dict[str, int] = {}
    levels: dict[str, float] = {}
    degenerate: list[str] = []

    for idx, k in enumerate(classes):
        in_k = y == k
        n_k = int(in_k.sum())
        n_cal[k] = n_k
        if n_k == 0:
            quantiles[k], levels[k] = float("inf"), float("nan")
            degenerate.append(k)
            continue
        scores = 1.0 - probs[in_k, idx]          # LAC nonconformity
        level = np.ceil((n_k + 1) * (1.0 - alpha)) / n_k
        levels[k] = float(level)
        if level > 1.0:
            quantiles[k] = float("inf")
            degenerate.append(k)
        else:
            quantiles[k] = float(np.quantile(scores, level, method="higher"))

    return MondrianHead(classes=classes, quantiles=quantiles, n_cal=n_cal,
                        levels=levels, degenerate=degenerate, alpha=float(alpha))


def mondrian_predict_sets(
    head: MondrianHead,
    estimator: SymbolicRuleClassifier,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (point predictions, set mask) under per-class thresholds."""
    probs = estimator.predict_proba(X)
    preds = np.asarray(estimator.classes_)[np.argmax(probs, axis=-1)]
    mask = np.zeros_like(probs, dtype=bool)
    for idx, k in enumerate(head.classes):
        mask[:, idx] = (1.0 - probs[:, idx]) <= head.quantiles[k]
    return preds, mask


def mondrian_report(
    head: MondrianHead,
    set_mask: np.ndarray,
    y_true: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """Summary + per-class table for a Mondrian head, matching the marginal API."""
    y = y_true.astype(str).to_numpy()
    idx_of = {c: i for i, c in enumerate(head.classes)}
    covered = np.array([set_mask[i, idx_of[c]] if c in idx_of else False
                        for i, c in enumerate(y)])
    sizes = set_mask.sum(axis=1)

    summary = {
        "alpha": head.alpha,
        "nominal_coverage": 1.0 - head.alpha,
        "empirical_coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - (1.0 - head.alpha)),
        "n": int(len(y)),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_fraction": float((sizes == 1).mean()),
        "empty_fraction": float((sizes == 0).mean()),
        "method": f"mondrian-{head.method}",
        "classes": head.classes,
        "degenerate_classes": head.degenerate,
        "min_calibration_points": min_calibration_points(head.alpha),
    }

    rows: list[dict[str, object]] = []
    for k in head.classes:
        in_k = y == k
        n = int(in_k.sum())
        rows.append({
            "class": k,
            "n": n,
            "n_calibration": head.n_cal[k],
            "quantile": head.quantiles[k],
            "degenerate": k in head.degenerate,
            "coverage": float(set_mask[in_k, idx_of[k]].mean()) if n else float("nan"),
            "mean_set_size": float(set_mask[in_k].sum(axis=1).mean()) if n else float("nan"),
        })
    return summary, pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
