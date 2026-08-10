"""Split-conformal wrapper — offline tests.

Uses synthetic ClassRules (linear in one feature per class) so MAPIE gets a
real prefit sklearn estimator with predict_proba, and empirical coverage is
easy to reason about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxycbm.symbolic import ClassRule
from galaxycbm.uncertainty import (
    SymbolicRuleClassifier,
    conformalize,
    coverage_and_set_size,
    per_class_coverage,
    predict_sets,
    selective_curve,
    selective_figure,
)


def _toy_dataset(n: int, rng) -> tuple[pd.DataFrame, pd.Series, list[ClassRule]]:
    """3-class problem: y ∈ {A, B, C}. Feature f_A, f_B, f_C are noisy indicators."""
    y = rng.choice(["A", "B", "C"], size=n)
    fA = (y == "A").astype(float) + rng.normal(0, 0.3, size=n)
    fB = (y == "B").astype(float) + rng.normal(0, 0.3, size=n)
    fC = (y == "C").astype(float) + rng.normal(0, 0.3, size=n)
    X = pd.DataFrame({"fA": fA, "fB": fB, "fC": fC})
    rules = [
        ClassRule(hubble_class="A", equation_str="fA", latex="fA",
                  complexity=1, pysr_score=1.0, cv_accuracy=0.9),
        ClassRule(hubble_class="B", equation_str="fB", latex="fB",
                  complexity=1, pysr_score=1.0, cv_accuracy=0.9),
        ClassRule(hubble_class="C", equation_str="fC", latex="fC",
                  complexity=1, pysr_score=1.0, cv_accuracy=0.9),
    ]
    return X, pd.Series(y), rules


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


def test_estimator_predict_proba_sums_to_one():
    rng = np.random.default_rng(0)
    X, y, rules = _toy_dataset(100, rng)
    est = SymbolicRuleClassifier(rules=rules, feature_columns=list(X.columns))
    est.fit(X, y)
    p = est.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert p.shape == (len(X), len(rules))


def test_estimator_predict_matches_argmax_proba():
    rng = np.random.default_rng(0)
    X, y, rules = _toy_dataset(50, rng)
    est = SymbolicRuleClassifier(rules=rules, feature_columns=list(X.columns))
    est.fit(X, y)
    p = est.predict_proba(X)
    pred = est.predict(X)
    np.testing.assert_array_equal(pred, est.classes_[np.argmax(p, axis=1)])


# ---------------------------------------------------------------------------
# Conformal coverage
# ---------------------------------------------------------------------------


def test_conformal_hits_nominal_coverage_on_large_calibration():
    rng = np.random.default_rng(42)
    X_cal, y_cal, rules = _toy_dataset(2000, rng)
    X_tst, y_tst, _ = _toy_dataset(2000, rng)
    est = SymbolicRuleClassifier(rules=rules, feature_columns=list(X_cal.columns))
    head = conformalize(est, X_cal, y_cal, alpha=0.10, method="lac", random_state=0)
    _, set_mask = predict_sets(head, X_tst)
    summary = coverage_and_set_size(head, set_mask, y_tst)
    # Finite-sample: expect coverage close to 0.90; wider tolerance for one seed.
    assert 0.86 < summary["empirical_coverage"] < 0.94, summary


def test_higher_alpha_shrinks_mean_set_size():
    rng = np.random.default_rng(1)
    X_cal, y_cal, rules = _toy_dataset(1500, rng)
    X_tst, y_tst, _ = _toy_dataset(1500, rng)
    est = SymbolicRuleClassifier(rules=rules, feature_columns=list(X_cal.columns))
    head_tight = conformalize(est, X_cal, y_cal, alpha=0.40)
    head_wide  = conformalize(est, X_cal, y_cal, alpha=0.05)
    _, mask_t = predict_sets(head_tight, X_tst)
    _, mask_w = predict_sets(head_wide, X_tst)
    assert coverage_and_set_size(head_tight, mask_t, y_tst)["mean_set_size"] < \
           coverage_and_set_size(head_wide,  mask_w, y_tst)["mean_set_size"]


# ---------------------------------------------------------------------------
# Per-class coverage + selective curve
# ---------------------------------------------------------------------------


def test_per_class_coverage_rows_match_classes():
    rng = np.random.default_rng(2)
    X_cal, y_cal, rules = _toy_dataset(600, rng)
    X_tst, y_tst, _ = _toy_dataset(600, rng)
    est = SymbolicRuleClassifier(rules=rules, feature_columns=list(X_cal.columns))
    head = conformalize(est, X_cal, y_cal, alpha=0.15)
    _, mask = predict_sets(head, X_tst)
    per = per_class_coverage(head, mask, y_tst)
    assert set(per["class"]) == set(head.classes)
    assert (per["coverage"].between(0, 1)).all()


def test_selective_curve_shape_and_monotone_on_average():
    rng = np.random.default_rng(3)
    X_cal, y_cal, rules = _toy_dataset(1200, rng)
    X_tst, y_tst, _ = _toy_dataset(1200, rng)
    est = SymbolicRuleClassifier(rules=rules, feature_columns=list(X_cal.columns))
    head = conformalize(est, X_cal, y_cal, alpha=0.10)
    _, mask = predict_sets(head, X_tst)
    curve = selective_curve(est, X_tst, y_tst, mask)
    assert set(curve["policy"].unique()) >= {"by_confidence"}
    by_c = curve[curve["policy"] == "by_confidence"].sort_values("kept_fraction")
    # Averaged over the top-10% most confident vs the whole sample: the
    # confident bucket must be at least as accurate. Single-point comparisons
    # are too noisy — averaging is what selective classification is for.
    top = by_c[by_c["kept_fraction"] <= 0.1]["accuracy"].mean()
    full = by_c[by_c["kept_fraction"] == 1.0]["accuracy"].iloc[0]
    assert top >= full - 1e-9


def test_selective_figure_writes(tmp_path):
    rng = np.random.default_rng(4)
    X_cal, y_cal, rules = _toy_dataset(200, rng)
    X_tst, y_tst, _ = _toy_dataset(200, rng)
    est = SymbolicRuleClassifier(rules=rules, feature_columns=list(X_cal.columns))
    head = conformalize(est, X_cal, y_cal, alpha=0.1)
    _, mask = predict_sets(head, X_tst)
    p = selective_figure(selective_curve(est, X_tst, y_tst, mask), tmp_path / "sel.png")
    assert p.exists() and p.stat().st_size > 0
