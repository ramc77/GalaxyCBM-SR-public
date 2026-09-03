"""Class-conditional (Mondrian) conformal calibration.

Checks the property the reported result rests on: per-class coverage is at
or above nominal for every class with enough calibration points, and the
degenerate case (too few points) is reported rather than silently under-covering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxycbm.symbolic import ClassRule
from galaxycbm.uncertainty import (
    SymbolicRuleClassifier,
    min_calibration_points,
    mondrian_conformalize,
    mondrian_predict_sets,
    mondrian_report,
)

FEATURES = ["f0", "f1", "f2"]
CLASSES = ["A", "B", "C"]


def _rules() -> list[ClassRule]:
    return [
        ClassRule(hubble_class="A", equation_str="f0", latex="", complexity=1,
                  pysr_score=0.0, cv_accuracy=0.0),
        ClassRule(hubble_class="B", equation_str="f1", latex="", complexity=1,
                  pysr_score=0.0, cv_accuracy=0.0),
        ClassRule(hubble_class="C", equation_str="f2", latex="", complexity=1,
                  pysr_score=0.0, cv_accuracy=0.0),
    ]


def _imbalanced_sample(n_a: int, n_b: int, n_c: int, seed: int = 0):
    """One-hot-ish features with noise, so each class is separable but not perfectly."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for cls_idx, n in enumerate((n_a, n_b, n_c)):
        base = np.zeros((n, 3))
        base[:, cls_idx] = 1.0
        rows.append(base + rng.normal(0, 0.6, size=(n, 3)))
        labels += [CLASSES[cls_idx]] * n
    X = pd.DataFrame(np.vstack(rows), columns=FEATURES)
    return X, pd.Series(labels, name="y")


def test_min_calibration_points():
    assert min_calibration_points(0.10) == 9
    assert min_calibration_points(0.05) == 19


def test_per_class_coverage_reaches_nominal_when_calibration_is_sufficient():
    alpha = 0.10
    X_cal, y_cal = _imbalanced_sample(400, 200, 200, seed=1)
    X_te, y_te = _imbalanced_sample(400, 200, 200, seed=2)
    est = SymbolicRuleClassifier(rules=_rules(), feature_columns=FEATURES).fit(X_cal, y_cal)

    head = mondrian_conformalize(est, X_cal, y_cal, alpha=alpha)
    assert head.degenerate == []
    _, mask = mondrian_predict_sets(head, est, X_te)
    _, per_class = mondrian_report(head, mask, y_te)

    # Finite-sample slack: coverage is guaranteed in expectation, so allow a
    # small band around nominal on a single draw.
    for cov in per_class["coverage"]:
        assert cov >= (1.0 - alpha) - 0.05


def test_rare_class_degenerates_to_full_inclusion_rather_than_undercovering():
    """With n_cal < ceil(1/alpha) - 1 no finite quantile is valid."""
    alpha = 0.10
    X_cal, y_cal = _imbalanced_sample(400, 200, 5, seed=3)
    X_te, y_te = _imbalanced_sample(200, 100, 40, seed=4)
    est = SymbolicRuleClassifier(rules=_rules(), feature_columns=FEATURES).fit(X_cal, y_cal)

    head = mondrian_conformalize(est, X_cal, y_cal, alpha=alpha)
    assert "C" in head.degenerate
    assert np.isinf(head.quantiles["C"])

    _, mask = mondrian_predict_sets(head, est, X_te)
    summary, per_class = mondrian_report(head, mask, y_te)

    # A degenerate class is in every set, so its coverage is exactly 1.
    assert mask[:, head.classes.index("C")].all()
    assert per_class.loc[per_class["class"] == "C", "coverage"].iloc[0] == pytest.approx(1.0)
    assert summary["degenerate_classes"] == head.degenerate


def test_mondrian_sets_are_never_smaller_on_average_than_they_must_be():
    """Conditional validity is bought with set size; the report exposes both."""
    alpha = 0.10
    X_cal, y_cal = _imbalanced_sample(400, 60, 40, seed=5)
    X_te, y_te = _imbalanced_sample(200, 30, 20, seed=6)
    est = SymbolicRuleClassifier(rules=_rules(), feature_columns=FEATURES).fit(X_cal, y_cal)

    head = mondrian_conformalize(est, X_cal, y_cal, alpha=alpha)
    _, mask = mondrian_predict_sets(head, est, X_te)
    summary, _ = mondrian_report(head, mask, y_te)

    assert 1.0 <= summary["mean_set_size"] <= len(CLASSES)
    assert summary["method"] == "mondrian-lac"
    assert summary["n"] == len(y_te)
