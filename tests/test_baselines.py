"""Baselines — linear CBM + intrinsic-weight + fidelity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
from galaxycbm.symbolic import ClassRule


def _toy(n: int, rng):
    y = rng.choice(["A", "B", "C"], size=n)
    fA = (y == "A").astype(float) + rng.normal(0, 0.3, n)
    fB = (y == "B").astype(float) + rng.normal(0, 0.3, n)
    fC = (y == "C").astype(float) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"fA": fA, "fB": fB, "fC": fC}), pd.Series(y)


def test_linear_cbm_trains_and_metrics_shape():
    rng = np.random.default_rng(0)
    Xtr, ytr = _toy(300, rng)
    Xva, yva = _toy(200, rng)
    est, metrics = train_linear_cbm(Xtr, ytr, Xva, yva, seed=0)
    for k in ("accuracy", "macro_f1", "cohen_kappa", "n", "labels", "confusion_matrix"):
        assert k in metrics
    assert metrics["accuracy"] > 0.5   # 3-class, well-separated → easy


def test_linear_concept_importance_matches_input_features():
    rng = np.random.default_rng(1)
    Xtr, ytr = _toy(200, rng)
    est, _ = train_linear_cbm(Xtr, ytr, Xtr, ytr, seed=0)
    w = linear_concept_importance(est)
    assert list(w.index) == list(Xtr.columns)
    assert (w >= 0).all()


def test_linear_interpretability_cost_is_nonzero_count():
    rng = np.random.default_rng(1)
    Xtr, ytr = _toy(200, rng)
    est, _ = train_linear_cbm(Xtr, ytr, Xtr, ytr, seed=0)
    assert linear_interpretability_cost(est) > 0


def _rules() -> list[ClassRule]:
    return [
        ClassRule("A", "fA", "fA", 1, 1.0, 0.9),
        ClassRule("B", "fB - 0.5*fA", "fB - 0.5 fA", 5, 1.0, 0.8),
        ClassRule("C", "fC", "fC", 1, 1.0, 0.7),
    ]


def test_symbolic_concept_weights_sums_by_feature():
    w = symbolic_concept_weights(_rules())
    # fA appears in rules A (cv=0.9) and B (cv=0.8) → 1.7
    # fB appears in B (0.8), fC in C (0.7)
    assert w["fA"] == pytest.approx(0.9 + 0.8)
    assert w["fB"] == pytest.approx(0.8)
    assert w["fC"] == pytest.approx(0.7)


def test_symbolic_interpretability_cost_sums_complexity():
    assert symbolic_interpretability_cost(_rules()) == 1 + 5 + 1


def test_concept_fidelity_perfect_positive_correlation():
    a = pd.Series({"fA": 3.0, "fB": 2.0, "fC": 1.0})
    b = pd.Series({"fA": 30.0, "fB": 20.0, "fC": 10.0})
    stats = concept_fidelity(a, b)
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["pearson"] == pytest.approx(1.0)


def test_concept_fidelity_returns_nan_on_too_few_shared():
    a = pd.Series({"fA": 1.0, "fB": 2.0})
    b = pd.Series({"fA": 10.0})   # only 1 shared
    stats = concept_fidelity(a, b)
    assert np.isnan(stats["spearman"]) and np.isnan(stats["pearson"])
    assert stats["n_shared"] == 1


def test_concept_fidelity_nan_on_constant_series():
    a = pd.Series({"fA": 5.0, "fB": 5.0, "fC": 5.0})
    b = pd.Series({"fA": 1.0, "fB": 2.0, "fC": 3.0})
    stats = concept_fidelity(a, b)
    assert np.isnan(stats["spearman"])


def test_comparison_row_and_table_shape():
    yt = pd.Series(["A", "A", "B"])
    yp = pd.Series(["A", "B", "B"])
    row = comparison_row("test_model", yt, yp,
                          interpretability_cost=5, interpretability_kind="intrinsic")
    tbl = comparison_table([row, skipped_row("stage1_dep", "torch missing")])
    assert list(tbl["model"]) == ["test_model", "stage1_dep"]
    assert set(tbl.columns) == {
        "model", "n", "accuracy", "macro_f1", "cohen_kappa",
        "interpretability_cost", "interpretability_kind", "note",
    }
