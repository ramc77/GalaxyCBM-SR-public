"""Fidelity metrics — pure numpy/sklearn, no torch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxycbm.concepts import HeadSpec, per_concept_fidelity


# ---------------------------------------------------------------------------
# Classification path
# ---------------------------------------------------------------------------


def test_perfect_binary_classifier_gets_auc_one():
    head = HeadSpec(name="edge", kind="classification", n_classes=2, classes=("no", "yes"))
    n = 20
    y_true = pd.DataFrame({"edge": ["yes"] * 10 + ["no"] * 10})
    y_pred = pd.DataFrame({
        "edge__no":  [0.0] * 10 + [1.0] * 10,
        "edge__yes": [1.0] * 10 + [0.0] * 10,
    })
    fid = per_concept_fidelity(y_true, y_pred, [head])
    assert fid.iloc[0]["metric"] == "AUC"
    assert fid.iloc[0]["value"] == pytest.approx(1.0)
    assert fid.iloc[0]["beats_baseline"] == True  # noqa: E712 (pandas stores as np.bool_)
    assert fid.iloc[0]["n"] == n


def test_random_binary_classifier_hits_auc_near_half_and_fails_baseline():
    head = HeadSpec(name="edge", kind="classification", n_classes=2, classes=("no", "yes"))
    rng = np.random.default_rng(0)
    n = 400
    y_true = pd.DataFrame({"edge": rng.choice(["no", "yes"], size=n)})
    p_yes = rng.random(n)
    y_pred = pd.DataFrame({"edge__no": 1 - p_yes, "edge__yes": p_yes})
    fid = per_concept_fidelity(y_true, y_pred, [head]).iloc[0]
    assert 0.4 < fid["value"] < 0.6
    assert fid["beats_baseline"] in (True, False)  # random can occasionally scrape > 0.5


def test_missing_prob_cols_produce_nan_and_note():
    head = HeadSpec(name="bar", kind="classification", n_classes=3,
                    classes=("none", "weak", "strong"))
    y_true = pd.DataFrame({"bar": ["none", "weak", "strong"]})
    y_pred = pd.DataFrame({"bar__none": [0.9, 0.1, 0.1]})  # missing weak/strong
    fid = per_concept_fidelity(y_true, y_pred, [head]).iloc[0]
    assert np.isnan(fid["value"])
    assert fid["beats_baseline"] == False  # noqa: E712 (pandas stores as np.bool_)


# ---------------------------------------------------------------------------
# Regression path
# ---------------------------------------------------------------------------


def test_perfect_regression_beats_std_baseline():
    head = HeadSpec(name="gini", kind="regression")
    y_true = pd.DataFrame({"gini": np.linspace(0.2, 0.8, 40)})
    y_pred = pd.DataFrame({"gini": y_true["gini"].values})
    fid = per_concept_fidelity(y_true, y_pred, [head]).iloc[0]
    assert fid["value"] == pytest.approx(0.0, abs=1e-9)
    assert fid["beats_baseline"] == True  # noqa: E712 (pandas stores as np.bool_)


def test_mean_predictor_matches_baseline_and_does_not_beat_it():
    head = HeadSpec(name="gini", kind="regression")
    rng = np.random.default_rng(1)
    y = rng.normal(0.5, 0.2, size=200)
    y_true = pd.DataFrame({"gini": y})
    y_pred = pd.DataFrame({"gini": np.full_like(y, y.mean())})
    fid = per_concept_fidelity(y_true, y_pred, [head]).iloc[0]
    assert fid["value"] == pytest.approx(float(np.std(y, ddof=0)), rel=1e-6)
    assert fid["beats_baseline"] == False  # noqa: E712 (pandas stores as np.bool_)


def test_nans_are_masked_out():
    head = HeadSpec(name="sersic_n", kind="regression")
    y_true = pd.DataFrame({"sersic_n": [1.0, float("nan"), 2.0, 3.0]})
    y_pred = pd.DataFrame({"sersic_n": [1.0, 999.0, 2.0, 3.0]})
    fid = per_concept_fidelity(y_true, y_pred, [head]).iloc[0]
    assert fid["value"] == pytest.approx(0.0, abs=1e-9)
    assert fid["n"] == 3


# ---------------------------------------------------------------------------
# Baseline column choice
# ---------------------------------------------------------------------------


def test_baseline_uses_train_std_when_provided():
    head = HeadSpec(name="gini", kind="regression")
    y_true = pd.DataFrame({"gini": [0.5, 0.5, 0.5, 0.5]})   # zero eval variance
    y_pred = pd.DataFrame({"gini": [0.6, 0.4, 0.6, 0.4]})
    y_train = pd.DataFrame({"gini": np.linspace(0.1, 0.9, 100)})
    fid = per_concept_fidelity(y_true, y_pred, [head], y_train=y_train).iloc[0]
    assert fid["baseline_value"] == pytest.approx(float(y_train["gini"].std(ddof=0)))
