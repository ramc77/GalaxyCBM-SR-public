"""Feature-matrix build — pure Python, no PySR / Julia."""

from __future__ import annotations

import pandas as pd

from galaxycbm.concepts.heads import HeadSpec
from galaxycbm.symbolic import build_features, safe_feature_name


def test_safe_feature_name_strips_illegal_chars():
    assert safe_feature_name("smooth-or-featured__smooth") == "smooth_or_featured__smooth"
    assert safe_feature_name("bar-dr5_no_fraction") == "bar_dr5_no_fraction"


def test_safe_feature_name_prefixes_leading_digit():
    assert safe_feature_name("2arms").startswith("f_")


def test_build_features_drops_reference_class_for_binary_heads():
    heads = [HeadSpec(name="edge", kind="classification", n_classes=2, classes=("no", "yes"))]
    preds = pd.DataFrame({
        "edge__no": [0.7, 0.2],
        "edge__yes": [0.3, 0.8],
    })
    X, spec = build_features(preds, heads)
    assert spec.raw_columns == ["edge__yes"]
    assert list(X.columns) == ["edge__yes"]
    assert list(X["edge__yes"]) == [0.3, 0.8]


def test_build_features_keeps_all_probs_for_multiclass_head():
    heads = [HeadSpec(name="bar", kind="classification", n_classes=3,
                     classes=("none", "weak", "strong"))]
    preds = pd.DataFrame({
        "bar__none":   [0.6, 0.2],
        "bar__weak":   [0.3, 0.3],
        "bar__strong": [0.1, 0.5],
    })
    _, spec = build_features(preds, heads)
    assert set(spec.raw_columns) == {"bar__none", "bar__weak", "bar__strong"}


def test_build_features_keeps_regression_columns():
    heads = [HeadSpec(name="gini", kind="regression")]
    preds = pd.DataFrame({"gini": [0.4, 0.5]})
    X, spec = build_features(preds, heads)
    assert list(X.columns) == ["gini"]


def test_build_features_raises_when_nothing_found():
    heads = [HeadSpec(name="edge", kind="classification", n_classes=2, classes=("no", "yes"))]
    preds = pd.DataFrame({"unrelated": [1, 2]})
    import pytest
    with pytest.raises(ValueError):
        build_features(preds, heads)
