"""Head-spec builder — pure Python, no torch."""

from __future__ import annotations

from galaxycbm.concepts import (
    HeadSpec,
    build_head_specs,
    classification_heads,
    prob_columns,
    regression_heads,
)
from galaxycbm.utils import load_config


def test_build_head_specs_from_real_config():
    cfg = load_config("concepts")
    heads = build_head_specs(cfg)
    assert heads, "expected at least one head"
    kinds = {h.kind for h in heads}
    assert kinds == {"classification", "regression"}


def test_classification_heads_have_classes_and_n_matches():
    cfg = load_config("concepts")
    for h in classification_heads(build_head_specs(cfg)):
        assert h.classes is not None
        assert h.n_classes == len(h.classes)


def test_regression_heads_have_no_classes():
    cfg = load_config("concepts")
    for h in regression_heads(build_head_specs(cfg)):
        assert h.classes is None
        assert h.n_classes == 1


def test_prob_columns_shape():
    h = HeadSpec(name="bar", kind="classification", n_classes=3,
                 classes=("none", "weak", "strong"))
    assert prob_columns(h) == ["bar__none", "bar__weak", "bar__strong"]


def test_prob_columns_empty_for_regression():
    h = HeadSpec(name="gini", kind="regression")
    assert prob_columns(h) == []
