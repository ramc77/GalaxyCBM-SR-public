"""Symbolic-head evaluation: score, predict, metrics, callable export.

Uses canned ClassRules (no PySR fit required) so tests are offline and fast.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from galaxycbm.symbolic import (
    ClassRule,
    SymbolicFitResult,
    compute_metrics,
    evaluate_expression,
    export_callable,
    export_latex,
    export_plain,
    predict_labels,
    rules_dataframe,
    score_expressions,
)


def _rule(cls: str, eqn: str, latex: str = "", cx: int = 3, cv: float = 0.9) -> ClassRule:
    return ClassRule(hubble_class=cls, equation_str=eqn, latex=latex or eqn,
                     complexity=cx, pysr_score=1.0, cv_accuracy=cv)


def _rules() -> list[ClassRule]:
    # Two toy rules over feature `f`. score_E dominates for large f; score_S for small.
    return [_rule("E", "f"), _rule("S", "1.0 - f")]


def test_evaluate_expression_broadcasts_over_dataframe():
    import sympy
    expr = sympy.sympify("2*f + 1")
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    np.testing.assert_allclose(evaluate_expression(expr, X), [1.0, 3.0, 5.0])


def test_score_and_predict_argmax():
    rules = _rules()
    X = pd.DataFrame({"f": [0.1, 0.9]})
    scores = score_expressions(rules, X)
    assert set(scores.columns) == {"E", "S"}
    labels = predict_labels(rules, X)
    assert labels.tolist() == ["S", "E"]


def test_compute_metrics_perfect_and_zero_kappa():
    yt = pd.Series(["E", "S", "E", "S"])
    yp_perfect = pd.Series(["E", "S", "E", "S"])
    m = compute_metrics(yt, yp_perfect)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["cohen_kappa"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_compute_metrics_ignores_nan_true_labels():
    yt = pd.Series(["E", None, "S"])
    yp = pd.Series(["E", "E", "S"])
    m = compute_metrics(yt, yp)
    assert m["n"] == 2
    assert m["accuracy"] == pytest.approx(1.0)


def test_rules_dataframe_columns():
    df = rules_dataframe(SymbolicFitResult(rules=_rules(), classes=["E", "S"]))
    assert set(df.columns) == {
        "hubble_class", "expression", "latex", "complexity",
        "cv_accuracy", "pysr_score",
    }
    assert len(df) == 2


def test_exports_write_files(tmp_path: Path):
    result = SymbolicFitResult(rules=_rules(), classes=["E", "S"], feature_columns=["f"])
    p_tex = export_latex(result, tmp_path / "rules.tex")
    p_txt = export_plain(result, tmp_path / "rules.txt")
    p_py  = export_callable(result, tmp_path / "exported_rules.py")
    for p in (p_tex, p_txt, p_py):
        assert p.exists() and p.stat().st_size > 0


def test_exported_callable_module_predicts(tmp_path: Path):
    result = SymbolicFitResult(rules=_rules(), classes=["E", "S"], feature_columns=["f"])
    p_py = export_callable(result, tmp_path / "exported_rules.py")

    spec = importlib.util.spec_from_file_location("_gen_rules", p_py)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    X = pd.DataFrame({"f": [0.1, 0.9]})
    labels = mod.predict(X)
    assert labels.tolist() == ["S", "E"]
    scores = mod.score(X)
    assert set(scores.columns) == {"E", "S"}
