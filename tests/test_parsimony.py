"""Compact-rule ablation: front matching, selection rule, and evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from galaxycbm.symbolic import ClassRule, evaluate_rule_set, load_fronts, select_compact_rules


def _adopted() -> list[ClassRule]:
    return [
        ClassRule(hubble_class="A", equation_str="f0 + f1", latex="", complexity=20,
                  pysr_score=0.0, cv_accuracy=0.900),
        ClassRule(hubble_class="B", equation_str="f1", latex="", complexity=15,
                  pysr_score=0.0, cv_accuracy=0.800),
    ]


def _front(complexities, accs) -> pd.DataFrame:
    return pd.DataFrame({
        "complexity": complexities,
        "cv_accuracy": accs,
        "equation": [f"f0 * {c}" for c in complexities],
        "score": [0.0] * len(complexities),
    })


def test_select_picks_smallest_within_tolerance():
    fronts = {
        # 0.898 is within tau=0.005 of the 0.900 best and much smaller.
        "A": _front([5, 9, 20], [0.870, 0.898, 0.900]),
        "B": _front([4, 15], [0.700, 0.800]),
    }
    compact, table = select_compact_rules(_adopted(), fronts, tau=0.005)
    by_class = {r.hubble_class: r for r in compact}
    assert by_class["A"].complexity == 9        # 5 is outside tolerance
    assert by_class["B"].complexity == 15       # nothing cheaper qualifies
    assert set(table["class"]) == {"A", "B"}
    assert table["front_available"].all()


def test_missing_front_keeps_adopted_rule_and_is_flagged():
    compact, table = select_compact_rules(_adopted(), fronts={}, tau=0.005)
    assert [r.complexity for r in compact] == [20, 15]
    assert not table["front_available"].any()


def test_load_fronts_rejects_a_front_that_does_not_contain_the_adopted_rule(tmp_path):
    """A stale front from another run must not be paired with current rules."""
    stale = _front([5, 9, 20], [0.100, 0.200, 0.300])   # no (20, 0.900) pair
    stale.to_parquet(tmp_path / "A__deadbeef__pareto.parquet", index=False)
    assert load_fronts(tmp_path, _adopted()) == {}

    fresh = _front([5, 9, 20], [0.870, 0.898, 0.900])
    fresh.to_parquet(tmp_path / "A__cafe1234__pareto.parquet", index=False)
    fronts = load_fronts(tmp_path, _adopted())
    assert set(fronts) == {"A"}


def test_evaluate_rule_set_reports_total_nodes():
    X = pd.DataFrame({"f0": [1.0, 0.0, 1.0, 0.0], "f1": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series(["A", "B", "A", "B"])
    rules = [
        ClassRule(hubble_class="A", equation_str="f0", latex="", complexity=3,
                  pysr_score=0.0, cv_accuracy=0.0),
        ClassRule(hubble_class="B", equation_str="f1", latex="", complexity=4,
                  pysr_score=0.0, cv_accuracy=0.0),
    ]
    out = evaluate_rule_set(rules, X, y)
    assert out["total_nodes"] == 7
    assert out["accuracy"] == pytest.approx(1.0)
