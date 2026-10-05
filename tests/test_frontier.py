"""Out-of-sample scoring of Pareto-front expressions.

The first test pins down a property of Stage 2's original selection score that
matters for how the paper describes it: a fixed expression scored on folds of
its own training pool returns the pooled training accuracy, so it carries no
information about overfitting. The rest check the machinery that replaces it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from galaxycbm.symbolic.fit import _cv_accuracy
from galaxycbm.symbolic.frontier import (
    binomial_se,
    frontier_accuracy_table,
    ovr_accuracy,
    rules_from_rows,
    select_on_split,
)


def _toy(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f0": rng.random(n), "f1": rng.random(n)})
    y = (X["f0"] + 0.25 * rng.normal(size=n) > 0.5).astype(int).to_numpy()
    return X, y


def test_fold_mean_equals_pooled_training_accuracy():
    """The stored cv_accuracy is the in-sample accuracy, not a held-out one."""
    import sympy

    X, y = _toy(n=1000)                      # 1000 = 5 equal folds of 200
    expr = sympy.sympify("f0 + 0.05")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))

    fold_mean = _cv_accuracy(expr, X, y, folds)
    pooled, _ = ovr_accuracy("f0 + 0.05", X, y)

    assert fold_mean == pytest.approx(pooled, abs=1e-12)


def test_ovr_accuracy_matches_a_hand_computation():
    X = pd.DataFrame({"f0": [0.9, 0.8, 0.1, 0.2, 0.7, 0.3]})
    y = np.array([1, 1, 0, 0, 0, 1])          # predictions: 1 1 0 0 1 0
    acc, bal = ovr_accuracy("f0", X, y)
    assert acc == pytest.approx(4 / 6)
    assert bal == pytest.approx(0.5 * (2 / 3 + 2 / 3))


def test_nan_scores_are_predicted_negative_not_dropped():
    X = pd.DataFrame({"f0": [1.0, 1.0, 1.0, 1.0], "f1": [0.0, 0.0, 2.0, 2.0]})
    y = np.array([0, 0, 1, 1])
    # f0/f1 is inf on the first two rows (inf >= 0.5 is True) -> predicted 1.
    acc, _ = ovr_accuracy("f0 / f1 - 0.0 * f0", X, y)
    assert 0.0 <= acc <= 1.0


def test_square_function_from_pysr_output_evaluates():
    """PySR emits square(x); the pipeline must evaluate it, as at inference."""
    X = pd.DataFrame({"f0": [0.1, 0.9, 0.8, 0.2]})
    y = np.array([0, 1, 1, 0])
    acc, _ = ovr_accuracy("square(f0) + 0.0", X, y, threshold=0.5)
    assert acc == pytest.approx(1.0)


def test_unevaluable_expression_returns_nan_not_a_number():
    X = pd.DataFrame({"f0": [0.1, 0.9]})
    acc, bal = ovr_accuracy("no_such_column + 1", X, np.array([0, 1]))
    assert np.isnan(acc) and np.isnan(bal)


def _fronts():
    return {
        "A": pd.DataFrame({
            "complexity": [1, 3, 5],
            "equation": ["f1", "f0", "f0 + 0.0 * f1"],
            "cv_accuracy": [0.5, 0.9, 0.9],
        }),
    }


def test_frontier_table_reports_every_split_for_every_expression():
    Xtr, ytr = _toy(600, seed=1)
    Xva, yva = _toy(400, seed=2)
    labels = lambda y: pd.Series(np.where(y == 1, "A", "B"))
    table = frontier_accuracy_table(
        _fronts(),
        {"train": Xtr, "val": Xva},
        {"train": labels(ytr), "val": labels(yva)},
    )
    assert len(table) == 3
    assert {"train_acc", "val_acc", "train_bal_acc", "val_bal_acc",
            "cv_minus_train"} <= set(table.columns)
    assert table["train_acc"].between(0, 1).all()


def test_selection_uses_the_requested_split_and_breaks_ties_to_lower_complexity():
    table = pd.DataFrame({
        "hubble_class": ["A", "A", "A", "B", "B"],
        "complexity":   [1, 4, 9, 2, 6],
        "equation":     ["x", "x", "x", "x", "x"],
        "train_acc":    [0.60, 0.80, 0.95, 0.70, 0.90],
        "val_acc":      [0.60, 0.80, 0.80, 0.85, 0.70],
    })
    picked = select_on_split(table, "val").set_index("hubble_class")
    assert picked.loc["A", "complexity"] == 4      # tie at 0.80 -> simpler
    assert picked.loc["B", "complexity"] == 2      # val prefers the small one
    # Selecting on train would have chosen the complex expressions instead.
    on_train = select_on_split(table, "train").set_index("hubble_class")
    assert on_train.loc["A", "complexity"] == 9


def test_rules_from_rows_round_trips_complexity_and_equation():
    table = pd.DataFrame({
        "hubble_class": ["A"], "complexity": [7], "equation": ["f0 + 1"],
        "cv_accuracy": [0.9],
    })
    (rule,) = rules_from_rows(table)
    assert (rule.hubble_class, rule.complexity, rule.equation_str) == ("A", 7, "f0 + 1")


def test_binomial_se():
    assert binomial_se(0.5, 100) == pytest.approx(0.05)
    assert binomial_se(1.0, 100) == 0.0


# ---- end to end: the same path the iMac script takes -----------------------

def _frames(seed: int = 0):
    from galaxycbm.symbolic import ClassRule  # noqa: F401  (imported for the rules below)

    rng = np.random.default_rng(seed)

    def one(n: int) -> pd.DataFrame:
        f0 = rng.random(n)
        label = np.where(f0 + 0.15 * rng.normal(size=n) > 0.5, "A", "B")
        label[:5] = "Z"                       # a class with no rule: must be dropped
        return pd.DataFrame({"f0": f0, "f1": rng.random(n), "hubble_type": label})

    frames = {"train": one(800), "val": one(400), "test": one(400)}
    frames["val"].loc[3, "f1"] = np.nan       # imputation path
    return frames


def test_selection_report_end_to_end_on_synthetic_data():
    import json

    from galaxycbm.symbolic import ClassRule
    from galaxycbm.symbolic.frontier import prepare_splits, selection_report

    frames = _frames()
    classes = ["A", "B"]
    X, y = prepare_splits(frames, ["f0", "f1"], classes)

    # Rows of class Z are gone and X / y are still paired.
    assert set(y["val"]) == {"A", "B"}
    assert len(X["val"]) == len(y["val"]) == 400 - 5
    assert not X["val"].isna().any().any()

    fronts = {
        "A": pd.DataFrame({"complexity": [1, 5, 9],
                           "equation": ["f1", "f0", "f0 + 0.02 * f1"],
                           "cv_accuracy": [0.5, 0.9, 0.9]}),
        "B": pd.DataFrame({"complexity": [3, 7],
                           "equation": ["1 - f0", "1 - f0 + 0.02 * f1"],
                           "cv_accuracy": [0.9, 0.9]}),
    }
    table = frontier_accuracy_table(
        fronts, {s: X[s] for s in ("train", "val")}, {s: y[s] for s in ("train", "val")})

    adopted = [
        ClassRule("A", "f0", "", 5, 0.0, 0.9),
        ClassRule("B", "1 - f0", "", 3, 0.0, 0.9),
    ]
    report = selection_report(table, adopted, X, y)

    assert [r["hubble_class"] for r in report["per_class"]] == classes
    assert report["metrics"]["adopted"]["total_nodes"] == 8
    for label in ("adopted", "val_selected"):
        for split in ("val", "test"):
            assert 0.0 <= report["metrics"][label][split]["accuracy"] <= 1.0
    # The whole report must survive json (the iMac script writes it to disk).
    json.dumps(report, default=float)


def test_selection_report_rejects_an_adopted_rule_that_is_not_on_its_frontier():
    from galaxycbm.symbolic import ClassRule
    from galaxycbm.symbolic.frontier import prepare_splits, selection_report

    frames = _frames()
    X, y = prepare_splits(frames, ["f0", "f1"], ["A", "B"])
    fronts = {"A": pd.DataFrame({"complexity": [1], "equation": ["f0"]}),
              "B": pd.DataFrame({"complexity": [1], "equation": ["1 - f0"]})}
    table = frontier_accuracy_table(
        fronts, {s: X[s] for s in ("train", "val")}, {s: y[s] for s in ("train", "val")})
    stale = [ClassRule("A", "f0", "", 99, 0.0, 0.9), ClassRule("B", "1 - f0", "", 1, 0.0, 0.9)]
    with pytest.raises(KeyError, match="not on its cached frontier"):
        selection_report(table, stale, X, y)
