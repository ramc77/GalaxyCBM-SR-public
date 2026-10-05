"""Score every Pareto-front expression on data the symbolic search never saw.

Why this module exists
----------------------
Stage 2 ranks the PySR frontier by ``cv_accuracy`` (see ``fit._cv_accuracy``).
That score evaluates a *fixed* expression, found by a search that had already
seen the whole training pool, on stratified folds *of that same pool*, at a
fixed threshold of 0.5. Nothing is refitted inside a fold, so the mean of the
fold accuracies is the pooled accuracy on the training pool (equal to it
exactly when the folds are the same size). The score is therefore an in-sample
accuracy and cannot detect overfitting by the search.

This module gives the missing comparison: the same one-vs-rest accuracy,
computed with the same threshold, on the training pool and on splits that were
held out from the search. It does not refit anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from galaxycbm.symbolic.fit import ClassRule, evaluate_expression

THRESHOLD = 0.5   # the fixed one-vs-rest threshold used by Stage 2


def ovr_accuracy(
    equation: str,
    X: pd.DataFrame,
    y_bin: np.ndarray,
    *,
    threshold: float = THRESHOLD,
) -> tuple[float, float]:
    """One-vs-rest (accuracy, balanced accuracy) of one fixed expression.

    A NaN score compares False against the threshold and is therefore
    predicted negative, exactly as in ``fit._cv_accuracy``. An expression that
    cannot be evaluated returns ``(nan, nan)`` rather than a made-up number.
    """
    import sympy

    try:
        expr = sympy.sympify(equation)
        score = evaluate_expression(expr, X.reset_index(drop=True))
    except Exception:
        return float("nan"), float("nan")

    y = np.asarray(y_bin).astype(int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    acc = float(np.mean(pred == y))

    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return acc, float("nan")
    tpr = float(np.mean(pred[pos] == 1))
    tnr = float(np.mean(pred[neg] == 0))
    return acc, 0.5 * (tpr + tnr)


def frontier_accuracy_table(
    fronts: dict[str, pd.DataFrame],
    X_by_split: dict[str, pd.DataFrame],
    y_by_split: dict[str, pd.Series],
    *,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Per-expression accuracies for every split, one row per frontier point.

    ``fronts`` maps a Hubble class to its cached Pareto front (needs
    ``complexity`` and ``equation``). ``X_by_split`` / ``y_by_split`` map a
    split name (``train``, ``val``, ...) to its features and Hubble labels.
    The stored ``cv_accuracy`` is carried through unchanged so it can be set
    beside the recomputed training accuracy.
    """
    rows: list[dict[str, object]] = []
    for cls, front in fronts.items():
        y_bin = {name: (y.astype(str).to_numpy() == cls).astype(int)
                 for name, y in y_by_split.items()}
        for _, r in front.sort_values("complexity").iterrows():
            row: dict[str, object] = {
                "hubble_class": cls,
                "complexity": int(r["complexity"]),
                "equation": str(r["equation"]),
            }
            for passthrough in ("latex", "score"):
                if passthrough in front.columns:
                    row[passthrough] = r[passthrough]
            if "cv_accuracy" in front.columns:
                row["cv_accuracy"] = float(r["cv_accuracy"])
            for name, X in X_by_split.items():
                acc, bal = ovr_accuracy(row["equation"], X, y_bin[name],
                                        threshold=threshold)
                row[f"n_{name}"] = int(len(X))
                row[f"{name}_acc"] = acc
                row[f"{name}_bal_acc"] = bal
            if "cv_accuracy" in row and "train_acc" in row:
                row["cv_minus_train"] = row["cv_accuracy"] - row["train_acc"]
            rows.append(row)
    return pd.DataFrame(rows)


def select_on_split(
    table: pd.DataFrame,
    split: str = "val",
    *,
    tie_atol: float = 1e-12,
) -> pd.DataFrame:
    """Pick one expression per class by accuracy on a split the search never saw.

    The highest accuracy wins; ties go to the lower complexity, the same
    tie-break Stage 2 uses. Returns one row per class.
    """
    col = f"{split}_acc"
    picked = []
    for cls, g in table.groupby("hubble_class", sort=True):
        g = g.dropna(subset=[col])
        best = g[col].max()
        tied = g[g[col] >= best - tie_atol]
        picked.append(tied.sort_values("complexity").iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def binomial_se(p: float, n: int) -> float:
    """Sampling standard error of an accuracy ``p`` measured on ``n`` objects.

    Differences in validation accuracy smaller than this are not evidence that
    one expression generalises better than another.
    """
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1)))


def rules_from_rows(rows: pd.DataFrame) -> list[ClassRule]:
    """Turn selected table rows into ``ClassRule`` objects for evaluation."""
    out: list[ClassRule] = []
    for _, r in rows.iterrows():
        out.append(ClassRule(
            hubble_class=str(r["hubble_class"]),
            equation_str=str(r["equation"]),
            latex=str(r["latex"]) if "latex" in r.index and pd.notna(r["latex"]) else "",
            complexity=int(r["complexity"]),
            pysr_score=float(r["score"]) if "score" in r.index and pd.notna(r["score"]) else 0.0,
            cv_accuracy=float(r["cv_accuracy"]) if "cv_accuracy" in r.index else float("nan"),
        ))
    return out


def prepare_splits(
    frames: dict[str, pd.DataFrame],
    feature_cols: list[str],
    classes: list[str],
    *,
    names: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Features and labels per split, imputed with TRAIN medians as in Stage 2.

    Rows whose true class has no rule are dropped, as everywhere else in the
    pipeline. Index alignment is reset so features and labels stay paired.
    """
    medians = frames["train"][feature_cols].median(numeric_only=True).fillna(0.0)
    X: dict[str, pd.DataFrame] = {}
    y: dict[str, pd.Series] = {}
    for name in names:
        feats = frames[name][feature_cols].fillna(medians).fillna(0.0).reset_index(drop=True)
        labels = frames[name]["hubble_type"].astype(str).reset_index(drop=True)
        keep = labels.isin(classes).to_numpy()
        X[name] = feats[keep].reset_index(drop=True)
        y[name] = labels[keep].reset_index(drop=True)
    return X, y


def selection_report(
    table: pd.DataFrame,
    adopted: list[ClassRule],
    X: dict[str, pd.DataFrame],
    y: dict[str, pd.Series],
) -> dict:
    """Adopted rule set versus the set chosen on validation accuracy.

    The validation-selected set is chosen on ``val``, so its ``val`` figures are
    optimistic; the fair comparison between the two sets is on ``test``.
    """
    from galaxycbm.symbolic.parsimony import evaluate_rule_set

    classes = [r.hubble_class for r in adopted]
    rows = []
    for rule in adopted:
        hit = table[(table["hubble_class"] == rule.hubble_class)
                    & (table["complexity"] == rule.complexity)]
        if hit.empty:
            raise KeyError(
                f"adopted {rule.hubble_class} rule (complexity {rule.complexity}) "
                "is not on its cached frontier")
        rows.append(hit.iloc[0])
    adopted_tbl = pd.DataFrame(rows).reset_index(drop=True)
    selected_tbl = select_on_split(table, "val")

    n_val = len(y["val"])
    per_class = []
    for cls in classes:
        a = adopted_tbl[adopted_tbl["hubble_class"] == cls].iloc[0]
        s = selected_tbl[selected_tbl["hubble_class"] == cls].iloc[0]
        per_class.append({
            "hubble_class": cls,
            "adopted_complexity": int(a["complexity"]),
            "adopted_train_acc": float(a["train_acc"]),
            "adopted_val_acc": float(a["val_acc"]),
            "selected_complexity": int(s["complexity"]),
            "selected_train_acc": float(s["train_acc"]),
            "selected_val_acc": float(s["val_acc"]),
            "val_acc_se": binomial_se(float(s["val_acc"]), n_val),
            "same_expression": bool(a["equation"] == s["equation"]),
        })

    sets = {"adopted": rules_from_rows(adopted_tbl),
            "val_selected": rules_from_rows(selected_tbl)}
    metrics: dict[str, dict] = {}
    for label, rules in sets.items():
        metrics[label] = {split: evaluate_rule_set(rules, X[split], y[split])
                          for split in ("val", "test")}
        metrics[label]["total_nodes"] = int(sum(r.complexity for r in rules))

    gap = table["cv_minus_train"].abs() if "cv_minus_train" in table else pd.Series([np.nan])
    return {
        "threshold": THRESHOLD,
        "n_train": len(y["train"]), "n_val": n_val, "n_test": len(y["test"]),
        "max_abs_cv_minus_train": float(gap.max()),
        "per_class": per_class,
        "metrics": metrics,
        "note": ("val_selected is chosen on the validation split, so its "
                 "validation figures are optimistic; compare the two sets on test."),
    }
