"""Evaluate the symbolic head: metrics + Pareto figure.

Pure numpy / sklearn / matplotlib / sympy. No PySR / Julia needed to
apply the rules or score predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.symbolic.fit import ClassRule, SymbolicFitResult, evaluate_expression


def score_expressions(rules: list[ClassRule], X: pd.DataFrame) -> pd.DataFrame:
    import sympy

    out: dict[str, np.ndarray] = {}
    for r in rules:
        expr = sympy.sympify(r.equation_str)
        out[r.hubble_class] = evaluate_expression(expr, X)
    return pd.DataFrame(out, index=X.index)


def predict_labels(rules: list[ClassRule], X: pd.DataFrame) -> pd.Series:
    scores = score_expressions(rules, X)
    return pd.Series(scores.idxmax(axis=1).to_numpy(), index=X.index, name="hubble_pred")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )

    mask = y_true.notna()
    yt = y_true[mask].astype(str)
    yp = y_pred[mask].astype(str)
    labels = sorted(set(yt) | set(yp))
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "n": int(mask.sum()),
        "labels": labels,
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
    }


def pareto_figure(result: SymbolicFitResult, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cls, pf in result.per_class_pareto.items():
        ax.scatter(pf["complexity"], pf["cv_accuracy"], alpha=0.35, s=20, label=cls)
    for r in result.rules:
        ax.scatter([r.complexity], [r.cv_accuracy], marker="*",
                   s=180, edgecolor="black", linewidth=0.8, zorder=5)
    ax.set_xlabel("expression complexity (nodes)")
    ax.set_ylabel("stratified k-fold CV accuracy")
    ax.set_title("Pareto: accuracy vs complexity per Hubble class")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
