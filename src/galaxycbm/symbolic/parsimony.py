"""Compact-rule ablation: is the adopted expression size actually necessary?

The Stage-2 selector maximises cross-validated accuracy over the PySR Pareto
front, which lands on 17--25 node expressions. Because every front flattens
below roughly ten nodes, it is fair to ask whether that size is
buying anything or is avoidable complexity (equivalently, mild overfitting of
the selection step).

This module answers the question directly. From each cached front we take the
SMALLEST expression whose cross-validated accuracy is within a tolerance
`tau` of the class-best, assemble those into an alternative rule set, and
evaluate it end to end on a held-out split alongside the adopted set. If the
compact set matches, the adopted size is unnecessary and should be reported
as such; if it does not, the size is justified by held-out evidence rather
than by the selector's preference.

Reads the Pareto frontiers cached by `fit_symbolic` under
``results/symbolic/fit_cache/<class>__<hash>__pareto.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from galaxycbm.symbolic.fit import ClassRule


def load_fronts(
    fit_cache: str | Path,
    adopted: list[ClassRule],
    *,
    atol: float = 1e-9,
) -> dict[str, pd.DataFrame]:
    """Load the Pareto front belonging to each adopted rule.

    A front is accepted for class k only if it actually contains the adopted
    rule's (complexity, cv_accuracy) pair. Stale fronts from an earlier run on
    a different sample are therefore rejected rather than silently paired with
    current rules.
    """
    cache = Path(fit_cache)
    fronts: dict[str, pd.DataFrame] = {}
    for rule in adopted:
        safe = rule.hubble_class.replace("/", "_")
        for path in sorted(cache.glob(f"{safe}__*__pareto.parquet")):
            front = pd.read_parquet(path)
            if not {"complexity", "cv_accuracy", "equation"} <= set(front.columns):
                continue
            hit = (
                (front["complexity"].astype(int) == int(rule.complexity))
                & ((front["cv_accuracy"] - rule.cv_accuracy).abs() < 1e-6)
            )
            if bool(hit.any()):
                fronts[rule.hubble_class] = front.sort_values("complexity").reset_index(drop=True)
                break
    return fronts


def select_compact_rules(
    adopted: list[ClassRule],
    fronts: dict[str, pd.DataFrame],
    *,
    tau: float = 0.005,
) -> tuple[list[ClassRule], pd.DataFrame]:
    """Smallest expression per class within `tau` CV accuracy of the best.

    Returns the compact rule set and a per-class comparison table. Classes
    with no usable front keep their adopted rule and are flagged.
    """
    compact: list[ClassRule] = []
    rows: list[dict[str, object]] = []

    for rule in adopted:
        front = fronts.get(rule.hubble_class)
        if front is None or front.empty:
            compact.append(rule)
            rows.append({
                "class": rule.hubble_class,
                "adopted_complexity": int(rule.complexity),
                "adopted_cv_accuracy": float(rule.cv_accuracy),
                "compact_complexity": int(rule.complexity),
                "compact_cv_accuracy": float(rule.cv_accuracy),
                "compact_equation": rule.equation_str,
                "front_available": False,
            })
            continue

        best = float(front["cv_accuracy"].max())
        eligible = front[front["cv_accuracy"] >= best - tau]
        pick = eligible.sort_values(["complexity", "cv_accuracy"],
                                    ascending=[True, False]).iloc[0]
        compact.append(ClassRule(
            hubble_class=rule.hubble_class,
            equation_str=str(pick["equation"]),
            latex=str(pick["latex"]) if "latex" in pick.index else "",
            complexity=int(pick["complexity"]),
            pysr_score=float(pick["score"]) if "score" in pick.index else 0.0,
            cv_accuracy=float(pick["cv_accuracy"]),
        ))
        rows.append({
            "class": rule.hubble_class,
            "adopted_complexity": int(rule.complexity),
            "adopted_cv_accuracy": float(rule.cv_accuracy),
            "compact_complexity": int(pick["complexity"]),
            "compact_cv_accuracy": float(pick["cv_accuracy"]),
            "compact_equation": str(pick["equation"]),
            "front_available": True,
        })

    return compact, pd.DataFrame(rows)


def evaluate_rule_set(
    rules: list[ClassRule],
    X: pd.DataFrame,
    y_true: pd.Series,
) -> dict:
    """Accuracy / macro-F1 / kappa plus the total node count of a rule set."""
    from galaxycbm.symbolic.eval import compute_metrics, predict_labels

    y_pred = predict_labels(rules, X)
    out = compute_metrics(y_true, y_pred)
    out["total_nodes"] = int(sum(int(r.complexity) for r in rules))
    return out
