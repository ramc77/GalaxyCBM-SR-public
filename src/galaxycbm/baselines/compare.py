"""Master comparison table + intrinsic-vs-post-hoc concept-fidelity.

Interpretability cost is model-specific:
    symbolic         → sum of PySR expression complexities across classes
    linear CBM       → non-zero coefficient count (dense → n_features × n_classes)
    XGBoost          → non-zero feature-importance count
    end-to-end CNN   → trainable parameter count (very large; the point)

Fidelity metric: Spearman + Pearson correlation between intrinsic per-concept
weights and each baseline's per-concept importance. High correlation ⇒ the
post-hoc explanation agrees with the intrinsic model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from galaxycbm.symbolic import ClassRule, compute_metrics


# ---------------------------------------------------------------------------
# Per-model importance / cost
# ---------------------------------------------------------------------------


def symbolic_concept_weights(rules: Iterable[ClassRule]) -> pd.Series:
    """Sum, per feature, of CV_accuracy over every rule that mentions it."""
    import sympy

    w: dict[str, float] = defaultdict(float)
    for r in rules:
        try:
            expr = sympy.sympify(r.equation_str)
        except Exception:
            continue
        used = {str(s.name) for s in expr.free_symbols}
        weight = float(r.cv_accuracy) if pd.notna(r.cv_accuracy) else 0.0
        for u in used:
            w[u] += weight
    return pd.Series(w, name="symbolic_weight").sort_index()


def symbolic_interpretability_cost(rules: Iterable[ClassRule]) -> int:
    return int(sum(int(r.complexity) for r in rules))


# ---------------------------------------------------------------------------
# Fidelity: symbolic vs another model's per-concept importance
# ---------------------------------------------------------------------------


def concept_fidelity(intrinsic: pd.Series, posthoc: pd.Series) -> dict:
    """Spearman + Pearson correlation across shared feature indices."""
    from scipy.stats import pearsonr, spearmanr

    idx = intrinsic.index.intersection(posthoc.index)
    if len(idx) < 3:
        return {"n_shared": int(len(idx)),
                "spearman": float("nan"), "pearson": float("nan")}
    a = intrinsic.reindex(idx).to_numpy(float)
    b = posthoc.reindex(idx).to_numpy(float)
    # Constant series → correlation undefined.
    if np.std(a) == 0 or np.std(b) == 0:
        return {"n_shared": int(len(idx)),
                "spearman": float("nan"), "pearson": float("nan")}
    return {
        "n_shared": int(len(idx)),
        "spearman": float(spearmanr(a, b).correlation),
        "pearson": float(pearsonr(a, b).statistic),
    }


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def comparison_row(
    model: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    interpretability_cost: int | float,
    interpretability_kind: str,
    note: str = "",
) -> dict:
    m = compute_metrics(y_true, y_pred)
    return {
        "model": model,
        "n": m["n"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "cohen_kappa": m["cohen_kappa"],
        "interpretability_cost": interpretability_cost,
        "interpretability_kind": interpretability_kind,
        "note": note,
    }


def skipped_row(model: str, reason: str) -> dict:
    return {
        "model": model, "n": 0,
        "accuracy": float("nan"),
        "macro_f1": float("nan"),
        "cohen_kappa": float("nan"),
        "interpretability_cost": float("nan"),
        "interpretability_kind": "n/a",
        "note": f"skipped: {reason}",
    }


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "model", "n", "accuracy", "macro_f1", "cohen_kappa",
        "interpretability_cost", "interpretability_kind", "note",
    ])
