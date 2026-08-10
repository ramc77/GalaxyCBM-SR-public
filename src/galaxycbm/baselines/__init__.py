"""Stage-7 baselines and ablations.

Runnable in the base env:
    linear_cbm.py     — sklearn LogisticRegression on predicted concepts
    compare.py        — intrinsic weights, fidelity metric, master table

Guarded (need extras — imported lazily inside functions):
    xgb_concepts.py   — requires the `baselines` extra (xgboost, shap)
    cnn.py            — requires the `stage1` extra (torch, lightning, zoobot)
"""

from galaxycbm.baselines.compare import (
    comparison_row,
    comparison_table,
    concept_fidelity,
    skipped_row,
    symbolic_concept_weights,
    symbolic_interpretability_cost,
)
from galaxycbm.baselines.linear_cbm import (
    concept_importance as linear_concept_importance,
)
from galaxycbm.baselines.linear_cbm import (
    interpretability_cost as linear_interpretability_cost,
)
from galaxycbm.baselines.linear_cbm import (
    train_linear_cbm,
)

__all__ = [
    "comparison_row",
    "comparison_table",
    "concept_fidelity",
    "linear_concept_importance",
    "linear_interpretability_cost",
    "skipped_row",
    "symbolic_concept_weights",
    "symbolic_interpretability_cost",
    "train_linear_cbm",
]
