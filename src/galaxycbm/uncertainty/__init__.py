"""Stage-3 split-conformal wrapper (MAPIE 1.4)."""

from galaxycbm.uncertainty.conformal import (
    ConformalHead,
    conformalize,
    coverage_and_set_size,
    per_class_coverage,
    predict_sets,
    selective_curve,
    selective_figure,
)
from galaxycbm.uncertainty.estimator import SymbolicRuleClassifier

__all__ = [
    "ConformalHead",
    "SymbolicRuleClassifier",
    "conformalize",
    "coverage_and_set_size",
    "per_class_coverage",
    "predict_sets",
    "selective_curve",
    "selective_figure",
]
