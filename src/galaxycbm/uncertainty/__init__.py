"""Stage-3 split-conformal wrapper (MAPIE 1.4)."""

from galaxycbm.uncertainty.conformal import (
    ConformalHead,
    MondrianHead,
    conformalize,
    coverage_and_set_size,
    min_calibration_points,
    mondrian_conformalize,
    mondrian_predict_sets,
    mondrian_report,
    per_class_coverage,
    predict_sets,
    selective_curve,
    selective_figure,
)
from galaxycbm.uncertainty.estimator import SymbolicRuleClassifier

__all__ = [
    "ConformalHead",
    "MondrianHead",
    "SymbolicRuleClassifier",
    "conformalize",
    "coverage_and_set_size",
    "min_calibration_points",
    "mondrian_conformalize",
    "mondrian_predict_sets",
    "mondrian_report",
    "per_class_coverage",
    "predict_sets",
    "selective_curve",
    "selective_figure",
]
