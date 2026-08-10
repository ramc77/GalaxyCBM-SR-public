"""Cross-survey robustness: apply frozen Stage-1→2→3 on Euclid Q1 / JWST-COSMOS
and measure Δ metrics + rule stability.
"""

from galaxycbm.robustness.metrics import (
    RuleStability,
    expected_calibration_error,
    findings_note,
    rule_stability,
    shift_delta_row,
    shift_figure,
)
from galaxycbm.robustness.pipeline import ShiftedRun, run_shifted_pipeline
from galaxycbm.robustness.sources import DEFAULT_SOURCES, SurveySource, download_all_shards

__all__ = [
    "DEFAULT_SOURCES",
    "RuleStability",
    "ShiftedRun",
    "SurveySource",
    "download_all_shards",
    "expected_calibration_error",
    "findings_note",
    "rule_stability",
    "run_shifted_pipeline",
    "shift_delta_row",
    "shift_figure",
]
