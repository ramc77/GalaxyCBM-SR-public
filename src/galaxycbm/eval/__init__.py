"""Consolidated evaluation, publication tables, figures, and consistency check."""

from galaxycbm.eval.aggregate import (
    aggregate_metrics,
    path_exists_and_is_populated,
    resolve_path,
)
from galaxycbm.eval.consistency import (
    ClaimReport,
    check_claims,
    load_claims_manifest,
)
from galaxycbm.eval.figures import build_all as build_all_figures
from galaxycbm.eval.style import OKABE_ITO, apply_paper_style, palette
from galaxycbm.eval.tables import dataframe_to_latex, write_table

__all__ = [
    "ClaimReport",
    "OKABE_ITO",
    "aggregate_metrics",
    "apply_paper_style",
    "build_all_figures",
    "check_claims",
    "dataframe_to_latex",
    "load_claims_manifest",
    "palette",
    "path_exists_and_is_populated",
    "resolve_path",
    "write_table",
]
