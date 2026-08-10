"""Feature matrix from Stage-1 PREDICTED concepts.

The symbolic head consumes predicted concepts — never ground-truth ones —
so the interpretability chain is end-to-end. For binary classification
heads we drop one probability column to break the sum-to-1 collinearity;
for k-way heads (k ≥ 3) we keep every prob column and let PySR pick.

Feature names are normalised (dashes, spaces → underscores) so PySR /
SymPy can round-trip them as bare identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from galaxycbm.concepts.heads import (
    HeadSpec,
    classification_heads,
    prob_columns,
    regression_heads,
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_]")


def safe_feature_name(raw: str) -> str:
    """Turn an arbitrary predicted-concept column name into a SymPy identifier."""
    name = _SAFE_NAME.sub("_", raw)
    if name and name[0].isdigit():
        name = "f_" + name
    return name


@dataclass(frozen=True)
class FeatureSpec:
    columns: list[str]        # sklearn/SymPy-safe names, aligned with X
    raw_columns: list[str]    # source names in preds.parquet, same order


def build_features(
    preds: pd.DataFrame,
    heads: list[HeadSpec],
) -> tuple[pd.DataFrame, FeatureSpec]:
    raw: list[str] = []
    for h in classification_heads(heads):
        cols = prob_columns(h)
        if h.n_classes == 2 and len(cols) == 2:
            # Drop the reference class; the second column carries all the info.
            candidates = cols[1:]
        else:
            candidates = cols
        raw.extend(c for c in candidates if c in preds.columns)
    for h in regression_heads(heads):
        if h.name in preds.columns:
            raw.append(h.name)
    if not raw:
        raise ValueError(
            "no predicted-concept columns found in preds.parquet. "
            "Did Stage 1 run and emit the expected prob__class columns?"
        )
    safe = [safe_feature_name(c) for c in raw]
    X = preds[raw].copy()
    X.columns = safe
    return X, FeatureSpec(columns=safe, raw_columns=raw)
