"""Concept-head specs.

Pure Python; no torch. Turns the two vocabularies in configs/concepts.yaml
into a flat list of heads for the CBM to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["classification", "regression"]


@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: Kind
    n_classes: int = 1
    classes: tuple[str, ...] | None = None  # only set for classification


def build_head_specs(cfg_concepts) -> list[HeadSpec]:
    """Perceptual GZ tasks → classification heads; statmorph → regression."""
    heads: list[HeadSpec] = []
    for c in cfg_concepts.perceptual:
        vals = tuple(str(v) for v in list(c["values"]))
        heads.append(
            HeadSpec(name=str(c.name), kind="classification", n_classes=len(vals), classes=vals)
        )
    for c in cfg_concepts.physical:
        heads.append(HeadSpec(name=str(c.name), kind="regression"))
    return heads


def classification_heads(heads: list[HeadSpec]) -> list[HeadSpec]:
    return [h for h in heads if h.kind == "classification"]


def regression_heads(heads: list[HeadSpec]) -> list[HeadSpec]:
    return [h for h in heads if h.kind == "regression"]


def prob_columns(head: HeadSpec) -> list[str]:
    """Column names used in preds.parquet for a classification head's probs."""
    if head.kind != "classification" or not head.classes:
        return []
    return [f"{head.name}__{c}" for c in head.classes]
