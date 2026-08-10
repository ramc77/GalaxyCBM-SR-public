"""Stage-1 concept predictor: Zoobot backbone + multi-head bottleneck.

Heavy deps (torch, lightning, zoobot) live under the `stage1` extra and are
imported lazily. This subpackage stays importable in the base env.
"""

from galaxycbm.concepts.fidelity import per_concept_fidelity, reliability_figure
from galaxycbm.concepts.heads import (
    HeadSpec,
    build_head_specs,
    classification_heads,
    prob_columns,
    regression_heads,
)

__all__ = [
    "HeadSpec",
    "build_head_specs",
    "classification_heads",
    "per_concept_fidelity",
    "prob_columns",
    "regression_heads",
    "reliability_figure",
]
