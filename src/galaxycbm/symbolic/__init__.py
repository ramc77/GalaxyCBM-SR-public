"""Stage-2 symbolic decision head (PySR).

PySR fitting lives behind lazy imports (Julia is heavy). Everything a
downstream consumer needs to *apply* the rules — score, predict, metrics,
Pareto plot — works in the base env without touching Julia.
"""

from galaxycbm.symbolic.eval import (
    compute_metrics,
    pareto_figure,
    predict_labels,
    score_expressions,
)
from galaxycbm.symbolic.features import (
    FeatureSpec,
    build_features,
    safe_feature_name,
)
from galaxycbm.symbolic.fit import (
    ClassRule,
    SymbolicFitResult,
    evaluate_expression,
    export_callable,
    export_latex,
    export_plain,
    fit_symbolic,
    rules_dataframe,
)

__all__ = [
    "ClassRule",
    "FeatureSpec",
    "SymbolicFitResult",
    "build_features",
    "compute_metrics",
    "evaluate_expression",
    "export_callable",
    "export_latex",
    "export_plain",
    "fit_symbolic",
    "pareto_figure",
    "predict_labels",
    "rules_dataframe",
    "safe_feature_name",
    "score_expressions",
]
