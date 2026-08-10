"""(b) Dense-linear CBM head.

Mirror of Stage 2 but with sklearn's multinomial logistic regression instead
of PySR. Same input features (predicted concepts), same train/val split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from galaxycbm.symbolic import compute_metrics


def train_linear_cbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    seed: int = 0,
    C: float = 1.0,
    max_iter: int = 2000,
) -> tuple[LogisticRegression, dict]:
    est = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
        n_jobs=1,
    )
    est.fit(X_train, y_train.astype(str).to_numpy())
    y_pred = pd.Series(est.predict(X_val), index=X_val.index)
    return est, compute_metrics(y_val, y_pred)


def concept_importance(est: LogisticRegression) -> pd.Series:
    """|coef| summed across classes — one weight per input feature."""
    names = list(getattr(est, "feature_names_in_", []))
    weights = np.abs(est.coef_).sum(axis=0)
    if not names:
        names = [f"x{i}" for i in range(len(weights))]
    return pd.Series(weights, index=names, name="linear_weight")


def interpretability_cost(est: LogisticRegression) -> int:
    """Non-zero coefficient count summed across classes."""
    return int((est.coef_ != 0.0).sum())
