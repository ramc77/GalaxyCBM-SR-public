"""(c) XGBoost on PREDICTED concepts.

Lives under the `baselines` extra (xgboost + shap). Import lazily so the
base env still passes pytest on Intel Macs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from galaxycbm.symbolic import compute_metrics


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    seed: int = 0,
    n_estimators: int = 300,
    max_depth: int = 4,
    lr: float = 0.1,
) -> tuple[object, dict, object]:
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    le = LabelEncoder().fit(y_train.astype(str))
    est = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
    )
    est.fit(X_train, le.transform(y_train.astype(str)))
    y_pred_num = est.predict(X_val)
    y_pred = pd.Series(le.inverse_transform(y_pred_num), index=X_val.index)
    return est, compute_metrics(y_val, y_pred), le


def concept_importance(est) -> pd.Series:
    imp = est.feature_importances_
    names = list(est.get_booster().feature_names or [])
    if not names:
        names = [f"x{i}" for i in range(len(imp))]
    return pd.Series(imp, index=names, name="xgb_importance")


def shap_importance(est, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature over rows of X (post-hoc explanation)."""
    import shap

    explainer = shap.TreeExplainer(est)
    values = explainer.shap_values(X)
    arr = np.asarray(values)
    # Multi-class → (n_classes, n_samples, n_features) or list of arrays.
    if arr.ndim == 3:
        mean_abs = np.abs(arr).mean(axis=(0, 1))
    elif arr.ndim == 2:
        mean_abs = np.abs(arr).mean(axis=0)
    else:
        mean_abs = np.abs(arr).mean(axis=tuple(range(arr.ndim - 1)))
    return pd.Series(mean_abs, index=list(X.columns), name="xgb_shap")


def interpretability_cost(est) -> int:
    """Non-zero feature-importance count."""
    return int((est.feature_importances_ > 0).sum())
