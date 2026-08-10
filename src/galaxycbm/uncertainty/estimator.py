"""sklearn-compatible wrapper around the Stage-2 SymPy rule set.

MAPIE's SplitConformalClassifier expects an estimator that quacks like a
sklearn ClassifierMixin — `classes_`, `predict`, `predict_proba`. Our
symbolic head is a bag of SymPy expressions, so we glue one to the other
with a stable-softmax over per-class scores.

Kept intentionally small: no learning, no gradients — just evaluate rules,
softmax, argmax.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from galaxycbm.symbolic import ClassRule, score_expressions


def _stable_softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    # SymPy rules can emit inf / -inf / nan when a rule divides by zero on a
    # shifted-survey row. Clip to a large finite range so the softmax stays
    # well-defined; nan → 0 (uninformative row).
    a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
    a = a - np.max(a, axis=axis, keepdims=True)
    exp = np.exp(a)
    denom = np.sum(exp, axis=axis, keepdims=True)
    # Guard against an all-underflow row: fall back to a uniform prior.
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(denom > 0, exp / denom, 1.0 / a.shape[axis])
    return probs


class SymbolicRuleClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a list[ClassRule] as a fit-once sklearn classifier.

    Parameters
    ----------
    rules
        The Stage-2 output. Each rule provides an expression over `feature_columns`.
    feature_columns
        Column order used at score time; must match the training feature order.
    temperature
        Softmax temperature. 1.0 leaves the scores as-is; smaller values
        sharpen the resulting probabilities.
    """

    def __init__(
        self,
        rules: list[ClassRule] | None = None,
        feature_columns: list[str] | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.rules = rules
        self.feature_columns = feature_columns
        self.temperature = temperature

    # sklearn requires `fit`. We're prefit — this is just a no-op that
    # sets `classes_` from the rule list so cross_val_predict / MAPIE
    # can introspect us cleanly.
    def fit(self, X, y=None) -> "SymbolicRuleClassifier":
        if self.rules is None:
            raise ValueError("SymbolicRuleClassifier requires `rules` at construction")
        self.classes_ = np.array([r.hubble_class for r in self.rules])
        return self

    def _as_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if self.feature_columns:
                return X[self.feature_columns]
            return X
        if not self.feature_columns:
            raise ValueError("feature_columns is required when passing a numpy array")
        return pd.DataFrame(np.asarray(X), columns=self.feature_columns)

    def _raw_scores(self, X) -> pd.DataFrame:
        if self.rules is None:
            raise RuntimeError("call fit() first")
        return score_expressions(self.rules, self._as_dataframe(X))

    def predict_proba(self, X) -> np.ndarray:
        scores = self._raw_scores(X).to_numpy(float)
        # Order the columns to match self.classes_
        # score_expressions preserves rule order, which matches classes_.
        probs = _stable_softmax(scores / float(self.temperature), axis=-1)
        return probs

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=-1)]
