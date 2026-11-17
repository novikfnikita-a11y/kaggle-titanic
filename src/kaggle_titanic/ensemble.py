from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FoldEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: Sequence[Any] = ()):
        self.estimators = estimators

    def fit(self, X=None, y=None) -> "FoldEnsemble":
        return self

    def predict_proba(self, X):
        return np.mean(
            [estimator.predict_proba(X) for estimator in self.estimators],
            axis=0,
        )

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
