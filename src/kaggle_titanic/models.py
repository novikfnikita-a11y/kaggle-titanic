from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostNative(BaseEstimator, ClassifierMixin):
    def __init__(self, cat_features: tuple = (), params: dict | None = None):
        self.cat_features = cat_features
        self.params = params

    def fit(self, X, y, **fit_params) -> "CatBoostNative":
        self.model_ = CatBoostClassifier(
            cat_features=list(self.cat_features),
            **(self.params or {}),
        )
        self.model_.fit(X, y, **fit_params)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self) -> Any:
        return self.model_.feature_importances_
