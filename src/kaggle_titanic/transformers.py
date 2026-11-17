from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str = "Age",
        group_by: tuple[str, ...] = ("Title", "Pclass"),
    ):
        self.column = column
        self.group_by = group_by

    def fit(self, X: pd.DataFrame, y=None) -> "GroupMedianImputer":
        self.global_median_ = X[self.column].median()

        medians = X.groupby(list(self.group_by))[self.column].median()
        self.group_medians_ = {
            key if isinstance(key, tuple) else (key,): value
            for key, value in medians.items()
        }
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()

        keys = zip(*(X[column] for column in self.group_by))
        fill = pd.Series(
            [self.group_medians_.get(key, self.global_median_) for key in keys],
            index=X.index,
        ).fillna(self.global_median_)

        X[self.column] = X[self.column].fillna(fill)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str,
        output_column: str | None = None,
    ):
        self.column = column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        self.frequencies_ = X[self.column].value_counts(dropna=False).to_dict()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()

        name = self.output_column or f"{self.column}_freq"
        X[name] = X[self.column].map(self.frequencies_).fillna(0).astype(float)
        return X


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, numeric=(), categorical=()):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X: pd.DataFrame, y=None) -> "SelectColumns":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for column in self.categorical:
            values = X[column].where(X[column].notna(), "Missing")
            X[column] = values.astype(str).astype("category")
        return X[list(self.numeric) + list(self.categorical)]
