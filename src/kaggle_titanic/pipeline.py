from __future__ import annotations

from typing import Any, Iterable

from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import FeatureEngineer
from .transformers import SelectColumns


def resolve_feature_columns(
    groups: DictConfig,
    feature_set: DictConfig,
) -> tuple[list[str], list[str], dict[str, list]]:
    exclude = set(feature_set.get("exclude", []) or [])

    numeric: list[str] = []
    categorical: list[str] = []
    ordinal: dict[str, list] = {}

    for name in feature_set.use:
        group = groups.get(name)
        if group is None:
            raise KeyError(f"Unknown feature group: {name!r}")

        numeric += [c for c in group.get("numeric", []) or [] if c not in exclude]
        categorical += [
            c for c in group.get("categorical", []) or [] if c not in exclude
        ]
        for column, categories in (group.get("ordinal", {}) or {}).items():
            if column not in exclude:
                ordinal[column] = list(categories)

    return numeric, categorical, ordinal


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    ordinal: dict[str, list] | None = None,
    estimator: Any | None = None,
    imputer: Any | None = None,
    encoders: Iterable[Any] = (),
    scale: bool = True,
    native_categorical: bool = False,
) -> Pipeline:
    ordinal = ordinal or {}

    steps: list[tuple[str, Any]] = [("features", FeatureEngineer())]

    encoders = list(encoders)
    if encoders:
        steps.append(
            ("encode", Pipeline([(f"enc{i}", e) for i, e in enumerate(encoders)]))
        )

    if native_categorical:
        cat_columns = list(categorical) + list(ordinal)
        steps.append(("select", SelectColumns(numeric, cat_columns)))
        if estimator is not None:
            steps.append(("clf", estimator))
        return Pipeline(steps=steps)

    if imputer is not None:
        steps.append(("impute", imputer))

    transformers = []
    if numeric:
        num_steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("sc", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric))

    if categorical:
        cat_steps = [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical))

    if ordinal:
        ord_columns = list(ordinal)
        ord_categories = [list(ordinal[column]) for column in ord_columns]
        ord_steps = [
            ("imp", SimpleImputer(strategy="most_frequent")),
            (
                "enc",
                OrdinalEncoder(
                    categories=ord_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
        if scale:
            ord_steps.append(("sc", StandardScaler()))
        transformers.append(("ord", Pipeline(ord_steps), ord_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
    steps.append(("prep", preprocessor))

    if estimator is not None:
        steps.append(("clf", estimator))

    return Pipeline(steps=steps)
