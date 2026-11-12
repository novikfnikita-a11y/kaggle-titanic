"""Build leakage-safe sklearn preprocessing from compiled FeatureSpec objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .builder import FeatureSpec


def _plain_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """Return a regular dictionary for an optional configuration mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"'{context}' must be a mapping/dictionary.")
    return dict(value)


def _build_imputer(
    impute_config: Any,
    *,
    kind: str,
    feature_name: str,
) -> SimpleImputer:
    """Create the imputer for one feature."""
    config = _plain_mapping(
        impute_config,
        context=f"{feature_name}.impute",
    )

    default_strategy = "median" if kind == "numeric" else "most_frequent"
    strategy = config.get("strategy", default_strategy)

    # Accept the common informal alias, but use sklearn's canonical spelling.
    if strategy == "mode":
        strategy = "most_frequent"

    allowed = {"mean", "median", "most_frequent", "constant"}
    if strategy not in allowed:
        raise ValueError(
            f"Feature '{feature_name}': unsupported imputation strategy "
            f"'{strategy}'. Choose from {sorted(allowed)}."
        )

    kwargs: dict[str, Any] = {
        "strategy": strategy,
        "keep_empty_features": True,
    }
    if strategy == "constant":
        default_fill = 0 if kind == "numeric" else "Unknown"
        kwargs["fill_value"] = config.get("value", default_fill)

    return SimpleImputer(**kwargs)


def _build_scaler(scale: Any, *, feature_name: str):
    """Create an optional scaler for a numeric feature."""
    if scale in (None, False, "none", "passthrough"):
        return None

    scalers = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }
    if scale not in scalers:
        raise ValueError(
            f"Feature '{feature_name}': unsupported scale '{scale}'. "
            f"Choose from {sorted(scalers)} or 'none'."
        )
    return scalers[scale]()


def _log1p_nonnegative(X):
    """Apply log1p and fail clearly for negative values below -1."""
    values = np.asarray(X, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("log1p=True requires non-negative feature values.")
    return np.log1p(values)


def _build_numeric_pipeline(spec: FeatureSpec) -> Pipeline:
    """Build imputation -> optional log1p -> optional scaling."""
    config = spec.preprocessing
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            _build_imputer(
                config.get("impute"),
                kind="numeric",
                feature_name=spec.name,
            ),
        )
    ]

    if bool(config.get("log1p", False)):
        steps.append(
            (
                "log1p",
                FunctionTransformer(
                    _log1p_nonnegative,
                    feature_names_out="one-to-one",
                    validate=False,
                ),
            )
        )

    scaler = _build_scaler(config.get("scale", "none"), feature_name=spec.name)
    if scaler is not None:
        steps.append(("scaler", scaler))

    return Pipeline(steps)


def _build_categorical_pipeline(spec: FeatureSpec) -> Pipeline:
    """Build imputation followed by one-hot or ordinal encoding."""
    config = spec.preprocessing
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            _build_imputer(
                config.get("impute"),
                kind="categorical",
                feature_name=spec.name,
            ),
        )
    ]

    encode = config.get("encode", "onehot")
    order = config.get("order")

    # Also allow an encoding mapping, e.g. encode: {type: onehot}.
    encoder_options: dict[str, Any] = {}
    if isinstance(encode, Mapping):
        encoder_options = dict(encode)
        encode = encoder_options.pop("type", "onehot")

    if encode == "onehot":
        allowed_options = {"handle_unknown", "min_frequency", "max_categories", "drop"}
        unexpected = set(encoder_options) - allowed_options
        if unexpected:
            raise ValueError(
                f"Feature '{spec.name}': unsupported OneHotEncoder options: "
                f"{sorted(unexpected)}"
            )
        steps.append(
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=encoder_options.get("handle_unknown", "ignore"),
                    min_frequency=encoder_options.get("min_frequency"),
                    max_categories=encoder_options.get("max_categories"),
                    drop=encoder_options.get("drop"),
                    sparse_output=True,
                ),
            )
        )
    elif encode == "ordinal":
        categories = "auto" if order is None else [list(order)]
        steps.append(
            (
                "encoder",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            )
        )
    else:
        raise ValueError(
            f"Feature '{spec.name}': unsupported encoding '{encode}'. "
            "Choose 'onehot' or 'ordinal'."
        )

    return Pipeline(steps)


def _transformer_name(index: int, feature_name: str, suffix: str) -> str:
    """Create a valid, unique ColumnTransformer step name."""
    import re

    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", feature_name).strip("_")
    if not safe_name:
        safe_name = "feature"
    return f"f{index}_{safe_name}_{suffix}"


def build_preprocessor(feature_specs: Sequence[FeatureSpec]) -> ColumnTransformer:
    """Build a per-feature ColumnTransformer from compiled feature specs.

    All fit-dependent operations (imputation, encoding and scaling) are fitted
    only on the training data passed to ``Pipeline.fit``. During cross-validation,
    sklearn clones and fits this preprocessor independently inside each fold.

    Numeric missing indicators are added as separate, unscaled columns rather
    than using SimpleImputer(add_indicator=True), so StandardScaler never scales
    the indicator values.
    """
    transformers: list[tuple[str, Any, list[str]]] = []

    if not feature_specs:
        raise ValueError("Cannot build a preprocessor without any selected features.")

    for index, spec in enumerate(feature_specs):
        if spec.kind == "numeric":
            transformers.append(
                (
                    _transformer_name(index, spec.name, "value"),
                    _build_numeric_pipeline(spec),
                    [spec.name],
                )
            )

            if bool(spec.preprocessing.get("missing_indicator", False)):
                transformers.append(
                    (
                        _transformer_name(index, spec.name, "missing"),
                        MissingIndicator(
                            features="all",
                            error_on_new=False,
                        ),
                        [spec.name],
                    )
                )

        elif spec.kind == "categorical":
            transformers.append(
                (
                    _transformer_name(index, spec.name, "category"),
                    _build_categorical_pipeline(spec),
                    [spec.name],
                )
            )
        else:
            raise ValueError(
                f"Feature '{spec.name}' has unsupported type '{spec.kind}'."
            )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
