"""Compile Hydra feature configuration and build sklearn feature pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .operations import OPERATIONS


_PREPROCESSING_KEYS = {
    "impute",
    "scale",
    "missing_indicator",
    "log1p",
    "encode",
    "order",
}


@dataclass(frozen=True)
class FeatureSpec:
    """A compiled description of one selected output feature."""

    name: str
    kind: str
    inputs: tuple[str, ...]
    operation: str | None
    params: dict[str, Any]
    preprocessing: dict[str, Any]
    source: str


def _to_plain_container(value: Any) -> Any:
    """Convert OmegaConf containers to regular Python containers."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return deepcopy(value)


def _as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """Convert a configuration node to a plain dictionary."""
    plain = _to_plain_container(value)
    if plain is None:
        return {}
    if not isinstance(plain, Mapping):
        raise TypeError(f"Configuration node '{context}' must be a mapping.")
    return dict(plain)


def _as_inputs(value: Any, *, feature_name: str) -> tuple[str, ...]:
    """Normalize a single input name or a sequence of names to a tuple."""
    if isinstance(value, str):
        inputs = (value,)
    elif isinstance(value, (Sequence, ListConfig)):
        inputs = tuple(value)
    else:
        raise TypeError(
            f"Feature '{feature_name}': 'inputs' must be a string or a list of strings."
        )

    if not inputs or any(not isinstance(column, str) or not column for column in inputs):
        raise ValueError(
            f"Feature '{feature_name}' must declare at least one non-empty input column."
        )
    return inputs


def compile_feature_specs(features_config: DictConfig | Mapping[str, Any]) -> tuple[FeatureSpec, ...]:
    """Compile ``cfg.features`` into ordered, validated feature specifications.

    Expected Hydra layout:

        features:
          column_order: [basic, family, cabin, identity, ticket]
          basic:
            raw:
              numeric: ...
              categorical: ...
          family:
            derived: ...

    Groups set to null are skipped. ``column_order`` determines feature order.
    """
    root = _as_mapping(features_config, context="features")

    column_order = root.get("column_order")
    if column_order is None:
        raise ValueError(
            "features.column_order is required and must list feature groups in order."
        )
    if isinstance(column_order, str) or not isinstance(column_order, Sequence):
        raise TypeError("features.column_order must be a list of group names.")

    specs: list[FeatureSpec] = []
    seen_names: set[str] = set()

    def add_spec(spec: FeatureSpec) -> None:
        if spec.name in seen_names:
            raise ValueError(
                f"Duplicate feature name '{spec.name}'. Raw and derived feature "
                "names must be unique across all enabled groups."
            )
        seen_names.add(spec.name)
        specs.append(spec)

    for group_name in column_order:
        if not isinstance(group_name, str):
            raise TypeError("Every item in features.column_order must be a string.")

        group_raw = root.get(group_name)
        if group_raw is None:
            # A group may intentionally be disabled via a null Defaults List entry.
            continue

        group = _as_mapping(group_raw, context=f"features.{group_name}")

        # Raw features: the section name defines the feature type.
        raw = _as_mapping(group.get("raw"), context=f"features.{group_name}.raw")
        for section_name, kind in (("numeric", "numeric"), ("categorical", "categorical")):
            section = _as_mapping(
                raw.get(section_name),
                context=f"features.{group_name}.raw.{section_name}",
            )

            for feature_name, feature_raw in section.items():
                if not isinstance(feature_name, str):
                    raise TypeError("Raw feature names must be strings.")

                feature_cfg = _as_mapping(
                    feature_raw,
                    context=f"features.{group_name}.raw.{section_name}.{feature_name}",
                )
                if not bool(feature_cfg.get("enabled", True)):
                    continue

                preprocessing = {
                    key: deepcopy(feature_cfg[key])
                    for key in _PREPROCESSING_KEYS
                    if key in feature_cfg
                }
                add_spec(
                    FeatureSpec(
                        name=feature_name,
                        kind=kind,
                        inputs=(feature_name,),
                        operation=None,
                        params={},
                        preprocessing=preprocessing,
                        source=f"{group_name}.raw.{section_name}",
                    )
                )

        # Derived features.
        derived = _as_mapping(
            group.get("derived"),
            context=f"features.{group_name}.derived",
        )
        for feature_name, feature_raw in derived.items():
            if not isinstance(feature_name, str):
                raise TypeError("Derived feature names must be strings.")

            feature_cfg = _as_mapping(
                feature_raw,
                context=f"features.{group_name}.derived.{feature_name}",
            )
            if not bool(feature_cfg.get("enabled", True)):
                continue

            inputs = _as_inputs(feature_cfg.get("inputs"), feature_name=feature_name)
            operation = feature_cfg.get("op")
            if not isinstance(operation, str) or not operation:
                raise ValueError(
                    f"Derived feature '{feature_name}' must declare a non-empty 'op'."
                )
            if operation not in OPERATIONS:
                available = ", ".join(sorted(OPERATIONS))
                raise ValueError(
                    f"Unknown operation '{operation}' for feature '{feature_name}'. "
                    f"Available operations: {available}"
                )

            kind = feature_cfg.get("type")
            if kind not in {"numeric", "categorical"}:
                raise ValueError(
                    f"Derived feature '{feature_name}' must have type 'numeric' "
                    "or 'categorical'."
                )

            params = _as_mapping(
                feature_cfg.get("params"),
                context=f"features.{group_name}.derived.{feature_name}.params",
            )
            preprocessing = {
                key: deepcopy(feature_cfg[key])
                for key in _PREPROCESSING_KEYS
                if key in feature_cfg
            }

            add_spec(
                FeatureSpec(
                    name=feature_name,
                    kind=kind,
                    inputs=inputs,
                    operation=operation,
                    params=params,
                    preprocessing=preprocessing,
                    source=f"{group_name}.derived",
                )
            )

    if not specs:
        raise ValueError(
            "No features are enabled. Enable at least one raw or derived feature."
        )

    return tuple(specs)


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate enabled raw and derived columns from the original DataFrame.

    The transformer deliberately receives the complete original DataFrame.
    This allows a raw feature to be excluded from the final model matrix while
    still being used as an input to an enabled derived feature.
    """

    def __init__(self, feature_specs: tuple[FeatureSpec, ...]):
        # Do not modify constructor parameters: sklearn.clone relies on this.
        self.feature_specs = feature_specs

    @staticmethod
    def _validate_dataframe(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureGenerator expects a pandas DataFrame. "
                f"Received {type(X).__name__}."
            )
        if X.columns.has_duplicates:
            duplicates = X.columns[X.columns.duplicated()].tolist()
            raise ValueError(f"Input DataFrame contains duplicate columns: {duplicates}")
        return X

    def _required_input_columns(self) -> list[str]:
        required: list[str] = []
        seen: set[str] = set()
        for spec in self.feature_specs:
            for column in spec.inputs:
                if column not in seen:
                    required.append(column)
                    seen.add(column)
        return required

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureGenerator":
        X = self._validate_dataframe(X)
        required = self._required_input_columns()
        missing = [column for column in required if column not in X.columns]
        if missing:
            raise ValueError(
                f"Feature generation requires columns absent from X: {missing}"
            )

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.output_feature_names_ = np.asarray(
            [spec.name for spec in self.feature_specs],
            dtype=object,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["feature_names_in_", "output_feature_names_"])
        X = self._validate_dataframe(X)

        required = self._required_input_columns()
        missing = [column for column in required if column not in X.columns]
        if missing:
            raise ValueError(
                f"Feature generation requires columns absent from X: {missing}"
            )

        output_columns: dict[str, pd.Series] = {}

        for spec in self.feature_specs:
            if spec.operation is None:
                result = X[spec.inputs[0]].copy()
            else:
                operation_fn = OPERATIONS[spec.operation]
                result = operation_fn(
                    X,
                    inputs=spec.inputs,
                    params=spec.params,
                )

            if isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"Operation '{spec.operation}' for '{spec.name}' returned a "
                    "DataFrame; each operation must return a one-dimensional Series."
                )
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=X.index)

            if len(result) != len(X):
                raise ValueError(
                    f"Feature '{spec.name}' returned {len(result)} rows; "
                    f"expected {len(X)}."
                )
            if not result.index.equals(X.index):
                raise ValueError(
                    f"Feature '{spec.name}' changed the input row index/order."
                )

            output_columns[spec.name] = result.rename(spec.name)

        output = pd.DataFrame(output_columns, index=X.index)
        if output.columns.has_duplicates:
            raise RuntimeError("Generated feature names are not unique.")
        return output

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        check_is_fitted(self, attributes=["output_feature_names_"])
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if hasattr(self, "feature_names_in_") and not np.array_equal(
                input_features, self.feature_names_in_
            ):
                raise ValueError(
                    "input_features must match the feature names seen during fit."
                )
        return self.output_feature_names_.copy()


def build_feature_pipeline(
    features_config: DictConfig | Mapping[str, Any],
    estimator: BaseEstimator | None = None,
):
    """Build ``FeatureGenerator -> Preprocessor -> optional estimator``.

    Parameters
    ----------
    features_config:
        The composed ``cfg.features`` Hydra node.
    estimator:
        An instantiated sklearn-compatible estimator. If omitted, the returned
        Pipeline contains feature generation and preprocessing only.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A cloneable pipeline suitable for cross-validation.
    """
    from sklearn.pipeline import Pipeline

    # Local import avoids a circular import between the compiler and preprocessor.
    from .preprocessing import build_preprocessor

    feature_specs = compile_feature_specs(features_config)

    steps = [
        ("feature_generator", FeatureGenerator(feature_specs)),
        ("preprocessor", build_preprocessor(feature_specs)),
    ]
    if estimator is not None:
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(
                "estimator must be a scikit-learn BaseEstimator instance. "
                f"Received {type(estimator).__name__}."
            )
        steps.append(("estimator", estimator))

    return Pipeline(steps=steps)
