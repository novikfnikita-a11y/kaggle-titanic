"""Stateless feature-generation operations for the Titanic project.

Every operation follows the same interface:

    operation(df, inputs, params) -> pandas.Series

These functions must not learn statistics from data. Imputation, encoding,
scaling and other fit-dependent transformations belong in preprocessing.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_input_columns(
    df: pd.DataFrame,
    inputs: Sequence[str],
    *,
    expected_count: int | None = None,
    operation: str = "operation",
) -> list[str]:
    """Validate and return the names of an operation's input columns."""
    if isinstance(inputs, str):
        columns = [inputs]
    else:
        columns = list(inputs)

    if not columns:
        raise ValueError(
            f"Operation '{operation}' requires at least one input column."
        )

    if expected_count is not None and len(columns) != expected_count:
        raise ValueError(
            f"Operation '{operation}' expects {expected_count} input column(s), "
            f"got {len(columns)}: {columns}"
        )

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(
            f"Operation '{operation}' requires missing column(s): {missing}"
        )

    return columns


def _get_single_input(
    df: pd.DataFrame,
    inputs: Sequence[str],
    operation: str,
) -> pd.Series:
    """Return exactly one input column after validation."""
    columns = _get_input_columns(
        df,
        inputs,
        expected_count=1,
        operation=operation,
    )
    return df[columns[0]]


def _get_numeric_inputs(
    df: pd.DataFrame,
    inputs: Sequence[str],
    operation: str,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Return input columns converted to numeric values; invalid values become NaN."""
    columns = _get_input_columns(
        df,
        inputs,
        expected_count=expected_count,
        operation=operation,
    )
    return df.loc[:, columns].apply(pd.to_numeric, errors="coerce")


def _get_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert a possibly OmegaConf-like mapping to a plain dictionary."""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError(
            "Operation params must be a mapping/dictionary, "
            f"got {type(params).__name__}."
        )
    return dict(params)


def _as_object_with_nan(series: pd.Series) -> pd.Series:
    """Return an object Series using np.nan for missing values.

    This gives sklearn's SimpleImputer a consistent missing-value marker even
    when string operations were performed with pandas' nullable StringDtype.
    """
    result = series.astype(object)
    return result.where(pd.notna(result), np.nan)


# ---------------------------------------------------------------------------
# Basic string / missingness operations
# ---------------------------------------------------------------------------


def not_null(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Return 1 when the input is present and 0 when it is missing."""
    series = _get_single_input(df, inputs, "not_null")
    return series.notna().astype("int8")


def first_char(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Extract the first non-whitespace character; empty/missing -> NaN."""
    series = _get_single_input(df, inputs, "first_char")
    result = series.astype("string").str.strip().str[:1]
    result = result.mask(result.eq(""), pd.NA)
    return _as_object_with_nan(result)


def extract_number(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Extract the first sequence of digits from a string as a number."""
    series = _get_single_input(df, inputs, "extract_number")
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("float64")


def string_length(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Return string length while preserving missing values as NaN."""
    series = _get_single_input(df, inputs, "string_length")
    return series.astype("string").str.len().astype("float64")


# ---------------------------------------------------------------------------
# Family operations
# ---------------------------------------------------------------------------


def family_size(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Calculate household size as SibSp + Parch + 1."""
    numeric = _get_numeric_inputs(
        df,
        inputs,
        "family_size",
        expected_count=2,
    )
    return numeric.sum(axis=1, min_count=2) + 1


def is_alone(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Return 1 when SibSp + Parch == 0, otherwise 0; missing inputs -> NaN."""
    numeric = _get_numeric_inputs(
        df,
        inputs,
        "is_alone",
        expected_count=2,
    )
    family_members = numeric.sum(axis=1, min_count=2)
    result = family_members.eq(0).astype("float64")
    return result.mask(family_members.isna(), np.nan)


def fare_per_person(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Calculate Fare / (SibSp + Parch + 1).

    Expected inputs, in order: [Fare, SibSp, Parch]. A missing input produces
    NaN; no missing values are filled in this stateless operation.
    """
    columns = _get_input_columns(
        df,
        inputs,
        expected_count=3,
        operation="fare_per_person",
    )
    fare_col, sibsp_col, parch_col = columns
    numeric = _get_numeric_inputs(df, columns, "fare_per_person")

    family_size_values = numeric[sibsp_col] + numeric[parch_col] + 1.0
    denominator = family_size_values.replace(0, np.nan)
    return numeric[fare_col] / denominator


# ---------------------------------------------------------------------------
# Identity operation
# ---------------------------------------------------------------------------


def extract_title(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Extract a passenger title and normalize it using a YAML mapping.

    Example: "Braund, Mr. Owen Harris" -> "Mr".

    Expected params:
        mapping: dictionary mapping extracted titles to normalized titles
        default: category for an extracted title absent from mapping
        missing_value: value for a missing/unparseable name (default NaN)
    """
    series = _get_single_input(df, inputs, "extract_title")
    operation_params = _get_params(params)

    mapping = operation_params.get("mapping") or {}
    if not isinstance(mapping, Mapping):
        raise TypeError("extract_title params.mapping must be a mapping/dictionary.")
    mapping = dict(mapping)

    default = operation_params.get("default", "Rare")
    missing_value = operation_params.get("missing_value", np.nan)

    title = (
        series.astype("string")
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
    )

    normalized = title.map(mapping)
    normalized = normalized.where(title.isna(), normalized.fillna(default))
    normalized = normalized.mask(title.isna(), missing_value)

    return _as_object_with_nan(normalized)


# ---------------------------------------------------------------------------
# Ticket operation
# ---------------------------------------------------------------------------


def ticket_prefix(
    df: pd.DataFrame,
    inputs: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Extract an alphabetic ticket prefix and normalize punctuation.

    Examples:
        "PC 17599"          -> "PC"
        "A/5 21171"         -> "A"
        "STON/O2. 3101282"  -> "STONO"
        "347082"            -> "NUMERIC"
    """
    series = _get_single_input(df, inputs, "ticket_prefix")
    operation_params = _get_params(params)

    numeric_label = operation_params.get("numeric_label", "NUMERIC")
    missing_value = operation_params.get("missing_value", np.nan)

    ticket = series.astype("string").str.upper().str.strip()
    prefix = (
        ticket.str.replace(r"\d+", "", regex=True)
        .str.replace(r"[^A-Z]+", "", regex=True)
        .str.strip()
    )

    # A non-missing ticket without letters is a numeric-only ticket.
    prefix = prefix.mask(prefix.eq(""), numeric_label)
    # Empty strings and actual missing values are considered missing.
    prefix = prefix.mask(ticket.isna() | ticket.eq(""), missing_value)

    return _as_object_with_nan(prefix)


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

OPERATIONS = {
    "not_null": not_null,
    "first_char": first_char,
    "extract_number": extract_number,
    "string_length": string_length,
    "family_size": family_size,
    "is_alone": is_alone,
    "fare_per_person": fare_per_person,
    "extract_title": extract_title,
    "ticket_prefix": ticket_prefix,
}
