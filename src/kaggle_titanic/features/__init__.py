"""Public API for Titanic feature engineering."""

# Feature configuration and generation
from .builder import (
    FeatureSpec,
    FeatureGenerator,
    compile_feature_specs,
    build_feature_pipeline,
)

# Preprocessing
from .preprocessing import (
    build_preprocessor,
)

# Feature operations
from .operations import (
    OPERATIONS,
    not_null,
    first_char,
    extract_number,
    string_length,
    family_size,
    is_alone,
    fare_per_person,
    extract_title,
    ticket_prefix,
)


__all__ = [
    # Feature configuration and generation
    "FeatureSpec",
    "FeatureGenerator",
    "compile_feature_specs",
    "build_feature_pipeline",

    # Preprocessing
    "build_preprocessor",

    # Operation registry
    "OPERATIONS",

    # Operations
    "not_null",
    "first_char",
    "extract_number",
    "string_length",
    "family_size",
    "is_alone",
    "fare_per_person",
    "extract_title",
    "ticket_prefix",
]