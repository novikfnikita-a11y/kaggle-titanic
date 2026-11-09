"""Dataset loading and basic train/test schema validation."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


@dataclass(frozen=True)
class DatasetBundle:
    """Titanic data with identifiers kept separate from model features."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    train_ids: pd.Series
    test_ids: pd.Series


def load_data(config: DictConfig, project_root: str | Path) -> DatasetBundle:
    """Load train/test CSVs, separate target and IDs, and validate columns.

    Passenger identifiers are deliberately excluded from model features but
    retained for constructing a correctly ordered Kaggle submission later.
    Test columns are reordered to match the training feature order.
    """
    root = Path(project_root).expanduser().resolve()
    data_config = config.data
    target_column = str(data_config.target)
    id_column = str(data_config.id_column)
    train_path = root / str(data_config.train_path)
    test_path = root / str(data_config.test_path)

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if train_df.empty:
        raise ValueError(f"Training dataset is empty: {train_path}")
    if test_df.empty:
        raise ValueError(f"Test dataset is empty: {test_path}")

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column {target_column!r} is missing from {train_path}"
        )
    if target_column in test_df.columns:
        raise ValueError(
            f"Target column {target_column!r} unexpectedly exists in {test_path}"
        )

    for frame_name, frame, path in (
        ("training", train_df, train_path),
        ("test", test_df, test_path),
    ):
        if id_column not in frame.columns:
            raise ValueError(
                f"ID column {id_column!r} is missing from {frame_name} data: {path}"
            )

    train_feature_columns = [
        column
        for column in train_df.columns
        if column not in {target_column, id_column}
    ]
    test_feature_columns = [column for column in test_df.columns if column != id_column]

    missing_in_test = sorted(set(train_feature_columns) - set(test_feature_columns))
    extra_in_test = sorted(set(test_feature_columns) - set(train_feature_columns))
    if missing_in_test or extra_in_test:
        raise ValueError(
            "Train/test feature schema mismatch. "
            f"Missing in test: {missing_in_test}; extra in test: {extra_in_test}"
        )

    x_train = train_df.loc[:, train_feature_columns].copy()
    x_test = test_df.loc[:, train_feature_columns].copy()
    y_train = train_df[target_column].copy()
    train_ids = train_df[id_column].copy()
    test_ids = test_df[id_column].copy()

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        train_ids=train_ids,
        test_ids=test_ids,
    )
