from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


@dataclass(frozen=True)
class DatasetBundle:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    test_ids: pd.Series


def load_data(config: DictConfig, project_root: str | Path) -> DatasetBundle:
    root = Path(project_root).expanduser().resolve()
    data_config = config.data
    target_column = str(data_config.target)
    id_column = str(data_config.id_column)

    train_df = pd.read_csv(root / str(data_config.train_path))
    test_df = pd.read_csv(root / str(data_config.test_path))

    feature_columns = [
        column
        for column in train_df.columns
        if column not in {target_column, id_column}
    ]

    return DatasetBundle(
        x_train=train_df[feature_columns].copy(),
        y_train=train_df[target_column].copy(),
        x_test=test_df[feature_columns].copy(),
        test_ids=test_df[id_column].copy(),
    )
