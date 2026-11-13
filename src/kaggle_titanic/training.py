"""Model training and cross-validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from omegaconf import DictConfig

from .data import DatasetBundle


@dataclass
class TrainingResult:
    metrics: dict[str, float]
    fold_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    fitted_pipeline: Pipeline


def train_model(
    pipeline: Pipeline,
    dataset: DatasetBundle,
    cv_config: DictConfig,
    id_column: str = "PassengerId",
) -> TrainingResult:

    X = dataset.x_train
    y = dataset.y_train
    X_test = dataset.x_test

    if cv_config.splitter != "stratified_kfold":
        raise ValueError(
            f"Unsupported CV splitter: {cv_config.splitter}"
        )

    splitter = StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=(
            cv_config.random_state
            if cv_config.shuffle
            else None
        ),
    )

    print("\n cv started")

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring="accuracy",
        return_estimator=True,
        error_score="raise",
    )

    fold_scores = cv_results["test_score"]
    n_folds = len(fold_scores)

    fold_results = [
        {"fold": i + 1, "accuracy": float(score)}
        for i, score in enumerate(fold_scores)
    ]

    for rec in fold_results:
        print(f" fold {rec['fold']}/{n_folds}")
        print(f"  accuracy: {rec['accuracy']:.4f}")

    fold_metrics_df = pd.DataFrame(fold_results)

    cv_accuracy_mean = float(np.mean(fold_scores))
    cv_accuracy_std = float(np.std(fold_scores, ddof=1)) if n_folds > 1 else 0.0

    metrics = {
        "cv_accuracy_mean": cv_accuracy_mean,
        "cv_accuracy_std": cv_accuracy_std,
    }

    print("\n fitting final model on full train data")

    fitted_pipeline = clone(pipeline)

    fitted_pipeline.fit(X, y)

    test_predictions_array = fitted_pipeline.predict(X_test)

    test_predictions_df = pd.DataFrame({
        id_column: dataset.test_ids.to_numpy(),
        "Survived": test_predictions_array,
    })

    print("\n cv finished")
    print(f"cv accuracy mean: {cv_accuracy_mean:.4f}")
    print(f"cv accuracy std:  {cv_accuracy_std:.4f}")

    return TrainingResult(
        metrics=metrics,
        fold_metrics=fold_metrics_df,
        test_predictions=test_predictions_df,
        fitted_pipeline=fitted_pipeline,
    )