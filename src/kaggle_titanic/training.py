from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .data import DatasetBundle
from .ensemble import FoldEnsemble

AGGREGATIONS = ("fold_ensemble", "full_fit")


@dataclass
class TrainingResult:
    metrics: dict[str, float]
    fold_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    model: Any


def _build_model(
    aggregate: str,
    pipeline: Pipeline,
    fold_estimators: list[Pipeline] | None,
    X: pd.DataFrame,
    y: pd.Series,
) -> Any:
    if aggregate == "fold_ensemble":
        return FoldEnsemble(estimators=fold_estimators)

    if aggregate == "full_fit":
        return clone(pipeline).fit(X, y)

    raise ValueError(f"Unsupported aggregate: {aggregate}")


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

    aggregate = cv_config.get("aggregate", "fold_ensemble")
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"Unsupported aggregate: {aggregate}")

    splitter = StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=(
            cv_config.random_state
            if cv_config.shuffle
            else None
        ),
    )

    print(f"\n cv started (aggregate={aggregate})")

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring="accuracy",
        return_estimator=aggregate == "fold_ensemble",
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

    print(f"\n building final model ({aggregate})")

    fold_estimators = (
        list(cv_results["estimator"]) if aggregate == "fold_ensemble" else None
    )
    model = _build_model(aggregate, pipeline, fold_estimators, X, y)

    test_predictions_array = model.predict(X_test)

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
        model=model,
    )
