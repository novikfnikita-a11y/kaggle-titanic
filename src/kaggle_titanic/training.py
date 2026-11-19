from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .data import DatasetBundle


class FoldEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: Sequence[Any] = ()):
        self.estimators = estimators

    def fit(self, X=None, y=None) -> "FoldEnsemble":
        return self

    def predict_proba(self, X):
        return np.mean(
            [estimator.predict_proba(X) for estimator in self.estimators],
            axis=0,
        )

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class TrainingResult:
    metrics: dict[str, float]
    fold_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    model: Any


def _build_model(
    ensemble: bool,
    pipeline: Pipeline,
    fold_estimators: list[Pipeline] | None,
    X: pd.DataFrame,
    y: pd.Series,
) -> Any:
    if ensemble:
        return FoldEnsemble(estimators=fold_estimators)
    return clone(pipeline).fit(X, y)


def train_model(
    pipeline: Pipeline,
    dataset: DatasetBundle,
    cv_config: DictConfig,
    id_column: str = "PassengerId",
) -> TrainingResult:

    X = dataset.x_train
    y = dataset.y_train
    X_test = dataset.x_test

    ensemble = cv_config.get("ensemble", True)

    splitter = instantiate(cv_config.splitter)

    print(f"\n cv started (ensemble={ensemble})")

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring="accuracy",
        return_estimator=ensemble,
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

    print(f"\n building final model (ensemble={ensemble})")

    fold_estimators = list(cv_results["estimator"]) if ensemble else None
    model = _build_model(ensemble, pipeline, fold_estimators, X, y)

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
