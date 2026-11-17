import hashlib
import json
from typing import Any, Mapping

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf


def config_fingerprint(cfg: DictConfig) -> str:
    data = OmegaConf.to_container(cfg, resolve=True)
    for key in ("hydra", "mlflow", "cli"):
        data.pop(key, None)
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def flatten_config(cfg: DictConfig) -> dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    result: dict[str, Any] = {}

    def _walk(current: dict, parent_key: str = "") -> None:
        for key, value in current.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                _walk(value, new_key)
            elif isinstance(value, list):
                result[new_key] = ",".join(map(str, value))
            else:
                result[new_key] = value

    _walk(cfg_dict)
    return result


def log_config(cfg: DictConfig) -> None:
    mlflow.log_dict(
        OmegaConf.to_container(cfg, resolve=True),
        "config/resolved_config.json",
    )


def log_params(cfg: DictConfig) -> None:
    params = {
        key: str(value) if value is not None else "None"
        for key, value in flatten_config(cfg).items()
    }
    mlflow.log_params(params)


def log_tags(cfg: DictConfig, config_hash: str | None = None) -> None:
    tags = {
        "model": str(cfg.model.name),
        "seed": str(cfg.seed),
        "cv": f"{cfg.cv.splitter}_{cfg.cv.n_splits}",
        "feature_set": str(cfg.feature_set.name),
    }
    if config_hash:
        tags["config_hash"] = config_hash
    mlflow.set_tags(tags)


def log_selected_features(features: list[str]) -> None:
    mlflow.log_param("n_selected_features", len(features))
    mlflow.log_dict({"selected_features": features}, "features/selected_features.json")


def log_metrics(metrics: Mapping[str, float]) -> None:
    mlflow.log_metrics(dict(metrics))


def _log_csv(dataframe: pd.DataFrame, path: str) -> None:
    mlflow.log_text(dataframe.to_csv(index=False), path)


def log_fold_metrics(fold_metrics: pd.DataFrame) -> None:
    _log_csv(fold_metrics, "metrics/fold_metrics.csv")


def log_predictions(predictions: pd.DataFrame) -> None:
    _log_csv(predictions, "predictions/test_predictions.csv")


def log_submission(predictions: pd.DataFrame) -> None:
    _log_csv(predictions, "submission/submission.csv")


def log_model(model: Any, X_train_sample: pd.DataFrame | None = None) -> str:
    signature: ModelSignature | None = None
    if X_train_sample is not None:
        signature = infer_signature(X_train_sample, model.predict(X_train_sample))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        serialization_format="cloudpickle",
    )
    return f"runs:/{mlflow.active_run().info.run_id}/model"


def register_model(
    model_uri: str,
    name: str,
    tags: dict | None = None,
    description: str | None = None,
    min_accuracy: float | None = None,
) -> None:
    current_acc = (
        float(tags["cv_accuracy_mean"])
        if tags and "cv_accuracy_mean" in tags
        else None
    )

    if (
        min_accuracy is not None
        and current_acc is not None
        and current_acc < min_accuracy
    ):
        print(
            f"  accuracy {current_acc:.4f} < min_accuracy {min_accuracy:.4f},"
            " skipping"
        )
        return

    result = mlflow.register_model(model_uri=model_uri, name=name, tags=tags)

    client = MlflowClient()
    if description:
        client.update_registered_model(name=name, description=description)

    client.set_registered_model_alias(name, "champion", result.version)
    print(f"  alias 'champion' -> version {result.version}")
