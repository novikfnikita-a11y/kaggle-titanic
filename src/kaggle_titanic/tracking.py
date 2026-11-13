
# src/kaggle_titanic/tracking.py

from pathlib import Path
from typing import Any, Mapping

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature, ModelSignature
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf


def flatten_config(cfg: DictConfig) -> dict:
    """
    вложенный конфиг omega становится в плоский словарь для входа млфлоу
    чтобы не онли артефакты трекать, но фильтровать через ui по фичам
    впример {'model': {'C': 1.0}} становится {'model.C': 1.0}
    """

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # py dict на выходе

    result = {}

    def _walk(current_dict: dict, parent_key: str = ""):

        for k, v in current_dict.items():

            if parent_key:
                new_key = f"{parent_key}.{k}"
            else:
                new_key = k

            if isinstance(v, dict):
                _walk(v, new_key)

            elif isinstance(v, list):
                result[new_key] = ",".join(map(str, v))

            else:
                result[new_key] = v

    _walk(cfg_dict)

    return result


def log_config(cfg: DictConfig) -> None:
    """
    сохраняем полный конфиг отдельным json
    чтобы потом можно было посмотреть с какими настройками запускали модель
    """

    resolved_config = OmegaConf.to_container(
        cfg,
        resolve=True,
    )

    mlflow.log_dict(
        resolved_config,
        "config/resolved_config.json",
    )


def log_params(cfg: DictConfig) -> None:
    """
    логируем параметры конфига в млфлоу
    чтобы можно было фильтровать и сравнивать раны через ui
    """

    flat_params = flatten_config(cfg)

    # mlflow ожидает значения параметров в строковом формате
    flat_params = {
        key: str(value) if value is not None else "None"
        for key, value in flat_params.items()
    }

    mlflow.log_params(flat_params)


def log_tags(cfg: DictConfig) -> None:
    """
    теги нужны чтобы удобно искать и группировать эксперименты в ui
    например по модели, seed и группам фичей
    """

    active_feature_groups = [
        group
        for group in cfg.features.column_order
        if group in cfg.features and cfg.features[group] is not None
    ]

    tags = {
        "model": str(cfg.model.name),
        "seed": str(cfg.seed),
        "cv": f"{cfg.cv.splitter}_{cfg.cv.n_splits}",
        "feature_groups": ",".join(active_feature_groups),
    }

    mlflow.set_tags(tags)


def log_selected_features(features: list[str]) -> None:
    """
    логируем выбранные фичи
    чтобы видеть какие признаки участвовали в конкретном ране
    """

    mlflow.log_param(
        "n_selected_features",
        len(features),
    )

    mlflow.log_dict(
        {"selected_features": features},
        "features/selected_features.json",
    )


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """
    логируем метрики эксперимента
    сюда пойдут accuracy и результаты cv
    """

    mlflow.log_metrics(
        dict(metrics),
        step=step,
    )


def log_fold_metrics(fold_metrics: pd.DataFrame) -> None:
    """
    сохраняем результаты каждого фолда в csv
    чтобы потом можно было посмотреть accuracy каждого фолда отдельно
    """

    mlflow.log_text(
        fold_metrics.to_csv(index=False),
        "metrics/fold_metrics.csv",
    )


def log_predictions(predictions: pd.DataFrame) -> None:
    """
    сохраняем предсказания для kaggle test
    чтобы потом можно было использовать их для submission
    """

    mlflow.log_text(
        predictions.to_csv(index=False),
        "predictions/test_predictions.csv",
    )


def log_model(model: Any, X_train_sample: pd.DataFrame | None = None) -> str:
    """
    сохраняем обученный pipeline в млфлоу с input/output схемой
    возвращает model_uri для регистрации в Model Registry
    """

    signature: ModelSignature | None = None
    if X_train_sample is not None:
        predictions = model.predict(X_train_sample)
        signature = infer_signature(X_train_sample, predictions)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        skops_trusted_types=[
            "numpy.dtype",
            "src.kaggle_titanic.features.builder.FeatureGenerator",
            "src.kaggle_titanic.features.builder.FeatureSpec",
            "src.kaggle_titanic.features.preprocessing._log1p_nonnegative",
            "sklearn.tree._tree.Tree",
        ],
    )

    return f"runs:/{mlflow.active_run().info.run_id}/model"


def _get_best_registered_accuracy(model_name: str) -> float | None:
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None

    best = None
    for v in versions:
        acc_str = v.tags.get("cv_accuracy_mean") if v.tags else None
        if acc_str:
            try:
                acc = float(acc_str)
                if best is None or acc > best:
                    best = acc
            except ValueError:
                continue
    return best


def register_model(
    model_uri: str,
    name: str,
    tags: dict | None = None,
    description: str | None = None,
    min_accuracy: float | None = None,
    promote_to_production: bool = False,
) -> None:
    client = MlflowClient()

    best_acc = _get_best_registered_accuracy(name)

    current_acc = float(tags.get("cv_accuracy_mean")) if tags and "cv_accuracy_mean" in tags else None

    if min_accuracy is not None and current_acc is not None and current_acc < min_accuracy:
        print(f"  accuracy {current_acc:.4f} < min_accuracy {min_accuracy:.4f}, skipping")
        return

    if best_acc is not None and current_acc is not None and current_acc < best_acc:
        print(f"  accuracy {current_acc:.4f} < best registered {best_acc:.4f}, skipping")
        return

    try:
        client.create_registered_model(name)
    except Exception:
        pass

    result = mlflow.register_model(
        model_uri=model_uri,
        name=name,
        tags=tags,
    )

    if description:
        client.update_registered_model(
            name=name,
            description=description,
        )

    if promote_to_production and current_acc is not None and (best_acc is None or current_acc >= best_acc):
        version = result.version if hasattr(result, "version") else None
        if version is None:
            versions = client.search_model_versions(f"name='{name}'")
            if versions:
                version = versions[-1].version

        if version:
            try:
                client.delete_registered_model_alias(name, "champion")
            except Exception:
                pass
            client.set_registered_model_alias(name, "champion", version)
            print(f"  set alias 'champion' → version {version}")


def log_dict(data: dict[str, Any], artifact_file: str) -> None:
    """
    сохраняем словарь как json в артефакты млфлоу
    например cv results или дополнительный отчет
    """

    mlflow.log_dict(
        data,
        artifact_file,
    )


def log_artifact_file(
    path: str | Path,
    artifact_path: str | None = None,
) -> None:
    """
    сохраняем файл в артефакты млфлоу
    например feature importance или дополнительный отчет
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact file not found: {path}"
        )

    mlflow.log_artifact(
        str(path),
        artifact_path=artifact_path,
    )
