
# src/kaggle_titanic/tracking.py

from pathlib import Path
from typing import Any, Mapping

import mlflow
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
    сюда пойдут accuracy, roc_auc, f1 и результаты cv
    """

    mlflow.log_metrics(
        dict(metrics),
        step=step,
    )


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
    например oof predictions, test predictions или feature importance
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
