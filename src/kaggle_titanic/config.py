from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(base_config_path: str | Path) -> DictConfig:
    '''
    # Результатом будет один объект, содержащий настройки из обоих файлов.

    config = OmegaConf.load("configs/base.yaml")

    # Обращение через точку:
    config.seed
    config.cv.n_splits
    config.cv.shuffle

    # OmegaConf.merge()
    # Объединяет несколько конфигураций.

    config = OmegaConf.merge(base_config, model_config)

    # DictConfig — тип возвращаемого объекта.
    '''



    base_cfg = OmegaConf.load(base_config_path)

    model_config_path = base_cfg.get("model_config")
    if model_config_path is not None:
        model_cfg = OmegaConf.load(
            Path(base_config_path).parent / model_config_path
        )
        base_cfg = OmegaConf.merge(base_cfg, model_cfg)
        OmegaConf.resolve(base_cfg)

    return base_cfg