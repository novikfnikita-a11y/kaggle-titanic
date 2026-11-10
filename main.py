# main.py
from pathlib import Path

import hydra

from omegaconf import DictConfig

import mlflow 

from src.kaggle_titanic.data import load_data


# config_path="conf"      указывает, что конфиги лежат в папке conf/ (рядом с main.py) оч удобно
# config_name="config"    указывает на главный файл конфига conf/config.yaml

@hydra.main(config_path="conf", config_name="config",version_base = '1.3')
def main(cfg: DictConfig) -> None:
    """

    hydra меняет рабочую директорию при запуске создает папку outputs хранит там
    смерженные конфиги по времени запуска

    
    """
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.expiment_name)

    
    project_root = Path(__file__).parent.absolute()
    
    print(f" модель: {cfg.model.name}")
    print(f" гиперпараметры : {cfg.model}")
    print(f" набор фичей: {cfg.features.base}")
    
    try:
        dataset = load_data(cfg, project_root)
        print("\n данные успешно загружены!")
        print(f"  x_train shape: {dataset.x_train.shape}")
        print(f"  x_test shape:  {dataset.x_test.shape}")
        print(f"    y_train shape: {dataset.y_train.shape}")
    except Exception as e:
        print(f"\n ошибка при загрузке данных: {e}")
        raise

if __name__ == "__main__":
    main()