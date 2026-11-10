# main.py
from pathlib import Path
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow 


from src.kaggle_titanic.data import load_data

from src.kaggle_titanic.tracking import flatten_config



warnings.filterwarnings("ignore", message=".*register_new_resolver.*")
warnings.filterwarnings("ignore", message=".*Implicit conversion from Version to str.*")


@hydra.main(config_path="conf", config_name="config",version_base = '1.3')
def main(cfg: DictConfig) -> None:
    """
    config_path="conf"      указывает, что конфиги лежат в папке conf/ (рядом с main.py) 
    оч удобно
    config_name="config"    указывает на главный файл конфига conf/config.yaml
    
    hydra меняет рабочую директорию при запуске создает папку outputs хранит там
    смерженные конфиги по времени запуска

    
    """
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = f"{cfg.model.name}_seed{cfg.seed}"


    with mlflow.start_run(run_name=run_name):
        print(f"млфлоу ран {run_name} запущен ")
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")
        
        
        flat_params = flatten_config(cfg)
        
        mlflow.log_params(flat_params)

        project_root = Path(__file__).parent.absolute()
    
        print(f"\n модель: {cfg.model.name}")
        print(f"  гиперпараметры : {cfg.model}")
        print(f" набор фичей: {cfg.features.base}")
    
    
    
        try:
            dataset = load_data(cfg, project_root)
            print("\n данные загружены")
            print(f"  x_train shape: {dataset.x_train.shape}")
            print(f"  x_test shape:  {dataset.x_test.shape}")
            print(f"    y_train shape: {dataset.y_train.shape}")
        except Exception as e:
            print(f"\n ошибка при загрузке данных: {e}")
            mlflow.end_run(status="FAILED")
            raise

if __name__ == "__main__":
    main()