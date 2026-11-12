
from pathlib import Path
import warnings

import hydra
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.kaggle_titanic.data import load_data
from src.kaggle_titanic.features import build_feature_pipeline
from src.kaggle_titanic.tracking import (
    log_config,
    log_params,
    log_tags,
    log_selected_features,
)

warnings.filterwarnings(
    "ignore",
    message=".*register_new_resolver.*",
)

warnings.filterwarnings(
    "ignore",
    message=".*Implicit conversion from Version to str.*",
)



@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:

    project_root = Path(__file__).resolve().parent

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = str(cfg.mlflow.run_name)

    with mlflow.start_run(run_name=run_name):

        print(f"начало рана: {run_name}")

        log_config(cfg)
        log_params(cfg)
        log_tags(cfg)

        print(f"\n модель: {cfg.model.name}")
        print(f"estimator config:\n{OmegaConf.to_yaml(cfg.model.estimator)}")

        print(
            "\nfeature cfg:\n"
            f"{OmegaConf.to_yaml(cfg.features)}"
        )

        # data load

        try:
            dataset = load_data(
                config=cfg,
                project_root=project_root,
            )

            print("\n данные на месте")
            print(f"X_train shape: {dataset.x_train.shape}")
            print(f"X_test shape:  {dataset.x_test.shape}")
            print(f"y_train shape: {dataset.y_train.shape}")

            estimator = instantiate(cfg.model.estimator)

            pipeline = build_feature_pipeline(
                features_config=cfg.features,
                estimator=estimator,
            )

            print("\nfeature pipeline done")

            feature_generator = pipeline.named_steps["feature_generator"]

            selected_features = [
                spec.name
                for spec in feature_generator.feature_specs
            ]

            log_selected_features(selected_features)

            print(f"selected features ({len(selected_features)}):")
            for feature in selected_features:
                print(f"   {feature}")

            print("\n structure:")
            for step_name, step in pipeline.steps:
                print(f"  {step_name}: {step.__class__.__name__}")

            print("\n cv continued xd.")

        except Exception as e:
            print(f"experiment failed: {e}")
            raise


if __name__ == "__main__":
    main()
