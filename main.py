import warnings
from pathlib import Path

import hydra
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.kaggle_titanic.data import load_data
from src.kaggle_titanic.pipeline import build_pipeline, resolve_feature_columns
from src.kaggle_titanic.tracking import (
    config_fingerprint,
    log_config,
    log_fold_metrics,
    log_metrics,
    log_model,
    log_params,
    log_predictions,
    log_selected_features,
    log_submission,
    log_tags,
    register_model,
)
from src.kaggle_titanic.training import train_model

warnings.filterwarnings("ignore", message=".*register_new_resolver.*")
warnings.filterwarnings("ignore", message=".*Implicit conversion from Version to str.*")


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:

    project_root = Path(__file__).resolve().parent

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    fingerprint = config_fingerprint(cfg)
    run_name = (
        cfg.mlflow.get("run_name")
        or f"{cfg.model.name}_{cfg.feature_set.name}_{fingerprint}"
    )

    with mlflow.start_run(run_name=run_name):

        log_config(cfg)
        log_params(cfg)
        log_tags(cfg, fingerprint)

        dataset = load_data(config=cfg, project_root=project_root)

        print(f"\nmodel: {cfg.model.name}")
        print(f"estimator config:\n{OmegaConf.to_yaml(cfg.model.estimator)}")
        print(f"feature set:\n{OmegaConf.to_yaml(cfg.feature_set)}")
        print(f"X_train shape: {dataset.x_train.shape}")
        print(f"y_train shape: {dataset.y_train.shape}")

        estimator = instantiate(cfg.model.estimator)

        imputer_cfg = cfg.imputation.get("imputer")
        imputer = instantiate(imputer_cfg) if imputer_cfg else None
        encoders = [instantiate(encoder) for encoder in cfg.encoding.encoders]

        numeric, categorical, ordinal = resolve_feature_columns(
            cfg.groups, cfg.feature_set
        )

        cat_arg = cfg.model.get("cat_feature_arg")
        if cat_arg:
            estimator.set_params(**{cat_arg: list(categorical) + list(ordinal)})

        pipeline = build_pipeline(
            numeric=numeric,
            categorical=categorical,
            ordinal=ordinal,
            estimator=estimator,
            imputer=imputer,
            encoders=encoders,
            scale=cfg.feature_set.scale,
            native_categorical=cfg.preprocessing.get("native_categorical", False),
        )

        selected_features = numeric + categorical + list(ordinal)
        log_selected_features(selected_features)

        print(f"\nselected features ({len(selected_features)}):")
        for feature in selected_features:
            print(f"   {feature}")

        print("\npipeline steps:")
        for step_name, step in pipeline.steps:
            print(f"  {step_name}: {step.__class__.__name__}")

        result = train_model(
            pipeline=pipeline,
            dataset=dataset,
            cv_config=cfg.cv,
            id_column=cfg.data.id_column,
        )

        log_metrics(result.metrics)
        log_fold_metrics(result.fold_metrics)
        log_predictions(result.test_predictions)
        log_submission(result.test_predictions)

        model_uri = log_model(
            result.model,
            X_train_sample=dataset.x_train.head(5),
        )

        if cfg.mlflow.register_model:
            register_model(
                model_uri=model_uri,
                name=f"titanic-{cfg.model.name.lower()}",
                tags={"cv_accuracy_mean": str(result.metrics["cv_accuracy_mean"])},
                description=(
                    f"model={cfg.model.name} | feature_set={cfg.feature_set.name} | "
                    f"seed={cfg.seed} | cv={cfg.cv.n_splits}fold"
                ),
                min_accuracy=cfg.mlflow.register_if_better_than,
            )


if __name__ == "__main__":
    main()
