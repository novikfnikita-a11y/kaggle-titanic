import warnings
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from kaggle_titanic.data import load_data
from kaggle_titanic.pipeline import build_from_config
from kaggle_titanic.tracking import (
    config_fingerprint,
    connect,
    log_config,
    log_params,
    log_tags,
)

warnings.filterwarnings("ignore", message=".*register_new_resolver.*")
warnings.filterwarnings("ignore", message=".*Implicit conversion from Version to str.*")


def report(searcher) -> None:
    print("\nbest params:")
    for key, value in searcher.best_params_.items():
        print(f"  {key}: {value}")
    print(f"best cv accuracy: {searcher.best_score_:.4f}")


@hydra.main(
    config_path="conf",
    config_name="tune",
    version_base="1.3",
)
def tune(cfg: DictConfig) -> None:

    project_root = Path(__file__).resolve().parent
    dataset = load_data(config=cfg, project_root=project_root)

    pipeline, _ = build_from_config(cfg)

    splitter = instantiate(cfg.cv.splitter)
    grid = OmegaConf.to_container(cfg.model.search, resolve=True)

    searcher = instantiate(
        cfg.tuning.searcher,
        pipeline,
        grid,
        cv=splitter,
        scoring=cfg.tuning.scoring,
        refit=False,
    )

    print(f"\nsearch: {searcher.__class__.__name__}")
    print(f"grid: {grid}")

    if not cfg.mlflow.enabled:
        searcher.fit(dataset.x_train, dataset.y_train)
        report(searcher)
        print("\nlogging disabled (mlflow.enabled=false)")
        return

    connect(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    fingerprint = config_fingerprint(cfg, ignore_tuning=False)
    run_name = cfg.mlflow.get("run_name") or (
        f"tune_{cfg.model.name}_{cfg.feature_set.name}_{fingerprint}"
    )

    with mlflow.start_run(run_name=run_name):

        log_config(cfg)
        log_params(cfg)
        log_tags(cfg, fingerprint)

        searcher.fit(dataset.x_train, dataset.y_train)

        best_params = dict(searcher.best_params_)
        best_score = float(searcher.best_score_)

        mlflow.log_params(
            {f"best_{key}": str(value) for key, value in best_params.items()}
        )
        mlflow.log_metric("best_cv_accuracy", best_score)
        mlflow.log_dict(
            {"best_params": best_params, "best_score": best_score},
            "tuning/best.json",
        )
        mlflow.log_text(
            pd.DataFrame(searcher.cv_results_).to_csv(index=False),
            "tuning/cv_results.csv",
        )

        report(searcher)


if __name__ == "__main__":
    tune()
