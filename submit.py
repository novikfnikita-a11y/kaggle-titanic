import re
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from kaggle_titanic.data import load_data
from kaggle_titanic.tracking import connect


def _resolve_model_uri(raw_uri: str, client: MlflowClient) -> str:
    m = re.match(r"^models:/([^/]+)/(.+)$", raw_uri)
    if not m:
        return raw_uri

    model_name, alias_or_version = m.group(1), m.group(2)

    if alias_or_version.isdigit():
        return raw_uri

    try:
        v = client.get_model_version_by_alias(model_name, alias_or_version)
        resolved = f"models:/{model_name}/{v.version}"
        print(f"  alias '{alias_or_version}' → version {v.version}")
        return resolved
    except Exception:
        return raw_uri


def _submission_filename(client: MlflowClient, run_id: str) -> str:
    run = client.get_run(run_id)
    model = run.data.tags.get("model", "model")
    feature_set = run.data.tags.get("feature_set", "features")
    cv = run.data.metrics.get("cv_accuracy_mean")
    cv_part = f"cv{round(cv * 10000)}" if cv is not None else "cvNA"
    return f"{model}_{feature_set}_{cv_part}_{run_id[:8]}.csv"


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.3",
)
def submit(cfg: DictConfig) -> None:

    client = connect(cfg)

    model_uri = cfg.cli.get("model_uri")
    run_id = cfg.cli.get("run_id")
    output_path = cfg.cli.get("output_path")

    if model_uri is None and run_id is None:
        print("укажи model_uri или run_id")
        return

    if model_uri is not None:
        model_uri = _resolve_model_uri(model_uri, client)
        m = re.match(r"^models:/([^/]+)/(\d+)$", model_uri)
        if m:
            run_id = client.get_model_version(m.group(1), m.group(2)).run_id
        else:
            m = re.match(r"^runs:/([^/]+)/", model_uri)
            run_id = m.group(1) if m else None
    else:
        model_uri = f"runs:/{run_id}/model"

    if output_path is None:
        output_path = (
            f"data/submissions/{_submission_filename(client, run_id)}"
            if run_id
            else "data/submissions/submission.csv"
        )

    project_root = Path(__file__).resolve().parent
    dataset = load_data(config=cfg, project_root=project_root)

    print(f"loading model from {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)

    print("predicting...")
    predictions = pipeline.predict(dataset.x_test)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        cfg.data.id_column: dataset.test_ids,
        "Survived": predictions,
    })
    submission.to_csv(output, index=False)

    print(f"submission saved to {output.resolve()}")
    print(f"shape: {submission.shape}")


if __name__ == "__main__":
    submit()
