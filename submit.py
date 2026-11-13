"""Generate a Kaggle submission CSV from a registered model or run."""

import re
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf


def _resolve_data_path(path: str, project_root: Path) -> Path:
    if path.startswith("/"):
        return Path(path)
    return project_root / path


def _resolve_model_uri(raw_uri: str, tracking_uri: str) -> str:
    """Resolve models:/name/alias to models:/name/version."""
    m = re.match(r"^models:/([^/]+)/(.+)$", raw_uri)
    if not m:
        return raw_uri

    model_name, alias_or_version = m.group(1), m.group(2)

    if alias_or_version.isdigit():
        return raw_uri

    client = MlflowClient(tracking_uri)
    try:
        v = client.get_model_version_by_alias(model_name, alias_or_version)
        resolved = f"models:/{model_name}/{v.version}"
        print(f"  alias '{alias_or_version}' → version {v.version}")
        return resolved
    except Exception:
        return raw_uri


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.3",
)
def submit(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    model_uri = cfg.get("model_uri")
    run_id = cfg.get("run_id")
    output_path = cfg.get("output_path")

    if model_uri is None and run_id is None:
        print("укажи model_uri или run_id")
        return

    resolved_run_id = run_id

    if model_uri is not None:
        resolved_uri = _resolve_model_uri(model_uri, cfg.mlflow.tracking_uri)
        if resolved_uri != model_uri:
            model_uri = resolved_uri

        m = re.match(r"^models:/([^/]+)/(\d+)$", model_uri)
        if m:
            model_name, version = m.group(1), m.group(2)
            default_output = f"data/submissions/{model_name}_v{version}.csv"
            output_path = output_path or default_output
            resolved_run_id = None
    else:
        model_uri = f"runs:/{run_id}/model"
        if output_path is None:
            client = MlflowClient(cfg.mlflow.tracking_uri)
            run = client.get_run(run_id)
            model_tag = run.data.tags.get("model", run_id[:8])
            output_path = f"data/submissions/{model_tag}_{run_id[:8]}.csv"

    if output_path is None:
        output_path = "data/submissions/submission.csv"

    project_root = Path(__file__).resolve().parent
    data_cfg = cfg.data

    train_path = _resolve_data_path(data_cfg.train_path, project_root)
    test_path = _resolve_data_path(data_cfg.test_path, project_root)

    target = data_cfg.target
    id_column = data_cfg.id_column

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in {target, id_column}]
    x_test = test_df[feature_cols].copy()
    test_ids = test_df[id_column].copy()

    print(f"loading model from {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)

    print("predicting...")
    predictions = pipeline.predict(x_test)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        id_column: test_ids,
        "Survived": predictions,
    })
    submission.to_csv(output, index=False)

    print(f"submission saved to {output.resolve()}")
    print(f"shape: {submission.shape}")


if __name__ == "__main__":
    submit()