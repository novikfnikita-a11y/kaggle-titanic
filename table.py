"""Generate experiment results table from MLflow runs."""

from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.3",
)
def table(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        print(f"Experiment '{cfg.mlflow.experiment_name}' not found")
        return

    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time ASC"],
    )

    rows = []
    for i, run in enumerate(runs, start=1):
        data = run.data
        acc_mean = data.metrics.get("cv_accuracy_mean")
        acc_std = data.metrics.get("cv_accuracy_std")

        model = data.tags.get("model", "")
        groups_tag = data.tags.get("feature_groups", "")
        groups = groups_tag.replace(",", "+") if groups_tag else "basic"
        approach = f"{model} ({groups})" if model else groups

        if acc_mean is None:
            continue

        start_ms = run.info.start_time
        date_str = datetime.fromtimestamp(start_ms / 1000).strftime("%Y-%m-%d") if start_ms else ""

        rows.append({
            "№": len(rows) + 1,
            "Approach": approach,
            "CV, %": f"{acc_mean * 100:.2f}",
            "CV STD, п.п.": f"{acc_std * 100:.2f}" if acc_std is not None else "",
            "LB": "",
            "Date": date_str,
            "run_id": run.info.run_id,
        })

    df = pd.DataFrame(rows)
    df = df.drop(columns=["run_id"])
    df = df.drop_duplicates(subset=["Approach"], keep="last").reset_index(drop=True)
    df["№"] = range(1, len(df) + 1)

    output = Path("data/submissions/experiments_table.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(df.to_string(index=False))
    print(f"\nsaved to {output}")


if __name__ == "__main__":
    table()