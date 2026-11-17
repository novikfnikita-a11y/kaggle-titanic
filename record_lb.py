import hydra
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.3",
)
def record_lb(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    run_id = cfg.cli.get("run_id")
    score = cfg.cli.get("score")
    source = cfg.cli.get("source", "public")

    if run_id is None or score is None:
        print("укажи run_id и score")
        return

    client = MlflowClient(cfg.mlflow.tracking_uri)
    client.log_metric(run_id, f"lb_{source}", float(score))
    client.set_tag(run_id, "lb_source", str(source))

    run = client.get_run(run_id)
    model = run.data.tags.get("model", "?")
    feature_set = run.data.tags.get("feature_set", "?")
    cv = run.data.metrics.get("cv_accuracy_mean")
    cv_str = f"{cv:.4f}" if cv is not None else "?"
    print(
        f"run {run_id[:8]} | {model}_{feature_set} | cv={cv_str} "
        f"-> lb_{source}={float(score):.5f}"
    )


if __name__ == "__main__":
    record_lb()
