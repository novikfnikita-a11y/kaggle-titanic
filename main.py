from pathlib import Path
from kaggle_titanic.config import load_config
from kaggle_titanic.data import load_data


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    base_config_path = PROJECT_ROOT / "configs" / "base.yaml"
    model_config_path = PROJECT_ROOT / "configs" / "models" / "logreg.yaml"

    config = load_config(base_config_path, model_config_path)
    print("cfg is loaded")

    dataset = load_data(config=config, project_root=PROJECT_ROOT)
    print(
        f"Data loaded: X_train {dataset.x_train.shape}, "
        f"X_test {dataset.x_test.shape}"
    )

    # TODO: add log (mlflow)
    # TODO: add feature engi
    # TODO: add model training and evalu
    # TODO: submission


if __name__ == "__main__":
    main()