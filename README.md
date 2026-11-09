# Kaggle Titanic — reproducible ML workflow

A small project for developing a reproducible scikit-learn workflow with
OmegaConf configuration and MLflow experiment tracking.

## Environment

```bash
uv sync
```

## CLI

Run from any working directory:

```bash
uv run --project /path/to/kaggle-titanic kaggle-titanic --help
uv run --project /path/to/kaggle-titanic kaggle-titanic
```

Or from the project root:

```bash
uv run kaggle-titanic
```

The CLI currently resolves the OmegaConf configuration, loads the Kaggle
train/test files, validates their schemas, and keeps `PassengerId` separate
from model features. Training, cross-validation, MLflow logging, and submission
generation are the next implementation steps.

## Configuration

- `configs/base.yaml`: seed, data paths, CV settings, and feature experiments.
- `configs/models/logreg.yaml`: LogisticRegression settings.

## Data

Place the competition files in `data/raw/`. The data directory is intentionally
git-ignored; do not commit competition data.
