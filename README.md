Reproducible Kaggle Titanic pipeline: scikit-learn Pipeline (features -> encode -> impute -> prep -> clf), Hydra config groups, MLflow tracking. Python 3.12, managed with uv.

To run it, install the dependencies, place the competition files at data/raw/train.csv and data/raw/test.csv (Survived is the target, PassengerId the id column), and start the MLflow UI that conf/config.yaml points at (http://127.0.0.1:5000). Training logs through it, so it must already be running; mlflow ui serves that address backed by sqlite:///mlflow.db with artifacts under ./mlartifacts:

```bash
uv sync
uv run mlflow ui
```

Training runs main.py through Hydra. It composes conf/config.yaml (defaults: logreg, baseline feature_set, age_group_median imputation, frequency encoding, onehot preprocessing), builds the pipeline, runs cross-validation (StratifiedKFold by default; any sklearn splitter via cv.splitter), predicts the test set, and logs everything to the active MLflow run: the resolved config, flat params, tags (model, seed, cv, feature_set, config_hash), selected features, per-fold metrics, test predictions, the byte-for-byte submission CSV, and the fitted model. Feature engineering and the fitted transformers live inside the pipeline, so they are refit on each fold's train split and never leak. Any config group can be overridden on the command line, -m sweeps combinations, and mlflow.enabled=false skips MLflow entirely (no server needed):

```bash
uv run python main.py
uv run python main.py model=rf feature_set=best
uv run python main.py feature_set.use=[base,family,cabin]
uv run python main.py imputation=none encoding=none
uv run python main.py mlflow.enabled=false
uv run python main.py -m feature_set=baseline,best model=logreg,rf
```

Runs are keyed by run_id; the auto run name is {model}_{feature_set}_{config_hash}, where config_hash is a short sha256 of the resolved config without the hydra, mlflow and cli blocks, so an identical hash means an identical experiment. The final model is the fold ensemble by default (soft vote over the five fold models); cv.ensemble=false refits a single model on all data instead.

Hyperparameter search is a separate MLflow run via tune.py; the method is chosen in config (tuning=grid|random) and the grid lives in conf/model/<name>.yaml under search:. It logs the best params and cv_results to MLflow (skipped when mlflow.enabled=false):

```bash
uv run python tune.py model=rf
uv run python tune.py model=rf tuning=random
```

To produce a Kaggle submission, point submit.py at a run or a registered model (models:/titanic-<model>/champion). It loads the model, feeds the raw test rows through the pipeline, and writes data/submissions/{model}_{feature_set}_cv{...}_{run_id}.csv with columns [PassengerId, Survived]:

```bash
uv run python submit.py cli.run_id=<RUN_ID>
uv run python submit.py cli.model_uri=models:/titanic-logisticregression/champion
uv run python submit.py cli.run_id=<RUN_ID> cli.output_path=my_sub.csv
```

After uploading to Kaggle, record the public leaderboard score back into the run:

```bash
uv run python record_lb.py cli.run_id=<RUN_ID> cli.score=0.78472
```

Feature groups are declared once in conf/features.yaml and selected by name from conf/feature_set/*.yaml; models, imputation, encoding, native-vs-onehot categorical handling and the tuning method live in conf/model/*.yaml, conf/imputation/*.yaml, conf/encoding/*.yaml, conf/preprocessing/*.yaml and conf/tuning/*.yaml. data/, outputs/, mlartifacts/ and mlflow.db are gitignored.
