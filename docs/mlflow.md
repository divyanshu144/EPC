# Optional MLflow Experiment Tracking

## Purpose

MLflow tracking is optional. The existing training script still writes local CSVs,
figures, model cards, registry entries, and `.joblib` artifacts when MLflow is
disabled.

When enabled, MLflow adds experiment tracking for model comparison and
reproducibility without changing the model choices or core training logic.

## Settings

Environment variables:

- `ENABLE_MLFLOW`  
  Disabled by default. Set to `true`, `1`, `yes`, `y`, or `on` to enable tracking.

- `MLFLOW_TRACKING_URI`  
  Tracking server or local file URI. Defaults to a local file-backed `mlruns`
  directory.

- `MLFLOW_EXPERIMENT_NAME`  
  Defaults to `epc-housing-energy`.

No credentials or secrets are logged by the tracking helper.

## Run Without MLflow

Default training remains unchanged:

```bash
ewhei train
```

This does not require an MLflow server.

## Run With Local File-Backed MLflow

```bash
ENABLE_MLFLOW=true ewhei train
```

or:

```bash
ewhei train --track
```

## Run With The Docker MLflow Server

Start MLflow:

```bash
docker compose up -d mlflow
```

Run training with tracking:

```bash
ENABLE_MLFLOW=true \
MLFLOW_TRACKING_URI=http://localhost:5000 \
MLFLOW_EXPERIMENT_NAME=epc-housing-energy \
ewhei train
```

Open the UI:

```text
http://localhost:5000
```

## What Gets Logged

For each regression/classification row in the generated model performance tables,
the tracking layer logs:

- task type: `regression` or `classification`
- model name
- feature list
- target name
- split strategy, including property-level split by `BUILDING_REFERENCE_NUMBER`
- random seed
- train/test row counts
- model hyperparameters where safely serializable
- regression metrics: RMSE, MAE, R2 where available
- classification metrics: accuracy, precision, recall, F1, ROC-AUC where available
- selected local artifacts such as metrics CSVs, feature importance tables/plots,
  SHAP plot, model cards, and `.joblib` files
- model objects where MLflow's scikit-learn logging supports them

## Design Notes

- MLflow imports are lazy.
- If `ENABLE_MLFLOW=false`, helpers are no-ops.
- If tracking is explicitly enabled but MLflow is unavailable, the helper raises a
  clear error.
- The tracking layer does not retrain models or alter model outputs.
- CI tests use mocks and do not require a live MLflow server.

## Limitations

- No model registry promotion workflow yet.
- No champion/challenger deployment policy yet.
- No automatic drift-triggered retraining yet.
- The Docker MLflow service uses a local file-backed artifact store for development.

## Future Work

- Register best models in MLflow Model Registry.
- Add model stage promotion rules.
- Link Prefect run metadata to MLflow runs.
- Add drift and retraining decisions as tracked artifacts.
