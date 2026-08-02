# Lightweight Drift Monitoring

## Purpose

The monitoring layer compares a reference EPC feature dataset with a current or
latest feature dataset. It produces drift alerts and a retraining recommendation
for human review.

It is intentionally lightweight:

- no Evidently dependency
- no live database requirement
- no live MLflow requirement
- no automatic retraining
- no model deployment or promotion

## What It Checks

Numeric columns:

- reference/current mean
- absolute and relative mean shift
- reference/current standard deviation
- reference/current missing rate
- missingness shift

Categorical columns:

- reference/current top category distributions
- total variation distance
- missingness shift

Targets, when present:

- `CURRENT_ENERGY_EFFICIENCY`
- `CO2_EMISSIONS_CURRENT`
- `BELOW_C`
- `BELOW_C_INT`

Coverage:

- certificate count changes
- year range changes
- authority coverage changes
- `BELOW_C` class-balance shift when available

## Retraining Recommendation

The recommendation is threshold based. It returns:

- `should_retrain`
- `severity`: `low`, `medium`, or `high`
- human-readable reasons
- triggered checks
- timestamp

Example triggers:

- multiple feature drift alerts
- target drift alerts
- `BELOW_C` class-balance movement
- authority coverage changes
- enough new data volume
- optional model performance degradation if metrics are passed in code

This is operational monitoring. It does not prove statistical significance and it
does not claim that model performance has definitely degraded unless performance
metrics are explicitly provided.

## CLI Usage

Preview the default file plan without reading datasets:

```bash
ewhei monitor --dry-run
```

Run with default paths:

```bash
ewhei monitor
```

The default reference and current path is the modelling feature CSV:

```text
data/processed/ew_epc_core_clean_sw/ew_epc_core_clean_sw_imputed.csv
```

For a real drift comparison, pass separate reference and current files:

```bash
ewhei monitor \
  --reference data/reference/features_2024.csv \
  --current data/current/features_2025.csv \
  --output reports/artifacts/drift_report.json
```

Markdown is written beside the JSON output:

```text
reports/artifacts/drift_report.json
reports/artifacts/drift_report.md
```

## Prefect Integration

The Prefect flow runs monitoring after ML training and quality reporting:

```text
train_ml_models -> generate_quality_report -> run_monitoring -> write_manifest
```

Skip it when needed:

```bash
ewhei orchestrate --skip-monitoring
```

The orchestration task uses the default feature dataset and skips cleanly if the
input file is missing. This keeps local dry-runs and CI smoke tests lightweight.

## MLflow Integration

If MLflow is enabled, the monitoring wrapper attempts to log the JSON and Markdown
reports as MLflow artifacts under `monitoring`.

MLflow remains optional:

```bash
ENABLE_MLFLOW=true ewhei monitor
```

If MLflow is disabled, the report is still written locally.

## Report Structure

The JSON report contains:

- `summary`
- `drift.numeric_features`
- `drift.categorical_features`
- `drift.target_drift`
- `drift.coverage`
- `retraining_recommendation`
- `limitations`

The Markdown report is intended for quick human review in interviews, pull
requests, or runbooks.

## Limitations

- The checks are threshold heuristics, not statistical tests.
- The default reference/current files are the same if no paths are supplied, so a
  meaningful production run should pass separate snapshots.
- A retraining recommendation does not retrain, register, promote, or deploy a
  model.
- The current implementation compares CSV feature artifacts only.
- No alert delivery channel is implemented yet.

## Future Work

- scheduled monitoring through Prefect deployments
- richer statistical drift tests
- persisted monitoring history in the analytical database
- MLflow run linkage to model versions
- automatic retraining proposal workflow
- model registry promotion rules with manual approval gates
