# Prefect Orchestration

## Flow Overview

The project uses Prefect as a lightweight orchestration layer for local and
resume-friendly workflow execution. The flow wraps existing pipeline scripts
instead of rewriting their business logic.

Entry point:

```bash
ewhei orchestrate
```

The implementation lives in:

- `src/ew_housing_energy_impact/orchestration/flows.py`
- `src/ew_housing_energy_impact/orchestration/tasks.py`

## Task Dependency Order

```mermaid
flowchart LR
  A[download_or_merge_epc] --> B[clean_south_west]
  B --> C[build_features]
  C --> D[validate_clean_dataset]
  D --> E[generate_eda]
  E --> F[run_policy_models]
  F --> G[train_ml_models]
  G --> H[generate_quality_report]
  H --> I[run_monitoring]
  I --> J[write_manifest]
  J --> K[register_artifacts]
```

## Task List

- `download_or_merge_epc`  
  Calls the existing EPC download/extract/merge script.

- `clean_south_west`  
  Calls the existing South West cleaning and geography rollup script.

- `build_features`  
  Calls the existing feature engineering script.

- `validate_clean_dataset`  
  Runs the existing validation utilities and raises on validation failure.

- `generate_eda`  
  Calls the existing EDA report script.

- `run_policy_models`  
  Calls the existing fixed-effects policy-period modelling script.

- `train_ml_models`  
  Calls the existing predictive modelling script.

- `generate_quality_report`  
  Calls the existing HTML quality report script.

- `run_monitoring`  
  Runs lightweight drift monitoring and writes JSON/Markdown retraining
  recommendation reports when input feature files are available.

- `write_manifest`  
  Writes the dataset manifest using the existing manifest utility.

- `register_artifacts`  
  Writes a lightweight orchestration completion marker to the artifact registry.

## Parameters

`ewhei orchestrate` supports:

- `--mode la` or `--mode year`  
  Selects local-authority ZIPs or full annual bulk downloads.

- `--skip-download`  
  Uses existing merged data and avoids EPC API calls.

- `--skip-eda`  
  Skips EDA figures/tables.

- `--skip-policy`  
  Skips fixed-effects policy-period modelling.

- `--skip-training`  
  Skips predictive ML training.

- `--skip-quality`  
  Skips HTML quality report generation.

- `--skip-monitoring`  
  Skips drift monitoring reports.

- `--dry-run`  
  Prints the planned task order without executing pipeline logic.

## Local Run Commands

Inspect the plan:

```bash
ewhei orchestrate --dry-run
```

Run from existing local data without downloading:

```bash
ewhei orchestrate --skip-download
```

Run a lighter local workflow without expensive modelling:

```bash
ewhei orchestrate --skip-download --skip-policy --skip-training
```

Run without monitoring:

```bash
ewhei orchestrate --skip-monitoring
```

Run the full local-authority mode pipeline:

```bash
ewhei orchestrate --mode la
```

## Failure Behaviour

Validation is a hard gate. If `validate_clean_dataset` fails, the task raises an
exception and downstream reporting/modelling tasks do not run.

This is intentional: EDA, policy models, ML training, quality reports,
monitoring, manifests, and the final orchestration marker should not be produced
from a dataset that failed required checks.

## CI Behaviour

CI does not download EPC data and does not run the full pipeline. It validates:

- flow imports
- dry-run execution
- expected task names
- CLI help and skip flag parsing
- monitoring dry-run and synthetic drift checks
- Airflow placeholder clarity
- existing geography, policy, and validation tests

## Limitations

- This is local Prefect orchestration, not a remote Prefect deployment.
- No schedule, work pool, retry policy, or server-side state backend is configured
  yet.
- The flow still uses file-based CSV/artifact outputs.
- Heavy stages can still be slow because they call the existing modelling scripts.
- Monitoring is threshold based and only recommends retraining; it does not
  retrain automatically.
- PostgreSQL and MLflow remain optional platform integrations.

## Future Production Deployment Options

- Add Prefect work pools and scheduled deployments.
- Configure retries and timeouts per task.
- Store task outputs and run metadata in a remote Prefect backend.
- Add PostgreSQL-backed analytical tables as downstream task outputs.
- Add MLflow tracking for model training tasks.
- Add scheduled monitoring and conditional retraining proposal tasks.
