# Optional PostgreSQL Analytical Layer

## Purpose

The database layer is an optional serving layer for summaries and metadata. It does
not replace the existing CSV/report pipeline.

Current source artifacts remain:

- `data/processed/`
- `reports/tables/`
- `reports/artifacts/`

Postgres stores compact analytical tables that are easier for API/dashboard
consumption. Full raw EPC certificate loading is intentionally out of scope for
this phase.

## Local Postgres

Start the database:

```bash
docker compose up -d postgres
```

Default development credentials:

```text
POSTGRES_DB=epc
POSTGRES_USER=epc
POSTGRES_PASSWORD=epc
```

Connection URL:

```bash
postgresql+psycopg://epc:epc@localhost:5432/epc
```

## Settings

Environment variables:

- `DATABASE_URL`  
  SQLAlchemy connection URL. If unset, the code uses a safe local SQLite file
  under `reports/artifacts/` for development experiments.

- `ENABLE_DATABASE`  
  Enables DB-backed API reads when set to `true`, `1`, `yes`, `y`, or `on`.

No secrets are committed. Use environment variables or a local `.env` file.

## Create Tables

```bash
DATABASE_URL=postgresql+psycopg://epc:epc@localhost:5432/epc ewhei db-init
```

This uses SQLAlchemy metadata creation. Alembic migrations are planned future
work.

## Load Existing Outputs

Preview what would be loaded:

```bash
ewhei db-load --dry-run
```

Load generated outputs:

```bash
DATABASE_URL=postgresql+psycopg://epc:epc@localhost:5432/epc ewhei db-load
```

The loader reads existing generated outputs. It does not call the EPC API, train
models, or run policy models.

## Tables

### `authority_year_summary`

One row per authority/year summary.

Columns include:

- `authority`
- `year`
- `mean_epc_score`
- `mean_current_energy_efficiency`
- `mean_co2`
- `below_c_rate`
- `certificate_count`
- `gas_share`
- `non_gas_share`
- `new_build_share`
- `created_at`

### `policy_period_summary`

Overall and authority-level policy period summaries.

Columns include:

- `policy_period`
- `authority`
- `mean_epc_score`
- `mean_current_energy_efficiency`
- `mean_co2`
- `below_c_rate`
- `certificate_count`
- `created_at`

### `validation_runs`

Loaded validation report metadata and payload.

Columns include:

- `run_id`
- `status`
- `checked_at`
- `row_count`
- `year_min`
- `year_max`
- `authority_count`
- `error_count`
- `warning_count`
- `payload_json`

### `artifact_registry`

Loaded rows from the JSONL artifact registry.

Columns include:

- `artifact_name`
- `artifact_type`
- `path`
- `sha256`
- `created_at`
- `metadata_json`

### `model_metrics`

Long-form regression/classification metrics loaded from generated model
performance CSVs.

Columns include:

- `task_type`
- `model_name`
- `metric_name`
- `metric_value`
- `split`
- `artifact_path`
- `created_at`
- `metadata_json`

## Query Helpers

The query layer provides:

- `get_authorities(session)`
- `get_trends(session, authority=None, year_min=None, year_max=None)`
- `get_policy_summary(session, authority=None)`
- `get_latest_validation(session)`
- `get_artifacts(session, limit=100)`
- `get_model_metrics(session, task_type=None)`

## API Behaviour

FastAPI uses DB-backed reads only when:

```bash
ENABLE_DATABASE=true
```

and `DATABASE_URL` is reachable. If DB mode is disabled or unavailable, the API
falls back to the existing CSV behaviour.

Useful endpoints:

- `GET /health`
- `GET /metadata`
- `GET /authorities`
- `GET /authority/{authority_name}/summary`
- `GET /trends`
- `GET /policy-summary`
- `GET /validation/latest`
- `GET /artifacts`
- `GET /model/metadata`
- `GET /drift/latest`
- `GET /retraining/status`

See `docs/api.md` for the full API behavior and curl examples.

## Future Work

- Alembic migrations.
- Full warehouse schema for raw/clean/feature data.
- Materialized views for heavier API queries.
- Scheduled DB refresh after Prefect runs.
- DB-backed Streamlit dashboard.
- Deeper model registry metadata after MLflow model promotion is added.
