# EW Housing Energy Impact

Production-style EPC housing-energy analytics platform for South West England:
raw EPC bulk data to validated features, policy analytics, ML models, monitoring,
FastAPI, Streamlit, and reproducible CI.

## 60-Second Pitch

This project turns a dissertation-style EPC analysis into a data/ML platform that
a team could operate and extend. It downloads and cleans UK Domestic EPC data,
rolls messy lower-tier authorities into stable South West analysis groups,
engineers modelling features, runs fixed-effects policy-period analysis, trains
predictive models, tracks artifacts, monitors drift, and serves results through
FastAPI and Streamlit.

What makes it production-style:

- reproducible CLI and Prefect orchestration
- optional PostgreSQL analytical serving tables
- optional MLflow experiment tracking
- validation reports, manifests, artifact registry hashes
- lightweight drift reports with retraining recommendations
- CSV/artifact fallback when optional services are unavailable
- Ruff, pytest, Makefile, and GitHub Actions quality gate

## Problem

EPC data is large, event-based, and geographically awkward. Bulk files are
published at lower-tier/unitary authority level, while the analysis needs stable
South West authority groups. The project answers:

> How have EPC efficiency and carbon outcomes changed across South West England
> during major UK housing-energy policy periods?

The analysis is intentionally honest: policy models are fixed-effects
policy-period models, not definitive causal DiD claims.

## Architecture

```mermaid
flowchart LR
  A[EPC API bulk ZIPs] --> B[Ingestion and merge]
  O[Prefect orchestration] --> B
  B --> C[Cleaning and geography rollup]
  C --> D[Feature engineering]
  D --> E[EDA reports]
  D --> F[Fixed-effects policy models]
  D --> G[Predictive ML models]
  D --> H[Validation, manifest, registry]
  G --> M[Optional MLflow tracking]
  D --> N[Drift monitoring]
  H --> P[(Optional PostgreSQL summaries)]
  P --> Q[FastAPI analytics]
  H --> Q
  N --> Q
  Q --> S[Dashboard/demo consumers]
```

Core CSV/report artifacts remain the source of truth. PostgreSQL and MLflow are
optional local services, not required for tests or basic demos.

## Quickstart

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the quality gate:

```bash
make ci
```

Inspect the orchestrated pipeline without downloading EPC data:

```bash
ewhei orchestrate --dry-run
```

Preview monitoring and database loads without heavy inputs:

```bash
ewhei monitor --dry-run
ewhei db-load --dry-run
```

## Demo Commands Without EPC Credentials

These commands work as smoke/demo commands and do not require EPC API credentials
or full EPC data:

```bash
make smoke
make demo-dry-run
PYTHONPATH=src:. python -m ew_housing_energy_impact --help
PYTHONPATH=src:. python -m ew_housing_energy_impact orchestrate --dry-run
PYTHONPATH=src:. python -m ew_housing_energy_impact monitor --dry-run
PYTHONPATH=src:. python -m ew_housing_energy_impact db-load --dry-run
```

Start the API:

```bash
make api
```

or:

```bash
uvicorn api:app --reload --port 8000
```

The API returns clear empty states for optional artifacts that have not been
generated yet.

## API Examples

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl http://localhost:8000/authorities
curl "http://localhost:8000/authority/Devon/summary"
curl http://localhost:8000/trends
curl http://localhost:8000/policy-summary
curl http://localhost:8000/validation/latest
curl "http://localhost:8000/artifacts?limit=25"
curl http://localhost:8000/model/metadata
curl http://localhost:8000/drift/latest
curl http://localhost:8000/retraining/status
```

See [docs/api.md](docs/api.md) for endpoint behavior, DB vs CSV fallback, and
empty-state examples.

## Full Pipeline Commands

Add EPC API credentials to `.env` only when running real ingestion:

```bash
EPC_EMAIL=you@example.com
EPC_API_KEY=your_key_here
```

Run stages:

```bash
ewhei download --mode la
ewhei clean
ewhei features
ewhei eda
ewhei fe
ewhei train
ewhei validate
ewhei manifest
```

Run everything through the original CLI path:

```bash
ewhei run-all --mode la
```

Run through Prefect:

```bash
ewhei orchestrate --mode la
```

## Optional Services

PostgreSQL analytical layer:

```bash
docker compose up -d postgres
DATABASE_URL=postgresql+psycopg://epc:epc@localhost:5432/epc ewhei db-init
DATABASE_URL=postgresql+psycopg://epc:epc@localhost:5432/epc ewhei db-load
```

MLflow tracking:

```bash
docker compose up -d mlflow
ENABLE_MLFLOW=true \
MLFLOW_TRACKING_URI=http://localhost:5000 \
ewhei train
```

Streamlit dashboard:

```bash
streamlit run app.py
```

## Testing And CI

Local checks:

```bash
make lint
make format-check
make test
make ci
```

CI runs Ruff linting, Ruff format checks, and pytest. Tests use synthetic data,
SQLite, mocks, and dry-runs; they do not download full EPC data or require live
Postgres/MLflow.

## Key Technical Talking Points

- Correct geography: lower-tier/unitary EPC files are downloaded and rolled into
  14 stable South West analysis authorities.
- Leakage-safe ML: train/test split uses `BUILDING_REFERENCE_NUMBER` so repeated
  certificates for the same dwelling do not leak across splits.
- Reproducibility: validation reports, manifests, model cards, and artifact
  registry hashes make outputs auditable.
- Production shape: Prefect, optional Postgres, optional MLflow, FastAPI,
  Streamlit, CI, and monitoring are added without making local development heavy.
- Honest monitoring: drift checks produce a retraining recommendation only; they
  do not retrain or deploy models automatically.

## Limitations

- EPC certificates are lodgement/event records, not a perfect annual stock census.
- Policy-period modelling is observational and should not be sold as conclusive
  causal inference.
- Optional PostgreSQL stores analytical summaries, not the full raw EPC warehouse.
- MLflow tracking is optional; model registry promotion is future work.
- Monitoring is threshold based; richer statistical tests and scheduled alerts are
  future work.
- API authentication, pagination, and prediction endpoints are not implemented yet.

## Resume Bullet Examples

- Built a reproducible Python data/ML platform for UK EPC housing-energy analysis,
  covering ingestion, cleaning, feature engineering, fixed-effects analytics, ML
  training, monitoring, API serving, and CI.
- Added production-style platform layers around an analytics pipeline: Prefect
  orchestration, optional PostgreSQL summaries, optional MLflow tracking,
  FastAPI endpoints, artifact registry, and drift-based retraining
  recommendations.
- Prevented common analytical errors by preserving raw authority identifiers,
  rolling EPC lower-tier geographies into stable analysis groups, and using
  property-level ML splits to reduce leakage.

## More Docs

- [docs/demo_script.md](docs/demo_script.md)
- [docs/resume_bullets.md](docs/resume_bullets.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/api.md](docs/api.md)
- [docs/orchestration.md](docs/orchestration.md)
- [docs/database.md](docs/database.md)
- [docs/mlflow.md](docs/mlflow.md)
- [docs/monitoring.md](docs/monitoring.md)
