# EW Housing Energy Impact

A reproducible data engineering + analytics + ML platform that turns 1.8M+ raw
UK Energy Performance Certificate (EPC) records into policy analysis, predictive
models, and served APIs — built as a dissertation project and engineered like a
production system.

**Research question:** How has domestic energy efficiency in South West England
changed across five UK housing-energy policy periods (2008–2025)?

```bash
pip install -r requirements.txt && pip install -e .
make ci              # lint + format + 129 tests, no data or credentials required
ewhei orchestrate --dry-run   # inspect the 11-stage Prefect plan
```

---

## Why this project is worth a look

This isn't a notebook that computes a p-value once. It's a pipeline that has to
keep working: it re-ingests monthly-changing government data, survives schema
drift, splits ML data without leaking, does inference with too few clusters for
textbook standard errors, and degrades cleanly when optional infrastructure
(Postgres, MLflow) isn't running. That combination — real statistics, real data
engineering, real ML hygiene — is the point.

| Area | What's demonstrated |
|---|---|
| **Data engineering** | Chunked ingestion of multi-GB government CSVs, geography normalization, S3 → Snowflake warehouse loading with reconciliation and schema-diff guards |
| **Applied statistics** | Fixed-effects panel OLS, wild cluster bootstrap inference (14 clusters — too few for asymptotic clustered SEs), event-study design |
| **ML engineering** | XGBoost + SHAP, property-level (not row-level) train/test split to prevent leakage, PCA + KMeans on model effects |
| **Platform engineering** | Prefect orchestration, optional Postgres/MLflow with enforced fallback, FastAPI + Streamlit serving layers, Docker Compose |
| **Software engineering rigor** | 129 tests (all offline — synthetic data, mocked S3/Snowflake/HTTP), Ruff-enforced style, GitHub Actions CI, artifact registry with SHA256 provenance |

---

## Architecture

```mermaid
flowchart LR
  A[GOV.UK EPC bulk API] -->|Bearer auth| B[Download + extract]
  B --> C[S3 raw archive]
  C --> D[Snowflake raw layer]
  B --> E[Clean: SW filter + geography rollup]
  E --> F[Feature engineering]
  F --> G[EDA]
  F --> H[Fixed-effects models + WCB + event study]
  F --> I[XGBoost + SHAP]
  H --> J[Validation, manifest, artifact registry]
  I --> J
  I -.-> K[(Optional MLflow tracking)]
  J -.-> L[(Optional PostgreSQL)]
  J --> M[FastAPI]
  L --> M
  F --> N[Drift monitoring]
  N --> M
  O[Prefect orchestration] -.-> B
  O -.-> E
  O -.-> F
  O -.-> H
  O -.-> I
  M --> P[Streamlit dashboard]
```

CSV/report artifacts are the source of truth. Postgres, MLflow, and the
S3/Snowflake warehouse path are optional layers — the pipeline, tests, and API
all run without them via CSV fallback.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language / packaging | Python 3.11+, `setuptools` editable install, `ewhei` console entry point |
| Data processing | pandas (chunked, 1M-row batches), NumPy, PyArrow |
| Statistics | `linearmodels.PanelOLS` (fixed effects), `statsmodels`, custom wild cluster bootstrap (999 Rademacher draws) |
| ML | XGBoost, scikit-learn, SHAP, PCA + KMeans |
| Orchestration | Prefect 2 (11-task flow, dry-run planning, per-stage skip flags) |
| Warehouse ingestion | boto3 (S3), `snowflake-connector-python`, reconciliation + header-diff guards |
| Serving | FastAPI (11 routes), Streamlit dashboard |
| Storage (optional) | PostgreSQL via SQLAlchemy 2.0, SQLite fallback |
| Experiment tracking (optional) | MLflow |
| Quality gates | Ruff (lint + format), pytest (129 tests), GitHub Actions CI |
| Infra | Docker Compose (Postgres, MLflow, Jupyter, FastAPI, Streamlit — 5 services) |

---

## Hard problems this pipeline actually solves

- **Too few clusters for standard inference.** With only 14 South West local
  authorities, conventional clustered standard errors are unreliable (Cameron,
  Gelbach & Miller, 2008). `run_did.py` runs a wild cluster bootstrap (B=999)
  alongside standard clustered SEs for every policy-period coefficient.
- **Geography that doesn't match the data source.** EPC bulk files are published
  per lower-tier/unitary authority, not per stable analysis authority. The
  pipeline rolls historical/lower-tier authorities into 14 consistent South West
  groups while preserving the raw authority label for traceability
  (`LOCAL_AUTHORITY_RAW`).
- **ML leakage from repeat inspections.** The same property can have multiple EPC
  certificates over time. Train/test splitting is done on
  `BUILDING_REFERENCE_NUMBER` (property level), not row level, so no dwelling
  appears in both sets.
- **A government API that changed its data model mid-project.** The EPC service
  moved from per-authority zips + Basic auth to one national archive per period +
  Bearer auth. The ingestion layer absorbs this without touching downstream code.
- **Loading a warehouse from a source with no reliable typing.** Raw EPC exports
  contain `'NO DATA!'`, `'INVALID!'`, and inconsistent dates. The Snowflake raw
  layer loads everything as STRING and preserves sentinels rather than coercing
  them to NULL, so the distinction between "surveyor recorded nothing" and "field
  absent" isn't destroyed at load time. Loads are verified with row-count
  reconciliation, not by trusting a zero exit code.
- **Confounded time trends.** New-build share per LA-year is included as a
  covariate because high-construction years mechanically raise mean EPC scores
  — without it, policy effects would be conflated with compositional shift.

---

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .          # installs the `ewhei` CLI
make ci                   # lint + format-check + 129 tests
```

Explore the pipeline without downloading any data or holding credentials:

```bash
ewhei orchestrate --dry-run     # print the 11-task Prefect plan
ewhei monitor --dry-run         # planned drift-report inputs/outputs
ewhei db-load --dry-run         # planned analytical table loads
ewhei ingest --dry-run          # planned S3 keys and expected row counts
```

Run the API against generated artifacts:

```bash
make api                        # uvicorn api:app --reload --port 8000
curl http://localhost:8000/health
curl http://localhost:8000/policy-summary
curl http://localhost:8000/model/metadata
```

The full command list: `make smoke`, `make test`, `make lint`, `make format`,
`make validate`.

---

## Running the full pipeline

Requires an `EPC_BEARER_TOKEN` in `.env` (free registration at
`get-energy-performance-data.communities.gov.uk`):

```bash
ewhei download           # one national archive per period, ~2.9 GB compressed
ewhei clean               # SW filter, geography rollup, standardise fields
ewhei features             # impute, policy periods, new-build share, log floor area
ewhei eda                  # EDA figures and tables
ewhei fe                    # fixed-effects models, wild cluster bootstrap, event study
ewhei train                  # XGBoost + SHAP
ewhei validate                # schema/range checks
ewhei manifest                 # dataset manifest
```

Or run everything through Prefect: `ewhei orchestrate`.

Warehouse path (S3 → Snowflake, needs `EPC_S3_BUCKET` and Snowflake credentials):

```bash
ewhei ingest --dry-run
ewhei ingest
```

## Optional services

```bash
# PostgreSQL analytical layer
docker compose up -d postgres
DATABASE_URL=postgresql+psycopg://epc:epc@localhost:5432/epc ewhei db-init
DATABASE_URL=postgresql+psycopg://epc:epc@localhost:5432/epc ewhei db-load

# MLflow experiment tracking
docker compose up -d mlflow
ENABLE_MLFLOW=true MLFLOW_TRACKING_URI=http://localhost:5000 ewhei train

# Streamlit dashboard
streamlit run app.py
```

Every optional integration degrades to a CSV/artifact fallback rather than
raising — this is enforced by tests, not just a design intention.

---

## Testing & CI

```bash
make ci     # ruff check . && ruff format --check . && pytest -q
```

129 tests, all offline: synthetic fixtures, SQLite in place of Postgres, mocked
`boto3`/`snowflake.connector`/HTTP for the ingestion path. GitHub Actions runs
the same gate on every push/PR to `main` (Python 3.11).

| Coverage area | File |
|---|---|
| Policy-period boundary logic | `test_policy.py` |
| Schema/range validation | `test_validation.py` |
| Local authority → analysis group rollup | `test_geography.py` |
| DB models, loaders, queries | `test_db.py` |
| FastAPI routes incl. DB-unavailable fallback | `test_api_database.py` |
| MLflow no-op-when-disabled helpers | `test_mlflow_tracking.py` |
| Drift metrics, retraining recommendation | `test_monitoring.py` |
| Prefect task/flow wiring | `test_orchestration.py` |
| Bearer auth, zip validation, period parsing | `test_epc_source.py` |
| S3 key construction, MD5 skip-unchanged | `test_s3_upload.py` |
| Snowflake DDL, COPY INTO, row reconciliation | `test_load.py` |
| End-to-end ingestion wiring (fake clients) | `test_ingestion_pipeline.py` |

A green test suite doesn't alone prove `ewhei` works as an installed CLI — see
`make smoke` for that check, run separately in CI.

---

## Repository map

```
config.py                     # paths, geography maps, band/fuel/tenure lookups
scripts/pipeline/              # download, clean, features, EDA, FE models, ML, quality
src/ew_housing_energy_impact/
  cli.py                        # ewhei entry point
  ingestion/                     # EPC source, S3 upload, Snowflake load, pipeline wiring
  orchestration/                  # Prefect flow + task wrappers
  db/                               # SQLAlchemy models, loaders, queries
  ml/tracking.py                    # MLflow wrappers, no-op when disabled
  monitoring/                        # drift detection, retraining recommendation
api.py                          # FastAPI, 11 routes, DB-with-CSV fallback
app.py                          # Streamlit dashboard
snowflake/                      # one-time warehouse + IAM setup SQL
tests/                          # 129 tests, fully offline
```

~6,200 lines of pipeline/application Python across `src/` and `scripts/`.

---

## Honest limitations

Stated up front because pretending otherwise would undercut the rest of this:

- EPC certificates are lodgement/event records, not a complete annual stock
  census — policy-period comparisons are observational, not a definitive causal
  DiD claim.
- PostgreSQL stores analytical summaries, not the full raw EPC warehouse; that's
  what the Snowflake raw layer is for.
- MLflow model registry promotion isn't implemented — tracking only.
- Drift monitoring is threshold-based; it recommends retraining, it doesn't
  trigger it.
- The API has no authentication, no pagination, and no live prediction endpoint
  yet.

## Further documentation

- [docs/architecture.md](docs/architecture.md) — design decisions and rationale
- [docs/ingestion.md](docs/ingestion.md) — S3/Snowflake warehouse path, failure modes
- [docs/api.md](docs/api.md) — endpoint behavior, DB vs CSV fallback
- [docs/orchestration.md](docs/orchestration.md) · [docs/database.md](docs/database.md) · [docs/mlflow.md](docs/mlflow.md) · [docs/monitoring.md](docs/monitoring.md)
- [docs/interview_guide.md](docs/interview_guide.md) — talking points and hardest technical challenges
