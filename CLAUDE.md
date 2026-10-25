# CLAUDE.md — EPC South West Analysis Pipeline

## Project overview

Reproducible academic pipeline for the dissertation:
**"The Impact of Energy Policy on Housing Energy Performance in the South West of England (2008–2025)"**

Analyses a versioned release of 2,776,781 domestic EPC certificates (2008–2025, built from 30,153,553
national records) across 14 South West local authorities. The older "~1.8 M" figure was never
substantiated — do not reuse it. Methods:
- Fixed-effects panel OLS (`linearmodels.PanelOLS`, indexed on `[LOCAL_AUTHORITY, YEAR]`)
- Wild cluster bootstrap for inference (14 clusters — too few for standard asymptotic clustered SEs)
- Year-by-year event study alongside five blunt policy-period dummies
- Property-grouped benchmark (`ewhei train`): tuned MLP / GBR regression and XGBoost classification vs
  linear/logistic/dummy baselines, with MLflow lineage, bootstrap CIs, SHAP and model cards
- PCA + KMeans clustering of LA random effects

The package is installed as `ew-housing-energy-impact` (editable). The CLI entry point is `ewhei`.

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # installs the ewhei CLI
```

Create a `.env` file in the project root:
```
EPC_BEARER_TOKEN=your_token_here
```

`EPC_EMAIL`/`EPC_API_KEY` are **obsolete** — the EPC service moved to GOV.UK in 2025 and auth
changed from Basic to Bearer. Obtain the token from the account page on
`get-energy-performance-data.communities.gov.uk` (free registration); there is no
programmatic way to mint one. See `tasks/lessons.md` (2026-08-11 entry) for why this changed.

For the warehouse ingestion path (`ewhei ingest`), also see `docs/ingestion.md` for
`EPC_S3_BUCKET` and the Snowflake env vars.

---

## Running the pipeline

### Full pipeline (one command)
```bash
ewhei run-all
```

### Step by step (recommended during development)
```bash
ewhei download          # One national archive per period (~2.92 GB compressed) — see Download modes
ewhei clean             # Filter to SW LAs, standardise fields, derive YEAR
ewhei features          # Impute, add LOG_FLOOR_AREA, POLICY_PERIOD, NEW_BUILD_SHARE
ewhei eda               # EDA figures and tables -> reports/
ewhei fe                # Fixed-effects models + event study + clustering -> reports/artifacts/
ewhei train --release <dir> --output <new-run-dir>   # versioned property/chronological benchmark
ewhei quality           # HTML data quality report -> reports/artifacts/
ewhei validate          # Schema / range checks -> reports/artifacts/validation_report.json
ewhei manifest          # Data manifest -> reports/artifacts/data_manifest.json
```

### Platform commands (optional subsystems)
```bash
ewhei orchestrate --dry-run   # Print the 11-task Prefect plan without running it
ewhei orchestrate             # Run the full pipeline under Prefect
ewhei monitor --dry-run       # Show planned drift-report inputs/outputs
ewhei monitor                 # Drift report -> reports/artifacts/drift_report.{json,md}
ewhei db-init                 # Create analytical tables (SQLite default)
ewhei db-load --dry-run       # Print planned table loads
ewhei mlflow-info             # Show MLflow config and whether tracking is enabled
ewhei legacy-train --track    # Historical path (train_models.py); not for new benchmark claims
ewhei ingest --dry-run        # Planned S3 keys and expected row counts, no network calls
ewhei ingest                  # Upload extracted EPC sources to S3, load Snowflake raw, reconcile
```

`ewhei ingest` is the S3 -> Snowflake warehouse ingestion path (Phase A of the warehouse
migration — see `docs/ingestion.md`). It reads whatever `ewhei download` + `ewhei clean`
already extracted under `RAW_EXTRACT_DIR`; it does not download anything itself. Needs
`EPC_S3_BUCKET` and `SNOWFLAKE_ACCOUNT`/`SNOWFLAKE_USER`/`SNOWFLAKE_PASSWORD` for the
non-dry-run path.

All `orchestrate` steps can be skipped individually: `--skip-download`, `--skip-eda`,
`--skip-policy`, `--skip-training`, `--skip-quality`, `--skip-monitoring`.

### Make shortcuts
```bash
make install       # pip install -r requirements.txt && pip install -e .
make ci            # lint + format-check + test — run this before claiming done
make lint          # ruff check .
make format        # ruff format .
make format-check  # ruff format --check .
make test          # python -m pytest -q
make smoke         # CLI --help + all three --dry-run paths
make validate      # python -m ew_housing_energy_impact validate
```

---

## Data paths (defined in `config.py`)

| Variable | Path |
|---|---|
| `RAW_BULK_DIR` | `data/raw/ew_bulk/` — downloaded `.zip` files |
| `RAW_EXTRACT_DIR` | `data/raw/ew_extracted/` — unzipped certificate folders |
| `EPC_MERGED_CSV` | `data/processed/ew_epc_core.csv` — merged national CSV |
| `EPC_CLEAN_SW_DIR` | `data/processed/ew_epc_core_clean_sw/` — SW-filtered outputs |
| `ARTIFACTS_DIR` | `reports/artifacts/` — tables, figures, registry |
| `REGISTRY_PATH` | `reports/artifacts/registry.jsonl` — artifact provenance log |

Key processed files:
- `ew_epc_core_clean_sw.csv` — cleaned SW records
- `ew_epc_core_clean_sw_imputed.csv` — imputed + feature-engineered, used by all models

**Always import `ARTIFACTS_DIR` and `REGISTRY_PATH` from `config.py`. Never redefine them inline.**

---

## Code map

```
config.py                              # All shared paths and lookup maps
scripts/pipeline/
  schemas.py                           # CORE_COLS, CORE_DTYPES, CLEAN_KEEP_COLS
  policy.py                            # policy_period(), POLICY_ORDER, _BOUNDARIES
  epc_download_merge_fast.py           # National-archive download + merge (LA mode retired — see Download modes)
  clean_epc.py                         # SW filter, standardise, derive YEAR
  build_features.py                    # Impute, LOG_FLOOR_AREA, NEW_BUILD_SHARE, POLICY_PERIOD
  report_eda.py                        # EDA figures and summary tables
  run_did.py                           # FE models, WCB, event study, clustering
  train_models.py                      # LEGACY XGBoost + SHAP + PCA clustering (ewhei legacy-train)
  build_release.py                     # Versioned release: source hashes, dedup, reconciled exclusions
  verify_release.py                    # Independent release verifier
  benchmark_release.py                 # Property/chronological benchmark, MLflow, bootstrap (ewhei train)
  verify_benchmark.py                  # Independent re-check: reloads models, recomputes metrics
  report_benchmark.py                  # Model cards, permutation importance, SHAP
  compare_benchmarks.py                # Reproduction check, fixed rtol=atol=1e-8, never auto-loosened
  fast_wcb.py                          # UNTRACKED/untested draft; no measured speedup — do not quote
  quality_report.py                    # HTML data quality report
src/ew_housing_energy_impact/
  cli.py                               # ewhei CLI (argparse, delegates to scripts)
  settings.py                          # Env-driven Settings for optional services
  validation.py                        # Schema/range validation
  manifest.py                          # Data manifest writer
  registry.py                          # Artifact registry (JSONL append)
  logging_utils.py                     # Structured logger setup
  paths.py                             # repo_root() helper
  ingestion/
    epc_source.py                      # Bearer auth, zip download/validate/extract, period parsing
    s3_upload.py                       # Period-keyed S3 uploads, skip-unchanged by stored MD5
    load.py                            # Raw DDL, COPY INTO, quote-aware row counts, header diff, reconcile
    pipeline.py                        # run_ingest() — wires source enumeration, upload, load, reconciliation, authority coverage
  orchestration/
    flows.py                           # epc_pipeline_flow() — Prefect flow
    tasks.py                           # 11 @task wrappers around pipeline steps
  db/
    engine.py                          # get_engine(), create_tables()
    models.py                          # 5 SQLAlchemy tables (DeclarativeBase)
    loaders.py                         # refresh_analytics_database() + per-table loaders
    queries.py                         # Read helpers used by api.py
  ml/
    evaluation.py                      # Property-separated splits, grouped folds, preprocessing
    serving.py                         # FastAPI prediction service for verified runs
    tracking.py                        # MLflow wrappers — all *_safe, no-op when disabled
  monitoring/
    drift.py                           # build_drift_report(), TVD + numeric/categorical drift
    reporting.py                       # run_monitoring(), JSON + Markdown rendering
    retraining.py                      # recommend_retraining()
api.py                                 # FastAPI, 11 routes, DB-with-CSV fallback
app.py                                 # Streamlit dashboard
dags/etl_epc_pipeline.py               # Tombstone — Airflow is NOT implemented, Prefect is
snowflake/                             # One-time Snowflake + IAM setup SQL, run in numeric order
  01_setup.sql                         # Warehouse, database, schemas, role
  02_storage_integration.sql           # Storage integration; DESCRIBE output feeds the IAM trust policy
  03_stage_and_format.sql              # External stage + CSV file format
  README.md                            # Order of operations, IAM policy/trust JSON, why the trust is bidirectional
```

### Optional subsystems

The database and MLflow are **optional and default-off**. Every call site degrades to
CSV/artifact fallback when the service is unavailable, which is why the test suite passes
with no Postgres and no MLflow server running.

| Env var | Default | Effect |
|---|---|---|
| `ENABLE_DATABASE` | `false` | Turns on the analytical DB path |
| `DATABASE_URL` | `sqlite:///reports/artifacts/ewhei_analytics.db` | Point at Postgres to use it |
| `ENABLE_MLFLOW` | `false` | Turns on experiment tracking |
| `MLFLOW_TRACKING_URI` | `file://<repo>/mlruns` | Point at the server for the UI |
| `MLFLOW_EXPERIMENT_NAME` | `epc-housing-energy` | Experiment name |

Read these through `settings.get_settings()`. Never call `os.getenv` for them directly.
When adding a call into either subsystem, preserve the fallback — a missing service must
degrade, never raise.

---

## Policy periods (defined in `scripts/pipeline/policy.py`)

| Period | Years | Notes |
|---|---|---|
| Pre-GreenDeal | 2008–2012 | Baseline |
| GreenDeal-ECO1 | 2013–**2015** | ECO2 launched April 2015 but 2015 stays here — annual data cannot resolve within-year transitions |
| ECO2 | 2016–2018 | |
| MEES | 2019–2020 | MEES enforcement April 2018; 2019–2020 = first full compliance cycle |
| Post-Strategy | 2021–2025 | |

**Do not move 2015 to ECO2.** The boundary is deliberate and documented in the policy.py module docstring.

---

## Statistical design decisions

### Why wild cluster bootstrap (not standard clustered SEs)
14 local authorities = 14 clusters. Cameron, Gelbach & Miller (2008) show clustered SEs are unreliable below ~30 clusters. `run_did.py` runs WCB with B=999 Rademacher weights alongside the standard clustered SEs; both appear in output tables.

### Event study vs policy periods
`BASE_RHS` uses `C(POLICY_PERIOD)` (five blunt dummies). `EVENT_STUDY_RHS` uses `C(YEAR)` with 2008 as the implicit reference. Both are estimated; the event study produces annotated year-by-year plots (`figure_6_5/6/7_event_study_*.png`).

### NEW_BUILD_SHARE covariate
Fraction of EPC lodgements flagged `TRANSACTION_TYPE == 'new dwelling'` per LA-year. Controls for the compositional shift that mechanically raises mean EPC scores in high-construction years. Computed in `build_features.py:add_new_build_share()` and included in both `BASE_RHS` and `EVENT_STUDY_RHS`.

### ML evaluation design
Splits are by property, never by row. `ml/evaluation.py` separates train/validation/test on
the identified property key (zero overlap, asserted) and tunes with `GroupKFold`. Rows without a
usable property ID (70,293) are excluded from the grouped benchmarks. The `chronological`
strategy trains before 2024, validates on 2024, tests on 2025, and purges earlier records of
held-out properties. Imputation lives inside the sklearn Pipeline (fold-local); missing floor
area stays missing in the release. Training + validation are capped at 250,000 rows; test is
uncapped (541,309 rows / 380,723 properties). Selection: min validation RMSE / max validation AUC.

### Verified results (property-v1; re-check `reports/benchmarks/` before quoting)
Regression MLP_tuned RMSE 9.429 vs LinearRegression 9.985. Classification XGBoost ROC-AUC 0.8858 vs
LogisticRegression 0.8784. 2025 chronological: 9.757 vs 10.212 RMSE, 0.8508 vs 0.8456 AUC.
Claim boundaries live in `docs/claim-evidence-ledger.md` — read it before writing any claim.
Not claimable: causal policy effects (policy bootstrap p-values blank), production deployment,
live S3/Snowflake execution, remote CI success. Clean-environment rerun: splits, selection and
selected-model predictions bit-identical; two LogisticRegression baseline metrics differ by up
to 8.8e-7 and FAIL the fixed 1e-8 tolerance. Never loosen it silently.

## Testing

```bash
make ci                 # lint + format-check + test — the full gate, mirrors GitHub Actions
python -m pytest -q     # 257 tests (last full run, 0 failures)
```

| File | Covers |
|---|---|
| `test_policy.py` | `policy_period()` boundary cases |
| `test_validation.py` | Imputed CSV schema validation |
| `test_geography.py` | LA → SW analysis-group rollup |
| `test_db.py` | SQLAlchemy models, loaders, queries |
| `test_api_database.py` | FastAPI routes incl. DB-unavailable fallback |
| `test_mlflow_tracking.py` | `*_safe` tracking helpers no-op when disabled |
| `test_monitoring.py` | Drift metrics, report rendering, retraining recommendation |
| `test_orchestration.py` | Prefect task/flow wiring and dry-run planning |
| `test_epc_source.py` | Bearer auth header, zip validation, period parsing, extracted-source discovery, compat shim |
| `test_s3_upload.py` | Period-keyed S3 key construction, skip-unchanged by stored MD5 — fake boto3 client |
| `test_load.py` | Raw DDL generation, `COPY INTO` generation, quote-aware row counting, header-diff detection, reconciliation — fake cursor |
| `test_ingestion_pipeline.py` | `run_ingest()` wiring, stage preflight, post-load authority coverage — fake clients |
| `test_cli.py` | `_require_snowflake_credentials()` — named errors on missing env vars |
| `test_release.py` | Release build: dedup, reconciled exclusions, early-year retention |
| `test_evaluation.py` | Property-separated splits, grouped folds, fold-local imputation |
| `test_benchmark_release.py` | Benchmark runner, MLflow integration, serving HTTP checks |

All ingestion tests are offline (mocked HTTP, fake boto3/cursor). They prove the logic is
internally consistent; they do not prove a real Snowflake account or S3 bucket behaves as
assumed. See `docs/ingestion.md`.

`tests/conftest.py` puts the repo root and `src/` on `sys.path`, so tests pass even when
the editable install is broken. Do not rely on a green test run as proof that `ewhei`
works — check the CLI separately with `make smoke`.

Run the suite before and after any change to `policy.py` or `validation.py`.

---

## Download modes

**Vestigial as of 2026-08-11 — the flag still exists but no longer does anything.** The EPC
service moved to GOV.UK (`epc.opendatacommunities.org` now 301-redirects to
`get-energy-performance-data.communities.gov.uk`) and no longer serves per-local-authority
zips. `--mode la` / `--mode year` are **still registered and accepted** by `ewhei download`,
`ewhei run-all` and `ewhei orchestrate` (`cli.py`'s `_MODE_HELP`, default `la`), and
`orchestrate --dry-run` still reports `'mode': 'la'` in its plan — but the choice no longer
selects between per-authority and national archives, because the bulk endpoint
(`GET /api/files/domestic/csv`) now serves **one national archive, split by period**
(~2.92 GB compressed, regenerated monthly), not one archive per LA and not one per year
selectable independently. `main(mode=...)` in `epc_download_merge_fast.py` accepts the
argument for backward compatibility with existing callers but ignores it. Both flags are
slated for removal in Phase C (Airflow cutover), when the CLI surface gets revisited anyway.
Auth is `Bearer <EPC_BEARER_TOKEN>`, not `Basic base64(email:api_key)` — see
`tasks/lessons.md` (2026-08-11 entry) for the full story and `docs/ingestion.md` for how the
downloaded archives move on into S3 and Snowflake via `ewhei ingest`.

South West filtering that used to happen at download time (picking 14 LA files) now happens
downstream, in `clean_epc.py` for the pandas pipeline and, going forward, in dbt staging for
the warehouse path (Phase B, not yet built).

---

## Docker

```bash
make build      # docker compose build
make up         # docker compose up (notebook + API + Streamlit)
make down       # docker compose down
make bash       # shell into ew-notebook container
make logs       # tail ew-notebook logs
```

Services defined in `docker-compose.yml` (service name → container → host port):

| Service | Container | Port | Notes |
|---|---|---|---|
| `postgres` | `ewhei-postgres` | 5432 | Postgres 16, healthchecked |
| `mlflow` | `ewhei-mlflow` | 5000 | Tracking server, SQLite backend |
| `notebook` | `ew-notebook` | 8888 | Jupyter, built from `Dockerfile` |
| `api` | `ewhei-api` | 8001 → 8000 | FastAPI, `Dockerfile.api` |
| `streamlit` | `ewhei-streamlit` | 8501 | `Dockerfile.streamlit` |

`make bash` and `make logs` target the `ew-notebook` container specifically.

---

## Key conventions

- **All paths from `config.py`.** Never hardcode directory paths in pipeline scripts.
- **All artifacts registered.** Every output file should call `register_artifact(REGISTRY_PATH, ...)` immediately after writing.
- **Chunked CSV reads.** Large CSVs (merged core, clean SW) are read with `chunksize=1_000_000` to avoid OOM.
- **No redundant writes.** Each intermediate CSV is written exactly once per pipeline run.
- **Optional services degrade, never raise.** DB and MLflow call sites must keep their fallback.
- **Three sanctioned `sys.path` inserts, no more.** `cli.py`, `tests/conftest.py`, and the
  `PYTHONPATH=src:.` Makefile targets reach the root-level `config` and `scripts.pipeline.*`
  modules, which are outside the src-layout package. This is why `pyproject.toml` ignores
  `E402`. Do not add a fourth — import through the package instead.


## Common Gotchas
- Never redefine `ARTIFACTS_DIR` or `REGISTRY_PATH` inline — always import from `config.py`
- `--mode la` / `--mode year` still exist as CLI flags but no longer change behaviour — the
  EPC API serves one national archive per period now, regardless of the choice. See
  Download modes.
- Always run `pytest` before and after changes to `policy.py` or `validation.py`
- Never move 2015 into ECO2 period — boundary is deliberate
- **A green `pytest` does not prove `ewhei` works.** `conftest.py` patches `sys.path`, so the
  suite passes even with a broken editable install. Verify the CLI with `make smoke`.
- **Moving or renaming the repo directory silently breaks `ewhei`.** The editable install
  records an absolute path; if it no longer resolves, `pip show` still reports the package as
  installed while every `ewhei` call dies with `ModuleNotFoundError`. Fix: re-run
  `pip install -e .` from the current directory. See `tasks/lessons.md`.
- Airflow is **not** implemented. `dags/etl_epc_pipeline.py` is a deliberate tombstone
  (`AIRFLOW_DAG_IMPLEMENTED = False`). Prefect is the orchestrator.


## Environment
- **CI runs Python 3.11.** Local dev on this machine is Anaconda **Python 3.13.5**
  (`/opt/anaconda3/bin/python`). `requires-python = ">=3.11"` permits both, but version-sensitive
  behaviour will not be caught locally — CI is the authority.
- Installed as editable package: `pip install -e .`
- CLI entry point: `ewhei` (equivalently `python -m ew_housing_energy_impact`)
- Lint/format: Ruff, line-length 100, rules `E`/`F`/`I`, `notebooks/` excluded

---

## Operating rules (non-negotiable)

**Process skills first.** If a relevant skill exists, invoke it BEFORE acting.
- New feature / "let's build X" → `brainstorming` (discuss → spec) → `writing-plans` →
  `subagent-driven-development` (fresh subagent per task, review between tasks) →
  `finishing-a-development-branch`. **Do not write code until the spec AND the plan are approved.**
- Bug / failure → `systematic-debugging`: Reproduce → Locate → Isolate → Fix → Verify.
  Never jump to a fix before isolating; never mark fixed before verifying.

**Plan mode by default.** For anything 3+ steps or architectural, plan and check in BEFORE
implementing, not during. If it goes sideways mid-task, STOP and re-plan.

**Subagents.** Offload research, codebase exploration, and parallel analysis to subagents to
keep the main context clean. Main context is for decisions and writing code.

**Core principles.**
- *Simplicity first* — the smallest change that works.
- *No laziness* — fix root causes; no workarounds, no deliberate debt.
- *Minimal impact* — touch only the files the task needs; no drive-by refactors.

**Verification before done.** Never claim done without proof. Run the commands and show the
output. "It should work" is not verification.

**Git.** Never commit on `main`; branch first. Commit and push only when asked. Use
`finishing-a-development-branch` to integrate.

**Knowledge graph.** The `code-review-graph` MCP is preferred over Grep/Glob for exploration,
but check freshness first — a stale graph will silently omit whole subsystems. Rebuild before
trusting it for impact analysis.

---

## Task & session tracking

| File | Purpose |
|---|---|
| `tasks/todo.md` | Plan as a checklist before implementing; mark `[x]` as each finishes — don't batch |
| `tasks/lessons.md` | Chronological log. After ANY correction or non-obvious discovery, append: what broke / root cause / what to do next time |
| `tasks/agent_memory.md` | Durable reference: Architecture Decisions, Known Gotchas, Solved Problems, Useful Patterns. Honour locked decisions even if the code suggests otherwise |
| `HANDOFF.md` | Current state written so a cold-start session can resume |
| `docs/plans/`, `docs/specs/` | Written specs and implementation plans |

**Update `HANDOFF.md` at every checkpoint:** task complete · milestone done · stop signalled ·
blocker hit · context ~70% · 30+ min since last update. It must contain: Current State,
Next Action (immediately actionable), In-Flight files, Open Questions, Verification Baseline.

**At the START of every session:** read `HANDOFF.md`, `tasks/todo.md`, `tasks/lessons.md`,
`tasks/agent_memory.md`, then run `git status` and `git log -1` and reconcile. **If they
diverge, the working tree is authoritative.**

---

## Definition of Done

A task is not complete until all of the following hold:

1. `make ci` passes — lint, format-check, and tests (last full run: **257 passing** — recount with `pytest --collect-only -q` before quoting).
2. `make smoke` passes if the CLI surface changed.
3. New logic has tests.
4. `HANDOFF.md` reflects the current state.

Run the checks. Do not assert them.