# EPC Warehouse Ingestion — Design (Phase A)

**Date:** 2026-08-02
**Status:** Approved, pending implementation plan
**Scope:** Phase A of a four-phase migration

---

## 1. Context

The EPC repository currently runs a Python/pandas pipeline: CSVs are downloaded from the
EPC API, merged, cleaned and feature-engineered on local disk, orchestrated by Prefect,
with results served through FastAPI and Streamlit and analysed by `run_did.py` (PanelOLS)
and `train_models.py` (XGBoost/SHAP).

The goal is to replace that transformation and orchestration layer with a warehouse-centred
stack — S3 → Snowflake → dbt → Airflow — running on the real EPC data, and to add an AI
lane on top.

This document specifies **Phase A only**: getting EPC source data into Snowflake's `raw`
schema as a faithful bronze layer.

### Why this shape

The reference for this work is a Zomato food-delivery pipeline tutorial. That project's AI
lane is built on `reviews.comment`, free prose written by customers. **EPC data contains no
prose.** It is a structured building survey: ratings, kWh, m², categoricals. Sentiment
classification over it would be meaningless, and a RAG chatbot over EPC's ~40 templated
recommendation strings would be close to useless.

The AI lane is therefore redesigned around problems EPC actually has (see Phase D), and the
transferable pattern taken from the reference is structural: **an LLM as a transformation
step inside the DAG, writing structured columns back to the warehouse, registered as a dbt
source so downstream models treat it as ordinary data.**

---

## 2. Phase decomposition

| Phase | Delivers | Replaces |
|---|---|---|
| **A** (this doc) | Certificates + recommendations in S3 and Snowflake `raw`; storage integration, IAM trust, stage, file format, raw tables, `COPY INTO`, row-count reconciliation | Retires the *merge* half of `epc_download_merge_fast.py`; relocates the *download* half |
| **B** | dbt `staging` views and `marts` tables, declarative tests, lineage docs | `clean_epc.py`, `build_features.py`, `validation.py` |
| **C** | Single Airflow DAG: load → dbt → Python analytics → AI. Prefect retired, `run_did`/`train_models` reading from Snowflake, API and Streamlit repointed | `orchestration/` (Prefect) |
| **D** | LLM categorical normalisation, text-to-SQL, RAG over recommendations and `docs/` | new |

Dependencies: **A → B → C**. D may start once B lands, as it requires staging tables.

Phase C carries the most risk: it is the only phase that removes working, tested code.

---

## 3. Decisions

Decisions are recorded with rationale, including where the chosen option differs from the
recommendation, so later readers understand what was traded away.

| # | Decision | Rationale |
|---|---|---|
| D1 | **Replace** the existing stack rather than extend it | Prefect + SQLAlchemy + dbt + Airflow in one repo means two orchestrators, two warehouses and two transformation layers. One of each. |
| D2 | **Snowflake** as the warehouse | Chosen over the Postgres already present in `docker-compose.yml`. Postgres was recommended: free, permanent, concurrent, and `dbt-postgres` is first-class at 1.8M rows. Snowflake was chosen for the learning value and recognisability. **Accepted trade-off: the trial is ~$400 / 30 days, after which the system stops.** See Risk R1. |
| D3 | **True bronze** — unmodified certificates *and* recommendations | Raw must be an honest copy of source. All filtering, rollup and cleaning moves into dbt where it is version-controlled and testable. |
| D4 | Ingest `recommendations.csv` | Already extracted from every zip and discarded. It is the only text-ish source available and Phase D needs it. Retrofitting later means touching S3 layout, raw tables and `COPY INTO` again. |
| D5 | **Split** `epc_download_merge_fast.py`, keep the download half | The download logic is well built — atomic writes, zip validation, cache reuse, retry. The merge half is obsolete under dbt. Preserve earned code, delete replaced code. |
| D6 | Raw tables take **every column**, not `CORE_COLS` | Column selection is a modelling decision belonging in staging. Critically, `CORE_COLS` discards the `*_DESCRIPTION` fields that Phase D's LLM normalisation operates on — selecting at ingestion would foreclose Phase D. |
| D7 | Raw tables typed **all `STRING`** | EPC carries `NO DATA!`, `INVALID!`, empty numerics and inconsistent dates. Typed raw would reject rows at load — exactly what bronze must not do. Cast in staging with `TRY_TO_*`, where a failed cast is a testable NULL rather than a lost row. |
| D8 | **Stable S3 keys**, no timestamps | Keys identify an authority's current snapshot, not a run. Stable keys let the uploader skip unchanged files by etag, and keep the stage listing readable. Timestamped keys would accumulate one copy per run forever. |
| D8b | Raw load is a **full refresh**: `TRUNCATE` then `COPY INTO`, per table, per run | Snowflake's `COPY INTO` load metadata **expires after 64 days**. Relying on its skip-already-loaded behaviour is therefore correct for 64 days and silently starts duplicating rows afterwards — a bug that would appear months later and be very hard to trace. EPC bulk files are *complete snapshots per authority*, not increments, so a full refresh is both semantically correct and deterministic. At ~1.8M rows the reload is cheap; correctness is worth more than the saved seconds. |
| D9 | **`ON_ERROR = ABORT_STATEMENT`** | `CONTINUE` silently discards malformed rows, contradicting the claim that raw is a true copy of source. With all-STRING columns genuine errors should be near-impossible, so one occurring means something is actually wrong. |
| D10 | **Minimal `NULL_IF`** — preserve sentinels verbatim | "Surveyor recorded nothing" and "field absent from the form" are different facts. Collapsing both to NULL at the bronze boundary destroys information irreversibly. Staging decides what each sentinel means. |
| D11 | Authority rollup becomes a **dbt seed** | `CODE_TO_ANALYSIS_AUTHORITY` is reference data, not logic. A seed CSV is diffable, testable and reviewable when boundaries change. S3 keys and raw tables use the raw EPC code. |
| D12 | Clients are **injected**, never constructed internally | Phase A is almost entirely I/O. Injection is what makes it testable without AWS, Snowflake or the EPC API. |

---

## 4. Architecture

```
EPC API (opendatacommunities.org)      Basic auth: base64(email:api_key)
        │
        ▼
ingestion/epc_source.py        14 SW LA zips → data/raw/ew_bulk/
        │                      atomic write · zip validation · cache reuse
        ▼                      extract → data/raw/ew_extracted/<la>/
        │                        ├── certificates.csv
        │                        └── recommendations.csv
        ▼
ingestion/s3_upload.py         s3://<bucket>/raw/certificates/<la_code>/certificates.csv
        │                      s3://<bucket>/raw/recommendations/<la_code>/recommendations.csv
        ▼
Snowflake storage integration  IAM role ⇄ bidirectional trust (external ID)
        │
        ▼
@epc_raw_stage ──COPY INTO──▶  raw.certificates · raw.recommendations
        │
        ▼
row-count reconciliation       Snowflake counts vs local line counts − headers
```

### Components

| Module | Single responsibility | Offline-testable |
|---|---|---|
| `ingestion/epc_source.py` | Fetch, validate, extract zips | Yes — mocked HTTP |
| `ingestion/s3_upload.py` | Put objects at stable keys; skip when the local MD5 equals the existing object's ETag | Yes — fake boto3 client |
| `ingestion/load.py` | Execute `COPY INTO`, return per-table row counts | Yes — fake cursor |
| `snowflake/01–05.sql` | Idempotent DDL | Re-runnable |

---

## 5. Snowflake objects

| Object | Name | Notes |
|---|---|---|
| Warehouse | `epc_wh` | XSMALL, `AUTO_SUSPEND = 60` |
| Database | `epc` | |
| Schemas | `raw`, `staging`, `marts`, `snapshots`, `ai` | Medallion + dbt snapshots + AI output |
| Role | `epc_dbt_role` | Granted on current *and future* objects |
| Storage integration | `epc_s3_int` | |
| Stage | `epc.raw.epc_raw_stage` | |
| File format | `epc.raw.csv_format` | `SKIP_HEADER=1`, `FIELD_OPTIONALLY_ENCLOSED_BY='"'`, `ERROR_ON_COLUMN_COUNT_MISMATCH=FALSE`, minimal `NULL_IF` per D10 |

**Region:** Snowflake must be created on AWS in `eu-west-2` (London) to match the S3 bucket
and avoid cross-region egress. The existing AWS account is `${AWS_ACCOUNT_ID}`, region
`eu-west-2`.

**Bucket name:** `epc-sw-pipeline-dev`, in `eu-west-2`. S3 bucket names are globally unique,
so if that name is taken the fallback is `epc-sw-pipeline-dev-<account-suffix>`. Confirm
availability as the first implementation step, before any IAM policy is written against it.

### IAM

A policy granting only `GetObject`, `GetObjectVersion`, `ListBucket`, scoped to the single
bucket. A role trusting Snowflake's IAM user ARN plus the external ID returned by
`DESCRIBE INTEGRATION`.

The trust is bidirectional by necessity, not ceremony. Snowflake's IAM user assumes roles
on behalf of many customers; without the external ID, anyone knowing that user's ARN could
write a trust policy naming it and have Snowflake — a trusted third party — fetch from a
bucket they are not entitled to. This is the confused-deputy problem, and the external ID
is the shared secret that scopes the assume-role to one specific integration.

The pipeline never uses admin credentials; only this scoped role.

---

## 6. Error handling

Principle: **fail loudly at the boundary where the cause is still obvious.**

| Stage | Failure | Handling |
|---|---|---|
| Auth | Missing/invalid EPC credentials | Immediate named error (already implemented) |
| Download | Truncated or corrupt zip | `_is_valid_zip` rejects; atomic temp-then-rename prevents a partial file resembling a valid cache; **3 retries, exponential backoff** |
| Download | An analysis authority has **no** file | **Hard fail.** A missing LA is a hole in a 14-cluster panel and every downstream fixed-effects estimate is silently wrong. `_missing_analysis_authorities` already detects this; Phase A promotes it from warning to error |
| Upload | S3 error mid-run | **3 retries, exponential backoff**, then fail. Partial uploads are safe: stable keys mean a re-run overwrites and reconciles |
| Snowflake | Storage integration misconfigured | Preflight `LIST @epc_raw_stage` before any `COPY INTO`, so a trust-policy failure surfaces as one clear error rather than fourteen confusing ones |
| Load | Malformed row | `ABORT_STATEMENT` (D9) |
| Verify | Row counts do not reconcile | Fail the phase |

---

## 7. Testing

**Binding constraint:** GitHub Actions has no AWS, Snowflake or EPC credentials, and the
existing 68 tests all run offline. That must remain true — a suite that passes only on one
laptop is not a suite.

Every new unit test injects its client and touches no network.

| Test file | Covers |
|---|---|
| `tests/test_epc_source.py` | Auth header construction, zip validation, cache reuse, retry — mocked HTTP |
| `tests/test_s3_upload.py` | Key construction per LA, skip-unchanged, retry — fake boto3 client |
| `tests/test_load.py` | `COPY INTO` generation, row-count parsing, reconciliation — fake cursor |
| `tests/test_geography.py` | **Existing 9 tests, imports updated only** — the regression guard on the split |

Integration tests that genuinely reach S3 and Snowflake sit behind a pytest marker,
deselected by default, run manually. They are valuable but must not gate CI.

### Rule for the split

`test_geography.py` currently imports `scripts.pipeline.epc_download_merge_fast` and
exercises `pick_la_files`, `_missing_analysis_authorities` and the rollup.
`test_orchestration.py` reaches it via the Prefect task.

When lifting functions into `ingestion/epc_source.py`, **those tests change imports only.**
If an assertion needs to change, the split went wrong — stop and investigate rather than
adjusting the test to match new behaviour.

---

## 8. Definition of done

1. `make ci` passes — lint, format-check, and the full suite including new tests.
2. All 14 analysis authorities present; a missing one fails the run.
3. `LIST @epc_raw_stage` shows certificates and recommendations for all 14 authorities.
4. Snowflake row counts reconcile against local CSV line counts minus headers, per table.
5. A second consecutive run produces **identical row counts, not doubled ones** — the
   full-refresh load (D8b) is deterministic, and the uploader skips files whose MD5 matches
   the existing ETag.
6. `HANDOFF.md` reflects the state.

Exit code zero from `COPY INTO` is not success. Reconciliation is.

---

## 9. Risks and open items

| # | Risk | Status |
|---|---|---|
| R1 | **Snowflake trial expires in ~30 days**, after which the system stops. Accepted under D2. Mitigation if it matters later: keep dbt SQL reasonably portable so a move to Postgres is a profile change plus minor edits | Open, accepted |
| R2 | **The EPC data is not on disk.** `data/` is empty and `.env` does not exist. Nothing runs until credentials are added and `ewhei download` completes (~200–300 MB) | Blocking, user action |
| R3 | **`dbt-core` on Python 3.13.** Local interpreter is Anaconda 3.13.5; dbt has historically lagged the newest Python. May require a 3.11/3.12 venv | Open, verify early |
| R4 | **Splitting a 335-line module** risks breaking 9 geography tests in ways easily papered over | Mitigated by the imports-only rule in §7 |
| R5 | Docker has 7.75 GB RAM; Airflow plus Snowflake connections is workable but not generous | Monitor in Phase C |
| R6 | Bucket name not yet chosen (must be globally unique) | Decide at implementation |
| R7 | Whether `ewhei` remains the human-facing CLI once Airflow owns scheduling | Defer to Phase C |

---

## 10. Out of scope for Phase A

- dbt models of any kind (Phase B)
- Airflow, and removal of Prefect (Phase C)
- All AI features (Phase D)
- Repointing FastAPI and Streamlit (Phase C)
- Migrating `run_did.py` and `train_models.py` to read from Snowflake (Phase C). Note these
  are PanelOLS and XGBoost/SHAP — not expressible as dbt models. They remain Python tasks
  in the DAG, reading from the warehouse.
