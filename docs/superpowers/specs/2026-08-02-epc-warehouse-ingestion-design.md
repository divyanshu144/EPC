# EPC Warehouse Ingestion — Design (Phase A)

**Date:** 2026-08-02
**Amended:** 2026-08-11 — see §0
**Status:** Approved, amended, pending implementation
**Scope:** Phase A of a four-phase migration

---

## 0. Amendment, 2026-08-11

The original design assumed the EPC bulk API served **per-local-authority zips**. That is no
longer true, and the assumption was load-bearing.

**What changed externally.** `epc.opendatacommunities.org` now 301-redirects to
`get-energy-performance-data.communities.gov.uk`, a rebuilt GOV.UK service. Discovered when
`ewhei download` failed with a JSON decode error: the redirect lands on an HTML page, so
`raise_for_status()` passes and `.json()` dies at char 0.

| | Original assumption | Actual |
|---|---|---|
| Base URL | `epc.opendatacommunities.org` | `api.get-energy-performance-data.communities.gov.uk` |
| Auth | `Basic base64(email:api_key)` | `Bearer <token>` from the account page |
| Bulk endpoint | `/api/v1/files` → 14 per-LA zips | `GET /api/files/domestic/csv` → one national zip |
| Granularity | per local authority | **national, split by period** |
| Size | ~200–300 MB | ~2.92 GB compressed |
| Refresh | ad hoc | regenerated monthly |

**Consequences.** Per-LA selection is impossible, so `pick_la_files` and code-based download
filtering have no job. S3 keys become **period-keyed**, not LA-keyed. The missing-authority
check moves from "is every LA's file present" to a post-load check on the data itself.
`requests` strips `Authorization` across the 302 to presigned S3, which is correct here.

**What was verified empirically** against 9 real national monthly zips
(`domestic-2025-01` … `2025-09`) held locally:

- `certificates.csv` — **93 columns**, header byte-identical across all 9 (`sha ee4ad0ef5800`)
- `recommendations.csv` — **7 columns**, header byte-identical across all 9 (`sha 306b72d45793`)
- Domestic recommendations **do** ship inside the domestic zip, despite the new API listing no
  domestic-recommendations endpoint. **D4 survives.**

Identical headers mean by-name column binding (D15) is a guard against future drift rather
than a fix for present breakage.

**Local data caveat.** `data copy/` holds those 9 zips plus a stale
`processed/ew_epc_core_clean_sw/` of 1,807,888 rows whose header lacks
`BUILDING_REFERENCE_NUMBER`, `ANALYSIS_AUTHORITY` and the `*_RAW` columns, and which contains
only `E06*` unitaries. It predates commit `5319ee9` and therefore carries the district-dropping
bug that commit fixed. **Do not feed it forward.** Certificates for 2008–2024 are absent
entirely; only 2025-01…09 exist locally.

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
| D5 | **Split** `epc_download_merge_fast.py`, keep the download *mechanics* only | **Amended 2026-08-11.** The plumbing survives — atomic writes, zip validation, cache reuse, retry. But `pick_la_files`, `_matched_codes` and code-based download selection are now dead: there are no per-LA files to select. `auth_headers` is dead too (Basic → Bearer). Move the mechanics, retire the selection and auth logic. |
| D13 | Source is **national, period-partitioned** | The API serves one national zip split by year; the 9 local files are national by month. Neither is per-LA. Raw is keyed by period (`2025-01`, `2025`), and South West filtering moves to dbt staging where it is version-controlled and testable. |
| D14 | Auth is **`Bearer <token>`** from `EPC_BEARER_TOKEN` | The new service issues a bearer token on the account page. `EPC_EMAIL`/`EPC_API_KEY` are obsolete. Read via `settings.get_settings()`, never `os.getenv` at the call site. |
| D15 | Bind columns **by name**, not position | `MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE` with `PARSE_HEADER = TRUE` (mutually exclusive with `SKIP_HEADER`), plus a Python header-equality preflight that fails with a diff. Every column is STRING, so positional binding would let a reordered export land postcodes in the rating column with nothing objecting. **`ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE` is removed — it directly contradicted D9.** |
| D16 | Row counting is **quote-aware** | Counting newlines miscounts any record with an embedded newline inside a quoted value, which D6's free-text `*_DESCRIPTION` columns can contain. Reconciliation is the entire proof of Phase A, so it must count CSV records via `csv.reader` (opened with `newline=""`), not lines. Snowflake's own `rows_loaded` is not a substitute: comparing it to `COUNT(*)` is Snowflake checking itself and cannot detect a truncated upload. |
| D6 | Raw tables take **every column**, not `CORE_COLS` | Column selection is a modelling decision belonging in staging. Critically, `CORE_COLS` discards the `*_DESCRIPTION` fields that Phase D's LLM normalisation operates on — selecting at ingestion would foreclose Phase D. |
| D7 | Raw tables typed **all `STRING`** | EPC carries `NO DATA!`, `INVALID!`, empty numerics and inconsistent dates. Typed raw would reject rows at load — exactly what bronze must not do. Cast in staging with `TRY_TO_*`, where a failed cast is a testable NULL rather than a lost row. |
| D8 | **Stable S3 keys**, no run timestamps | **Amended:** keys identify a *period's* snapshot, not a run — `raw/certificates/<period>/certificates.csv`. The period is part of the data's identity; a run timestamp is not, and would accumulate one copy per run forever. Stable keys also let the uploader skip unchanged files by stored hash. |
| D8b | Raw load is a **full refresh**: `TRUNCATE` then `COPY INTO`, per table, per run | Snowflake's `COPY INTO` load metadata **expires after 64 days**. Relying on its skip-already-loaded behaviour is therefore correct for 64 days and silently starts duplicating rows afterwards — a bug that would appear months later and be very hard to trace. EPC bulk files are *complete snapshots per authority*, not increments, so a full refresh is both semantically correct and deterministic. At ~1.8M rows the reload is cheap; correctness is worth more than the saved seconds. |
| D9 | **`ON_ERROR = ABORT_STATEMENT`** | `CONTINUE` silently discards malformed rows, contradicting the claim that raw is a true copy of source. With all-STRING columns genuine errors should be near-impossible, so one occurring means something is actually wrong. |
| D10 | **Minimal `NULL_IF`** — preserve sentinels verbatim | "Surveyor recorded nothing" and "field absent from the form" are different facts. Collapsing both to NULL at the bronze boundary destroys information irreversibly. Staging decides what each sentinel means. |
| D11 | Authority rollup becomes a **dbt seed** | `CODE_TO_ANALYSIS_AUTHORITY` is reference data, not logic. A seed CSV is diffable, testable and reviewable when boundaries change. S3 keys and raw tables use the raw EPC code. |
| D12 | Clients are **injected**, never constructed internally | Phase A is almost entirely I/O. Injection is what makes it testable without AWS, Snowflake or the EPC API. |

---

## 4. Architecture

```
api.get-energy-performance-data.communities.gov.uk
        │                      Authorization: Bearer <EPC_BEARER_TOKEN>
        │                      GET /api/files/domestic/csv → 302 → presigned S3
        ▼
ingestion/epc_source.py        national zip(s) → data/raw/ew_bulk/
        │                      atomic write · zip validation · cache reuse
        ▼                      extract → data/raw/ew_extracted/<period>/
        │                        ├── certificates.csv      (93 cols)
        │                        └── recommendations.csv   (7 cols)
        ▼
ingestion/s3_upload.py         s3://<bucket>/raw/certificates/<period>/certificates.csv
        │                      s3://<bucket>/raw/recommendations/<period>/recommendations.csv
        ▼
Snowflake storage integration  IAM role ⇄ bidirectional trust (external ID)
        │
        ▼
@epc_raw_stage ──COPY INTO──▶  raw.certificates · raw.recommendations
        │                      MATCH_BY_COLUMN_NAME, ABORT_STATEMENT
        ▼
row-count reconciliation       Snowflake COUNT(*) vs local csv.reader record counts
        │
        ▼
authority coverage check       all 14 analysis authorities present in loaded data
```

`<period>` is whatever partition the source ships — `2025-01` for the monthly files held
locally, `2025` for the API's yearly split. It is derived from the archive name, never
invented by us.

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
| File format | `epc.raw.csv_format` | `PARSE_HEADER=TRUE` (**not** `SKIP_HEADER` — mutually exclusive), `FIELD_OPTIONALLY_ENCLOSED_BY='"'`, minimal `NULL_IF` per D10. Loaded with `MATCH_BY_COLUMN_NAME=CASE_INSENSITIVE` per D15 |

**Region:** Snowflake must be created on AWS in `eu-west-2` (London) to match the S3 bucket
and avoid cross-region egress. The AWS account id is supplied via the `AWS_ACCOUNT_ID` environment variable and deliberately kept out of version control (public repo); region is
`eu-west-2`.

**Bucket name:** `epc-sw-pipeline-dev`, in `eu-west-2`. S3 bucket names are globally unique,
so if that name is taken the fallback is `epc-sw-pipeline-dev-<account-suffix>`. Confirm
availability as the first implementation step, before any IAM policy is written against it.

### IAM

A policy granting only `GetObject` and `ListBucket`, scoped to the single bucket. A role
trusting Snowflake's IAM user ARN plus the external ID returned by `DESCRIBE INTEGRATION`.

**`GetObjectVersion` is deliberately excluded** and bucket versioning stays off. We never
read a specific object version, and enabling versioning would break D8's claim that stable
keys avoid accumulating copies — each overwrite would retain a noncurrent version, requiring
a lifecycle rule to expire them. Off is simpler and keeps D8 true.

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
| Extract | Zip passes download validation but fails extraction | **Delete the zip and raise**, naming the period. The next run re-downloads because the archive is gone. An inline re-download couples the stages and can loop on a persistently corrupt file. The current `print`-and-continue is wrong: it degrades into a missing period discovered much later |
| Post-load | An analysis authority is **absent from the loaded data** | **Hard fail.** A missing authority is a hole in a 14-cluster panel and every downstream fixed-effects estimate is silently wrong. **Amended:** with national period files there is no per-LA file whose absence we can detect, so this check moves *after* the load — count distinct rolled-up authorities in `raw.certificates` and fail unless all 14 appear |
| Load | Header set differs from the raw table's columns | **Hard fail with an explicit diff** naming the period and the added/removed columns. Overridable only via `--allow-schema-change`, so absorbing a source schema change is a deliberate human act rather than a silent one |
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
| `tests/test_epc_source.py` | Bearer header construction, zip validation, period parsing, extracted-source discovery, compat shim |
| `tests/test_s3_upload.py` | Key construction per period, skip-unchanged by stored hash — fake boto3 client |
| `tests/test_load.py` | DDL generation, `COPY INTO` generation, **quote-aware row counting**, header-diff detection, reconciliation — fake cursor |
| `tests/test_ingestion_pipeline.py` | Wiring, stage preflight, post-load authority coverage |
| `tests/test_geography.py` | **Imports updated only** — see below |

Integration tests that genuinely reach S3 and Snowflake sit behind a pytest marker,
deselected by default, run manually. They are valuable but must not gate CI.

### Rule for the split

`test_geography.py` has nine test functions, but they do not all guard the same thing:

| Exercises | Functions | Effect of the split |
|---|---|---|
| `epc_download_merge_fast` | `test_pick_la_files_selects_by_district_code`, `test_missing_analysis_authorities_detection` | **Both now test dead code** (D5/D13) — per-LA selection no longer exists |
| `config` directly | 5 functions, one parametrized over 19 cases | Unaffected in behaviour; only the module-level import moves |
| `clean_epc.clean_chunk` | 2 functions | Unaffected — Phase B's concern |

So the split's behavioural regression guard was always **two** functions, and the amendment
has retired both. The imports-only rule still binds for the other seven: **if an assertion
needs to change, the split went wrong — stop and investigate rather than adjusting the test.**

Because the guard is now thin, Task 1's new tests carry the weight, plus a cheap identity
assertion that the compatibility shim exposes the same objects
(`epc_download_merge_fast.extract_all_zips is epc_source.extract_all_zips`), which catches a
botched split that a silent re-implementation would slip past.

The two tests covering retired functions are **deleted, not rewritten** — keeping a green
test over code that can never run again is worse than having no test, because it reads as
coverage.

---

## 8. Definition of done

1. `make ci` passes — lint, format-check, and the full suite including new tests.
2. `LIST @epc_raw_stage` shows certificates and recommendations for every loaded period.
3. Snowflake row counts reconcile against local **`csv.reader` record counts** (D16), per table.
4. All 14 analysis authorities are present **in the loaded data**; a missing one fails the run.
5. A second consecutive run produces **identical row counts, not doubled ones** — the
   full-refresh load (D8b) is deterministic, and the uploader skips files whose stored hash
   matches.
6. **Reconciliation is proven against a file containing a quoted embedded newline**, not only
   against clean fixtures.
7. **Column binding is by name**, and a header mismatch fails rather than NULL-filling.
8. `HANDOFF.md` reflects the state.

Exit code zero from `COPY INTO` is not success. Reconciliation is.

**Scope note.** Items 2–5 are satisfied over the periods actually loaded. With only
2025-01…09 available locally, item 4 is a genuine test — the South West's 14 authorities all
lodge certificates in any given month — but full-history coverage awaits the backfill (R8/R9).

---

## 9. Risks and open items

| # | Risk | Status |
|---|---|---|
| R1 | **Snowflake trial expires in ~30 days**, after which the system stops. Accepted under D2. Mitigation if it matters later: keep dbt SQL reasonably portable so a move to Postgres is a profile change plus minor edits | Open, accepted |
| R2 | ~~EPC data not on disk~~ — **partly resolved.** 9 national monthly zips (2025-01…09) exist in `data copy/`, enough to build and prove the full path on real data. **Certificates for 2008–2024 are still absent** and require the bearer token plus the new yearly endpoint | Partly resolved |
| R8 | **`EPC_BEARER_TOKEN` does not exist yet.** The account-page token is a user action — it cannot be obtained programmatically. Phase A can be built and unit-tested without it; the 2008–2024 backfill cannot | Blocking backfill only |
| R9 | **The 2008–2024 backfill is unscoped.** The API serves one ~2.92 GB national zip split by year, estimated 15–25 GB unzipped. Disk is fine (184 GB free) but load volume into a trial Snowflake account is untested | Open |
| R10 | **`data copy/processed/` is stale and carries a fixed bug.** Missing `BUILDING_REFERENCE_NUMBER`/`ANALYSIS_AUTHORITY`, `E06`-only. Predates `5319ee9`. Must not be fed forward; regenerate from raw via dbt | Mitigated by not using it |
| R3 | **`dbt-core` on Python 3.13.** Local interpreter is Anaconda 3.13.5; dbt has historically lagged the newest Python. May require a 3.11/3.12 venv | Open, verify early |
| R4 | **Splitting a 335-line module** risks breaking 9 geography tests in ways easily papered over | Mitigated by the imports-only rule in §7 |
| R5 | Docker has 7.75 GB RAM; Airflow plus Snowflake connections is workable but not generous | Monitor in Phase C |
| R6 | ~~Bucket name undecided~~ — **resolved.** `epc-sw-pipeline-dev` confirmed available in `eu-west-2` (`HeadBucket` → 404), and the account currently holds no buckets | Closed |
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
