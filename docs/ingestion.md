# Ingestion

## What it does
EPC archives -> S3 -> Snowflake raw, with reconciliation proving the load.

## Running it
    ewhei ingest --dry-run    # planned keys and expected row counts
    ewhei ingest              # upload, load, reconcile, check authorities

One-time Snowflake and IAM setup: see `snowflake/README.md`.

Required environment variables (read through `settings.get_settings()`, never
`os.getenv` at the call site):

| Variable | Purpose |
|---|---|
| `EPC_BEARER_TOKEN` | Auth for the EPC download step (`ewhei download`), obtained from the account page. Not needed for `ewhei ingest` itself if extracted sources already exist on disk. |
| `EPC_S3_BUCKET` | Defaults to `epc-sw-pipeline-dev` if unset. |
| `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` | Required for the non-dry-run path. Missing any of them raises one `ValueError` naming all that are absent, before touching S3 or Snowflake. |

`ewhei ingest` reads whatever `iter_extracted_sources()` finds under
`RAW_EXTRACT_DIR` — it does not download anything itself. `--dry-run` builds
the plan and returns without importing `boto3` or `snowflake.connector`, so it
runs with none of the above credentials set.

## S3 layout
    raw/certificates/<period>/certificates.csv
    raw/recommendations/<period>/recommendations.csv
Period is `2025-09` or `2024`, taken from the archive name. Keys carry no run
timestamp, so a re-run overwrites in place instead of accumulating copies.

## Why raw is all STRING
EPC ships 'NO DATA!', 'INVALID!', empty numerics and inconsistent dates.
Typing here would reject rows at load — exactly what a faithful copy of
source must not do.

## Why sentinels are preserved
'Surveyor recorded nothing' and 'field absent from the form' are different
facts. Nulling both at load destroys that distinction irreversibly.

## Why columns bind by name
Every column is STRING, so positional binding would let a reordered export
land postcodes in the rating column with nothing objecting.

## Why the load is a full refresh
Snowflake's COPY load metadata expires after 64 days, so skip-already-loaded
is correct for 64 days then silently duplicates. EPC archives are complete
snapshots per period, so TRUNCATE then COPY is correct and deterministic.

## Why row counting parses instead of counting lines
Free-text columns can contain newlines inside quoted values. A line count
would exceed the true record count and fail a correct load.

## Failure modes

Principle: fail loudly at the boundary where the cause is still obvious.

| Stage | Failure | Handling |
|---|---|---|
| Auth | Missing/invalid EPC credentials | Immediate named error from `auth_headers()` |
| Download | Truncated or corrupt zip | `_is_valid_zip` rejects the cached file and re-downloads; atomic temp-then-rename write means an interrupted download never leaves a partial file that looks cached; 3 retries with exponential backoff |
| Extract | Zip passes download validation but fails extraction | The zip is deleted and a `RuntimeError` is raised naming the period. The next run re-downloads it — an inline retry would couple the stages and could loop on a persistently corrupt file |
| Load | Header set differs from the raw table's existing columns | Hard fail with an explicit added/removed diff (`load_kind` / `header_diff`), naming the period. Overridable only with `ewhei ingest --allow-schema-change`, so absorbing a source schema change is a deliberate human act, not a silent one |
| Upload | S3 error mid-run | 3 retries with exponential backoff (botocore's standard retry mode, configured in `cli.py`), then fail. Partial uploads are safe: stable keys mean a re-run overwrites and reconciles |
| Snowflake | Storage integration or IAM trust misconfigured | `preflight_stage` runs `LIST @epc_raw_stage` before any `COPY INTO`, so a broken integration surfaces as one clear error instead of one confusing failure per table |
| Load | Malformed row | `ON_ERROR = ABORT_STATEMENT` on the `COPY INTO` — with all-STRING columns a load error means something is genuinely wrong, not a formatting quirk to skip past |
| Verify | Row counts do not reconcile | `reconcile()` raises, naming which kind mismatched and by how much. A zero exit code from `COPY INTO` is not evidence of a correct load — this is |
| Post-load | An analysis authority is absent from the loaded data | Hard fail — `assert_authority_coverage` raises after the load, naming the missing authority. There is no per-authority source file any more (national period files), so this check can only run after data lands in `raw.certificates` |

**What is and is not verified.** Every test behind these failure modes is
offline — a fake boto3 client, a fake cursor, mocked HTTP. They prove the
logic is internally consistent: given a broken stage, `preflight_stage`
raises; given a mismatched count, `reconcile` raises; given a missing
authority, `assert_authority_coverage` raises. None of them prove that
`MATCH_BY_COLUMN_NAME` behaves as assumed against a real Snowflake account,
that the storage integration actually authenticates through IAM, or that
reconciliation matches on a real EPC file rather than a hand-built fixture.
That is only proven by running `ewhei ingest` against a real Snowflake
account with a real bearer token — see the manual verification steps in
`.superpowers/sdd/2026-08-11-epc-warehouse-ingestion/task-7-brief.md` and
`HANDOFF.md`. As of this writing that run has not happened.
