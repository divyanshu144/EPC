# EPC Warehouse Ingestion (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land EPC certificates and recommendations in S3 and Snowflake's `raw` schema as a faithful bronze layer, with row-count reconciliation proving the load.

**Architecture:** Split the download half of `scripts/pipeline/epc_download_merge_fast.py` into a new `ingestion` subpackage; add an S3 uploader keyed by local authority and a Snowflake loader that truncates then copies. All three modules take their external client as an argument so they are testable with no network. The merge half stays in place, unused, until Phase C deletes it.

**Tech Stack:** Python 3.11+, boto3, snowflake-connector-python, pytest, Snowflake, AWS S3 (eu-west-2).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md`. Every decision ID (D1–D12) below refers to it.
- **All new unit tests run offline.** GitHub Actions has no AWS, Snowflake, or EPC credentials. Any test needing them is marked `@pytest.mark.integration` and deselected by default.
- **The existing 68 tests must keep passing at every commit.**
- `test_geography.py` changes **imports only** (spec §7). If an assertion needs changing, stop — the split went wrong.
- Raw tables are typed **all `STRING`** (D7).
- `ON_ERROR = ABORT_STATEMENT` (D9). Never `CONTINUE`.
- `NULL_IF` stays minimal — sentinels `NO DATA!` and `INVALID!` are preserved verbatim (D10).
- S3 keys carry **no timestamps** (D8).
- Raw load is **`TRUNCATE` then `COPY INTO`** (D8b).
- Lint/format: `ruff`, line-length 100. Run `make ci` before every commit.
- Never commit on `main`. Work happens on `feat/warehouse-ingestion`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ew_housing_energy_impact/ingestion/__init__.py` | Package marker |
| `src/ew_housing_energy_impact/ingestion/epc_source.py` | Fetch, validate, extract EPC zips; enumerate extracted sources |
| `src/ew_housing_energy_impact/ingestion/s3_upload.py` | Upload to stable keys, skip unchanged via stored MD5 |
| `src/ew_housing_energy_impact/ingestion/load.py` | Generate raw DDL, run `TRUNCATE`/`COPY INTO`, reconcile row counts |
| `src/ew_housing_energy_impact/ingestion/pipeline.py` | Wire the three together; authority check, stage preflight, reconciliation |
| `snowflake/01_setup.sql`, `02_storage_integration.sql`, `03_stage_and_format.sql` | Hand-applied DDL. Raw table DDL and `COPY INTO` are **generated at runtime** by `load.py` from the CSV header, not written by hand — the ~90 columns are not known until the data is on disk |
| `snowflake/README.md` | One-time setup runbook including the IAM trust dance |
| `tests/test_epc_source.py` | Offline tests, mocked HTTP |
| `tests/test_s3_upload.py` | Offline tests, fake S3 client |
| `tests/test_load.py` | Offline tests, fake cursor |
| `tests/test_ingestion_pipeline.py` | Offline tests for the wiring |
| `tests/test_geography.py` | **Modify imports only** |
| `scripts/pipeline/epc_download_merge_fast.py` | **Modify** — re-import moved functions, keep merge half working |
| `src/ew_housing_energy_impact/cli.py`, `settings.py`, `Makefile` | **Modify** — add `ewhei ingest`, the bucket setting, and the smoke dry-run |
| `docs/ingestion.md` | Explains the design decisions to a future reader |

---

### Task 1: Create the ingestion package and move download functions

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/__init__.py`
- Create: `src/ew_housing_energy_impact/ingestion/epc_source.py`
- Modify: `scripts/pipeline/epc_download_merge_fast.py`
- Modify: `tests/test_geography.py:13` (import line only)
- Test: `tests/test_epc_source.py`

**Interfaces:**
- Consumes: `config.SOUTH_WEST_EPC_CODES`, `config.ANALYSIS_AUTHORITIES`, `config.CODE_TO_ANALYSIS_AUTHORITY`, `config.RAW_BULK_DIR`, `config.RAW_EXTRACT_DIR`
- Produces: `auth_headers() -> dict`, `list_available_files(headers: dict) -> dict`, `pick_la_files(files: dict) -> list[str]`, `_missing_analysis_authorities(matched: set[str]) -> list[str]`, `_matched_codes(candidates: list[str]) -> set[str]`, `_is_valid_zip(path) -> bool`, `download_one_file(args) -> str`, `download_all(headers: dict, candidates: list[str]) -> None`, `extract_all_zips() -> None`, `ensure_dirs() -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_epc_source.py`:

```python
"""Offline tests for the EPC ingestion source module."""

from __future__ import annotations

import base64
import zipfile

import pytest

from ew_housing_energy_impact.ingestion import epc_source


def test_auth_headers_builds_basic_token(monkeypatch):
    monkeypatch.setenv("EPC_EMAIL", "a@b.com")
    monkeypatch.setenv("EPC_API_KEY", "secret")
    headers = epc_source.auth_headers()
    expected = base64.b64encode(b"a@b.com:secret").decode()
    assert headers == {"Authorization": f"Basic {expected}"}


def test_auth_headers_raises_when_missing(monkeypatch):
    monkeypatch.delenv("EPC_EMAIL", raising=False)
    monkeypatch.delenv("EPC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="EPC_EMAIL"):
        epc_source.auth_headers()


def test_is_valid_zip_rejects_truncated(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip at all")
    assert epc_source._is_valid_zip(bad) is False


def test_is_valid_zip_accepts_real_zip(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("certificates.csv", "LMK_KEY\n1\n")
    assert epc_source._is_valid_zip(good) is True
```

**Note on `auth_headers`:** it calls `load_dotenv()`, which does **not** overwrite existing environment variables by default, so `monkeypatch.setenv` wins even if a real `.env` exists.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_epc_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ew_housing_energy_impact.ingestion'`

- [ ] **Step 3: Create the package and move the functions**

Create `src/ew_housing_energy_impact/ingestion/__init__.py`:

```python
"""EPC ingestion: source download, S3 upload, and warehouse load."""
```

Create `src/ew_housing_energy_impact/ingestion/epc_source.py` by **moving** — not copying — these functions verbatim from `scripts/pipeline/epc_download_merge_fast.py`: `ensure_dirs`, `auth_headers`, `list_available_files`, `pick_year_from_name`, `_is_valid_zip`, `download_one_file`, `download_all`, `extract_all_zips`, `pick_la_files`, `_missing_analysis_authorities`, `_matched_codes`.

Carry over the module constants they depend on (`TIMEOUT = 120`, `MAX_WORKERS = 4`) and these imports:

```python
"""Download, validate and extract EPC bulk source files."""

from __future__ import annotations

import base64
import concurrent.futures as futures
import glob
import os
import zipfile

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    ANALYSIS_AUTHORITIES,
    CODE_TO_ANALYSIS_AUTHORITY,
    RAW_BULK_DIR,
    RAW_EXTRACT_DIR,
    SOUTH_WEST_EPC_CODES,
)

TIMEOUT = 120
MAX_WORKERS = 4
```

`ensure_dirs` currently also creates `EPC_MERGED_CSV.parent`. Drop that — the merged CSV is Phase B's problem, not ingestion's. The new body is:

```python
def ensure_dirs() -> None:
    """Make sure raw download and extraction folders exist."""
    for d in (RAW_BULK_DIR, RAW_EXTRACT_DIR):
        os.makedirs(d, exist_ok=True)
```

- [ ] **Step 4: Keep the old module working via re-import**

In `scripts/pipeline/epc_download_merge_fast.py`, delete the moved function bodies and replace them with a re-import so `main()` and the Prefect task keep working until Phase C:

```python
from ew_housing_energy_impact.ingestion.epc_source import (  # noqa: F401
    _is_valid_zip,
    _matched_codes,
    _missing_analysis_authorities,
    auth_headers,
    download_all,
    download_one_file,
    ensure_dirs,
    extract_all_zips,
    list_available_files,
    pick_la_files,
    pick_year_from_name,
)
```

Keep `merge_core_columns()` and `main()` where they are. `main()` must additionally call `os.makedirs(EPC_MERGED_CSV.parent, exist_ok=True)` before merging, since `ensure_dirs` no longer does.

- [ ] **Step 5: Update the geography test import**

In `tests/test_geography.py`, change line 13 only:

```python
from ew_housing_energy_impact.ingestion import epc_source as dl
```

Change nothing else in that file. All 9 assertions must pass unmodified.

- [ ] **Step 6: Run the full suite**

Run: `make ci`
Expected: ruff clean, format clean, **72 passed** (68 existing + 4 new).

If any `test_geography.py` assertion fails, **stop and investigate** — do not adjust the assertion.

- [ ] **Step 7: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/ tests/test_epc_source.py \
        tests/test_geography.py scripts/pipeline/epc_download_merge_fast.py
git commit -m "Move EPC download functions into ingestion package"
```

---

### Task 2: Enumerate extracted sources including recommendations

**Files:**
- Modify: `src/ew_housing_energy_impact/ingestion/epc_source.py`
- Test: `tests/test_epc_source.py`

**Interfaces:**
- Produces: `ExtractedSource` (NamedTuple with fields `la_code: str`, `kind: str`, `path: Path`), `iter_extracted_sources(root: Path | None = None) -> list[ExtractedSource]`, `la_code_from_dirname(name: str) -> str | None`

`kind` is `"certificates"` or `"recommendations"` — these become both the S3 path segment and the raw table name.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_epc_source.py`:

```python
def test_la_code_from_dirname_extracts_code():
    assert epc_source.la_code_from_dirname("domestic-E06000023-Bristol-City-of") == "E06000023"
    assert epc_source.la_code_from_dirname("readme") is None
    assert epc_source.la_code_from_dirname("domestic") is None


def test_iter_extracted_sources_finds_both_kinds(tmp_path):
    d = tmp_path / "domestic-E06000023-Bristol-City-of"
    d.mkdir()
    (d / "certificates.csv").write_text("LMK_KEY\n1\n")
    (d / "recommendations.csv").write_text("LMK_KEY\n1\n")
    (d / "columns.csv").write_text("ignored\n")

    found = epc_source.iter_extracted_sources(tmp_path)

    assert {(s.la_code, s.kind) for s in found} == {
        ("E06000023", "certificates"),
        ("E06000023", "recommendations"),
    }


def test_iter_extracted_sources_skips_missing_recommendations(tmp_path):
    d = tmp_path / "domestic-E07000041-Exeter"
    d.mkdir()
    (d / "certificates.csv").write_text("LMK_KEY\n1\n")

    found = epc_source.iter_extracted_sources(tmp_path)

    assert [(s.la_code, s.kind) for s in found] == [("E07000041", "certificates")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_epc_source.py -k extracted -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'la_code_from_dirname'`

- [ ] **Step 3: Implement**

Add to `epc_source.py` (add `from pathlib import Path` and `from typing import NamedTuple` to the imports):

```python
SOURCE_KINDS = ("certificates", "recommendations")


class ExtractedSource(NamedTuple):
    """One extracted CSV belonging to one local authority."""

    la_code: str
    kind: str
    path: Path


def la_code_from_dirname(name: str) -> str | None:
    """Pull the GSS code out of an extracted folder name.

    'domestic-E06000023-Bristol-City-of' -> 'E06000023'
    """
    parts = name.split("-")
    if len(parts) < 2:
        return None
    return parts[1] or None


def iter_extracted_sources(root: Path | None = None) -> list[ExtractedSource]:
    """List every certificates/recommendations CSV under the extract root.

    Recommendations are optional: some authorities ship certificates only, and a
    missing recommendations file is not an error.
    """
    base = Path(root) if root is not None else Path(RAW_EXTRACT_DIR)
    sources: list[ExtractedSource] = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        la_code = la_code_from_dirname(folder.name)
        if la_code is None:
            continue
        for kind in SOURCE_KINDS:
            csv_path = folder / f"{kind}.csv"
            if csv_path.exists():
                sources.append(ExtractedSource(la_code, kind, csv_path))
    return sources
```

- [ ] **Step 4: Run tests**

Run: `make ci`
Expected: **75 passed**.

- [ ] **Step 5: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/epc_source.py tests/test_epc_source.py
git commit -m "Enumerate extracted certificates and recommendations per authority"
```

---

### Task 3: S3 uploader with content-hash skip

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/s3_upload.py`
- Test: `tests/test_s3_upload.py`

**Interfaces:**
- Consumes: `ExtractedSource` from Task 2
- Produces: `s3_key(kind: str, la_code: str) -> str`, `local_md5(path: Path) -> str`, `needs_upload(client, bucket: str, key: str, path: Path) -> bool`, `upload_source(client, bucket: str, source: ExtractedSource) -> bool`, `upload_all(client, bucket: str, sources: list[ExtractedSource]) -> dict[str, int]`

`upload_source` returns `True` if it uploaded, `False` if it skipped. `upload_all` returns `{"uploaded": n, "skipped": n}`.

**Why stored metadata rather than ETag:** S3 sets ETag to the MD5 **only for single-part uploads**. boto3's `upload_file` switches to multipart above 8 MB and then the ETag looks like `"<hash>-<partcount>"`, which never equals the file's MD5. Certificate files exceed 8 MB, so comparing against ETag would re-upload everything every run. We store our own MD5 in object metadata instead, which is correct regardless of transfer size.

- [ ] **Step 1: Write the failing test**

Create `tests/test_s3_upload.py`:

```python
"""Offline tests for the S3 uploader using a fake client."""

from __future__ import annotations

from pathlib import Path

import pytest

from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource
from ew_housing_energy_impact.ingestion import s3_upload


class FakeClientError(Exception):
    """Stands in for botocore.exceptions.ClientError."""

    def __init__(self):
        self.response = {"Error": {"Code": "404"}}


class FakeS3:
    """Minimal in-memory S3 double."""

    def __init__(self, existing: dict[str, str] | None = None):
        self.existing = existing or {}
        self.uploads: list[tuple[str, str]] = []

    def head_object(self, Bucket: str, Key: str):
        if Key not in self.existing:
            raise FakeClientError()
        return {"Metadata": {"content-md5": self.existing[Key]}}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self.uploads.append((Bucket, Key))
        self.existing[Key] = ExtraArgs["Metadata"]["content-md5"]


@pytest.fixture(autouse=True)
def _patch_client_error(monkeypatch):
    monkeypatch.setattr(s3_upload, "ClientError", FakeClientError)


def _source(tmp_path: Path, kind: str = "certificates") -> ExtractedSource:
    p = tmp_path / f"{kind}.csv"
    p.write_text("LMK_KEY\n1\n")
    return ExtractedSource("E06000023", kind, p)


def test_s3_key_has_no_timestamp():
    key = s3_upload.s3_key("certificates", "E06000023")
    assert key == "raw/certificates/E06000023/certificates.csv"


def test_uploads_when_object_absent(tmp_path):
    client = FakeS3()
    assert s3_upload.upload_source(client, "b", _source(tmp_path)) is True
    assert client.uploads == [("b", "raw/certificates/E06000023/certificates.csv")]


def test_skips_when_hash_matches(tmp_path):
    src = _source(tmp_path)
    digest = s3_upload.local_md5(src.path)
    key = s3_upload.s3_key(src.kind, src.la_code)
    client = FakeS3(existing={key: digest})

    assert s3_upload.upload_source(client, "b", src) is False
    assert client.uploads == []


def test_reuploads_when_content_changed(tmp_path):
    src = _source(tmp_path)
    key = s3_upload.s3_key(src.kind, src.la_code)
    client = FakeS3(existing={key: "stale-hash"})

    assert s3_upload.upload_source(client, "b", src) is True
    assert client.uploads == [("b", key)]


def test_upload_all_counts_both_outcomes(tmp_path):
    a = _source(tmp_path, "certificates")
    b = _source(tmp_path, "recommendations")
    key_a = s3_upload.s3_key(a.kind, a.la_code)
    client = FakeS3(existing={key_a: s3_upload.local_md5(a.path)})

    result = s3_upload.upload_all(client, "b", [a, b])

    assert result == {"uploaded": 1, "skipped": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_s3_upload.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...ingestion.s3_upload'`

- [ ] **Step 3: Implement**

Create `src/ew_housing_energy_impact/ingestion/s3_upload.py`:

```python
"""Upload extracted EPC sources to S3 at stable, timestamp-free keys."""

from __future__ import annotations

import hashlib
from pathlib import Path

from botocore.exceptions import ClientError

from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource

_READ_BLOCK = 8 << 20  # 8 MB
_MD5_METADATA_KEY = "content-md5"


def s3_key(kind: str, la_code: str) -> str:
    """Stable key for one authority's source file.

    Contains no timestamp or run id, so re-running overwrites in place rather
    than accumulating one copy per run.
    """
    return f"raw/{kind}/{la_code}/{kind}.csv"


def local_md5(path: Path) -> str:
    """MD5 of a file, read in blocks so large certificates do not load in full."""
    digest = hashlib.md5()  # noqa: S324 - content addressing, not security
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def needs_upload(client, bucket: str, key: str, path: Path) -> bool:
    """True when the object is absent or its stored hash differs from the file.

    Compares against an MD5 we store in object metadata rather than the ETag:
    multipart uploads give an ETag that is not the file's MD5, which would make
    every run look like a change.
    """
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return True
    stored = head.get("Metadata", {}).get(_MD5_METADATA_KEY)
    return stored != local_md5(path)


def upload_source(client, bucket: str, source: ExtractedSource) -> bool:
    """Upload one source unless S3 already holds identical content."""
    key = s3_key(source.kind, source.la_code)
    if not needs_upload(client, bucket, key, source.path):
        return False
    client.upload_file(
        Filename=str(source.path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"Metadata": {_MD5_METADATA_KEY: local_md5(source.path)}},
    )
    return True


def upload_all(client, bucket: str, sources: list[ExtractedSource]) -> dict[str, int]:
    """Upload every source, returning counts of uploaded and skipped files."""
    result = {"uploaded": 0, "skipped": 0}
    for source in sources:
        key = "uploaded" if upload_source(client, bucket, source) else "skipped"
        result[key] += 1
    return result
```

- [ ] **Step 4: Add boto3 to dependencies**

In `pyproject.toml`, add `"boto3>=1.35"` to the `dependencies` list. Add the same line to `requirements.txt`.

- [ ] **Step 5: Run tests**

Run: `pip install -e . && make ci`
Expected: **80 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/s3_upload.py tests/test_s3_upload.py \
        pyproject.toml requirements.txt
git commit -m "Add S3 uploader with content-hash skip"
```

---

### Task 4: Snowflake DDL scripts

**Files:**
- Create: `snowflake/01_setup.sql`, `snowflake/02_storage_integration.sql`, `snowflake/03_stage_and_format.sql`, `snowflake/README.md`

**Interfaces:**
- Produces: database `epc`; schemas `raw`, `staging`, `marts`, `snapshots`, `ai`; warehouse `epc_wh`; role `epc_dbt_role`; integration `epc_s3_int`; stage `epc.raw.epc_raw_stage`; file format `epc.raw.csv_format`

These are applied by hand in the Snowflake console, in numeric order. They are not executed by tests.

- [ ] **Step 1: Write `snowflake/01_setup.sql`**

```sql
-- Warehouse, database, schemas and role. Safe to re-run.
USE ROLE ACCOUNTADMIN;

CREATE WAREHOUSE IF NOT EXISTS epc_wh
  WAREHOUSE_SIZE = XSMALL
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS epc;

CREATE SCHEMA IF NOT EXISTS epc.raw;
CREATE SCHEMA IF NOT EXISTS epc.staging;
CREATE SCHEMA IF NOT EXISTS epc.marts;
CREATE SCHEMA IF NOT EXISTS epc.snapshots;
CREATE SCHEMA IF NOT EXISTS epc.ai;

CREATE ROLE IF NOT EXISTS epc_dbt_role;

GRANT USAGE, OPERATE ON WAREHOUSE epc_wh TO ROLE epc_dbt_role;
GRANT USAGE ON DATABASE epc TO ROLE epc_dbt_role;
GRANT ALL ON ALL SCHEMAS IN DATABASE epc TO ROLE epc_dbt_role;
GRANT ALL ON FUTURE SCHEMAS IN DATABASE epc TO ROLE epc_dbt_role;
GRANT ALL ON ALL TABLES IN DATABASE epc TO ROLE epc_dbt_role;
GRANT ALL ON FUTURE TABLES IN DATABASE epc TO ROLE epc_dbt_role;
GRANT ALL ON ALL VIEWS IN DATABASE epc TO ROLE epc_dbt_role;
GRANT ALL ON FUTURE VIEWS IN DATABASE epc TO ROLE epc_dbt_role;

-- Replace with the Snowflake login that dbt and the loader will use.
GRANT ROLE epc_dbt_role TO USER IDENTIFIER('<your_snowflake_user>');
```

- [ ] **Step 2: Write `snowflake/02_storage_integration.sql`**

```sql
-- Storage integration. Run this, then DESCRIBE, then edit the IAM trust policy,
-- as documented in snowflake/README.md. Single quotes only: Snowflake rejects
-- double quotes here.
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE STORAGE INTEGRATION epc_s3_int
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = 'S3'
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::${AWS_ACCOUNT_ID}:role/epc-snowflake-s3-role'
  STORAGE_ALLOWED_LOCATIONS = ('s3://epc-sw-pipeline-dev/raw/');

GRANT USAGE ON INTEGRATION epc_s3_int TO ROLE epc_dbt_role;

-- Copy STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID from this output
-- into the IAM role's trust policy.
DESCRIBE INTEGRATION epc_s3_int;
```

- [ ] **Step 3: Write `snowflake/03_stage_and_format.sql`**

```sql
USE ROLE ACCOUNTADMIN;
USE DATABASE epc;
USE SCHEMA raw;

-- NULL_IF is deliberately minimal. EPC sentinels 'NO DATA!' and 'INVALID!' are
-- preserved verbatim: "surveyor recorded nothing" and "field absent from the
-- form" are different facts, and collapsing both to NULL here would destroy
-- that distinction irreversibly. Staging decides what each sentinel means.
CREATE OR REPLACE FILE FORMAT epc.raw.csv_format
  TYPE = CSV
  COMPRESSION = AUTO
  FIELD_DELIMITER = ','
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  SKIP_HEADER = 1
  EMPTY_FIELD_AS_NULL = TRUE
  NULL_IF = ()
  TRIM_SPACE = FALSE
  ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE;

CREATE OR REPLACE STAGE epc.raw.epc_raw_stage
  STORAGE_INTEGRATION = epc_s3_int
  URL = 's3://epc-sw-pipeline-dev/raw/'
  FILE_FORMAT = epc.raw.csv_format;

-- Preflight. Must list certificates and recommendations for all 14 authorities
-- before any COPY INTO is attempted.
LIST @epc.raw.epc_raw_stage;
```

- [ ] **Step 4: Write `snowflake/README.md`**

````markdown
# Snowflake setup

Run these in the Snowflake console in numeric order. `04_raw_tables.sql` and
`05_copy_into.sql` are generated by the loader — see below.

## Order of operations

1. **`01_setup.sql`** — warehouse, database, schemas, role. Replace
   `<your_snowflake_user>` first.
2. **AWS: create the IAM policy and role.**
   - Policy `epc-s3-read`, least privilege, this bucket only:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": ["s3:GetObject", "s3:GetObjectVersion"],
           "Resource": "arn:aws:s3:::epc-sw-pipeline-dev/raw/*"
         },
         {
           "Effect": "Allow",
           "Action": ["s3:ListBucket"],
           "Resource": "arn:aws:s3:::epc-sw-pipeline-dev",
           "Condition": {"StringLike": {"s3:prefix": ["raw/*"]}}
         }
       ]
     }
     ```
   - Role `epc-snowflake-s3-role` with that policy attached. Give it any
     placeholder trust policy for now; step 4 replaces it.
3. **`02_storage_integration.sql`** — then read the `DESCRIBE` output.
4. **AWS: edit the role's trust policy** with the two values from `DESCRIBE`:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {"AWS": "<STORAGE_AWS_IAM_USER_ARN>"},
         "Action": "sts:AssumeRole",
         "Condition": {
           "StringEquals": {"sts:ExternalId": "<STORAGE_AWS_EXTERNAL_ID>"}
         }
       }
     ]
   }
   ```
5. **`03_stage_and_format.sql`** — the final `LIST` must return files.

## Why the trust is bidirectional

Snowflake's IAM user assumes roles for many customers. Without the external ID,
anyone who knows that user's ARN could write a trust policy naming it and have
Snowflake — a trusted third party — read a bucket they are not entitled to.
That is the confused-deputy problem; the external ID is the shared secret that
scopes the assume-role to one specific integration.

## Region

Snowflake must be on **AWS / eu-west-2** to match the bucket. A different region
works but pays cross-region egress on every load.
````

- [ ] **Step 5: Verify no double quotes crept into the SQL**

Run: `grep -n '"' snowflake/*.sql`
Expected: no output. Snowflake rejects double-quoted string literals in these statements, and this is the single most common setup error.

- [ ] **Step 6: Commit**

```bash
git add snowflake/
git commit -m "Add Snowflake DDL and setup runbook"
```

---

### Task 5: Raw table DDL generation and the COPY INTO loader

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/load.py`
- Test: `tests/test_load.py`

**Interfaces:**
- Consumes: `ExtractedSource` from Task 2
- Produces: `read_header(path: Path) -> list[str]`, `generate_raw_ddl(table: str, columns: list[str]) -> str`, `copy_into_sql(table: str, kind: str) -> str`, `local_row_count(path: Path) -> int`, `load_kind(cursor, kind: str, columns: list[str]) -> int`, `reconcile(actual: dict[str, int], expected: dict[str, int]) -> None`

**Why DDL is generated, not hardcoded:** raw takes every column EPC ships (D6), roughly 90 of them, and the exact list is not known until the data is on disk. Generating from the CSV header keeps the DDL honest and means a schema change upstream surfaces as a diff rather than a silent column drop.

- [ ] **Step 1: Write the failing test**

Create `tests/test_load.py`:

```python
"""Offline tests for raw DDL generation, COPY INTO, and reconciliation."""

from __future__ import annotations

import pytest

from ew_housing_energy_impact.ingestion import load


class FakeCursor:
    """Records statements and replays canned results."""

    def __init__(self, results: list | None = None):
        self.statements: list[str] = []
        self._results = list(results or [])
        self._last = None

    def execute(self, sql: str):
        self.statements.append(sql)
        self._last = self._results.pop(0) if self._results else None
        return self

    def fetchone(self):
        return self._last


def test_read_header_returns_columns(tmp_path):
    p = tmp_path / "certificates.csv"
    p.write_text("LMK_KEY,POSTCODE,TENURE\nA,B,C\n")
    assert load.read_header(p) == ["LMK_KEY", "POSTCODE", "TENURE"]


def test_generate_raw_ddl_types_every_column_as_string():
    ddl = load.generate_raw_ddl("epc.raw.certificates", ["LMK_KEY", "TOTAL_FLOOR_AREA"])
    assert "CREATE OR REPLACE TABLE epc.raw.certificates" in ddl
    assert '"LMK_KEY" STRING' in ddl
    assert '"TOTAL_FLOOR_AREA" STRING' in ddl
    assert "NUMBER" not in ddl and "FLOAT" not in ddl


def test_generate_raw_ddl_rejects_empty_columns():
    with pytest.raises(ValueError, match="no columns"):
        load.generate_raw_ddl("epc.raw.certificates", [])


def test_copy_into_aborts_on_error():
    sql = load.copy_into_sql("epc.raw.certificates", "certificates")
    assert "ON_ERROR = ABORT_STATEMENT" in sql
    assert "CONTINUE" not in sql
    assert "@epc.raw.epc_raw_stage/certificates/" in sql


def test_local_row_count_excludes_header(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("H1,H2\na,b\nc,d\n")
    assert load.local_row_count(p) == 2


def test_local_row_count_handles_header_only(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("H1,H2\n")
    assert load.local_row_count(p) == 0


def test_load_kind_truncates_before_copying():
    cursor = FakeCursor(results=[None, None, None, (5,)])
    rows = load.load_kind(cursor, "certificates", ["LMK_KEY"])

    joined = " ".join(cursor.statements)
    assert "TRUNCATE TABLE epc.raw.certificates" in joined
    assert cursor.statements.index(
        next(s for s in cursor.statements if "TRUNCATE" in s)
    ) < cursor.statements.index(next(s for s in cursor.statements if "COPY INTO" in s))
    assert rows == 5


def test_reconcile_passes_on_match():
    load.reconcile({"certificates": 10}, {"certificates": 10})


def test_reconcile_raises_on_mismatch():
    with pytest.raises(ValueError, match="certificates"):
        load.reconcile({"certificates": 9}, {"certificates": 10})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_load.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...ingestion.load'`

- [ ] **Step 3: Implement**

Create `src/ew_housing_energy_impact/ingestion/load.py`:

```python
"""Create raw tables, load them from the stage, and prove the load."""

from __future__ import annotations

import csv
from pathlib import Path

DATABASE = "epc"
RAW_SCHEMA = "raw"
STAGE = "epc.raw.epc_raw_stage"
FILE_FORMAT = "epc.raw.csv_format"


def raw_table(kind: str) -> str:
    """Fully qualified raw table name for a source kind."""
    return f"{DATABASE}.{RAW_SCHEMA}.{kind}"


def read_header(path: Path) -> list[str]:
    """First row of a CSV, as a list of column names."""
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        return next(csv.reader(handle))


def generate_raw_ddl(table: str, columns: list[str]) -> str:
    """CREATE TABLE with every column typed STRING.

    Raw is a bronze layer: EPC ships 'NO DATA!', 'INVALID!', empty numerics and
    inconsistent dates, so typing here would reject rows at load time — exactly
    what a faithful copy of source must not do. Casting happens in staging with
    TRY_TO_*, where a failed cast becomes a testable NULL rather than a lost row.
    """
    if not columns:
        raise ValueError(f"Cannot generate DDL for {table}: no columns found in header")
    body = ",\n".join(f'  "{name}" STRING' for name in columns)
    return f"CREATE OR REPLACE TABLE {table} (\n{body}\n)"


def copy_into_sql(table: str, kind: str) -> str:
    """COPY INTO for one source kind, failing loudly on any malformed row."""
    return (
        f"COPY INTO {table}\n"
        f"FROM @{STAGE}/{kind}/\n"
        f"FILE_FORMAT = (FORMAT_NAME = '{FILE_FORMAT}')\n"
        f"PATTERN = '.*{kind}[.]csv'\n"
        f"ON_ERROR = ABORT_STATEMENT"
    )


def local_row_count(path: Path) -> int:
    """Data rows in a CSV, excluding the header."""
    with open(path, "rb") as handle:
        lines = sum(1 for _ in handle)
    return max(lines - 1, 0)


def load_kind(cursor, kind: str, columns: list[str]) -> int:
    """Recreate, truncate and load one raw table. Returns its row count.

    TRUNCATE precedes COPY INTO deliberately. Snowflake's COPY load metadata
    expires after 64 days, so relying on its skip-already-loaded behaviour is
    correct for 64 days and silently duplicates rows afterwards. EPC bulk files
    are complete snapshots per authority, so a full refresh is both correct and
    deterministic. TRUNCATE also clears the load metadata, so the COPY reloads
    without needing FORCE.
    """
    table = raw_table(kind)
    cursor.execute(generate_raw_ddl(table, columns))
    cursor.execute(f"TRUNCATE TABLE {table}")
    cursor.execute(copy_into_sql(table, kind))
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def reconcile(actual: dict[str, int], expected: dict[str, int]) -> None:
    """Fail unless warehouse counts match local counts exactly.

    A zero exit from COPY INTO is not evidence of a correct load.
    """
    mismatches = [
        f"{kind}: warehouse={actual.get(kind)} local={count}"
        for kind, count in expected.items()
        if actual.get(kind) != count
    ]
    if mismatches:
        raise ValueError("Row count reconciliation failed — " + "; ".join(mismatches))
```

- [ ] **Step 4: Run tests**

Run: `make ci`
Expected: **89 passed**.

- [ ] **Step 5: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/load.py tests/test_load.py
git commit -m "Add raw DDL generation, COPY INTO loader and reconciliation"
```

---

### Task 6: Wire it together behind `ewhei ingest`

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/pipeline.py`
- Modify: `src/ew_housing_energy_impact/cli.py`
- Modify: `src/ew_housing_energy_impact/settings.py`
- Test: `tests/test_ingestion_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5
- Produces: `ingest_plan(sources: list[ExtractedSource], bucket: str) -> dict`, `assert_all_authorities_present(sources: list[ExtractedSource]) -> None`, `preflight_stage(cursor) -> int`, `run_ingest(s3_client, cursor, bucket: str, sources: list[ExtractedSource], dry_run: bool = False) -> dict`

This task implements three spec §6 requirements that no earlier task covers: the
missing-authority hard fail, the stage preflight, and upload retries.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion_pipeline.py`:

```python
"""Offline tests for the ingestion pipeline wiring."""

from __future__ import annotations

import pytest

from ew_housing_energy_impact.ingestion import pipeline
from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource


def _sources(tmp_path):
    c = tmp_path / "certificates.csv"
    c.write_text("LMK_KEY,POSTCODE\n1,AA1 1AA\n")
    r = tmp_path / "recommendations.csv"
    r.write_text("LMK_KEY,IMPROVEMENT_ID\n1,5\n")
    return [
        ExtractedSource("E06000023", "certificates", c),
        ExtractedSource("E06000023", "recommendations", r),
    ]


def test_dry_run_touches_nothing(tmp_path):
    result = pipeline.run_ingest(None, None, "b", _sources(tmp_path), dry_run=True)
    assert result["status"] == "dry_run"
    assert result["planned_uploads"] == [
        "raw/certificates/E06000023/certificates.csv",
        "raw/recommendations/E06000023/recommendations.csv",
    ]


def test_plan_groups_expected_rows_by_kind(tmp_path):
    plan = pipeline.ingest_plan(_sources(tmp_path), "b")
    assert plan["expected_rows"] == {"certificates": 1, "recommendations": 1}


def test_run_ingest_reconciles_and_raises_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "upload_all", lambda *a, **k: {"uploaded": 2, "skipped": 0})
    monkeypatch.setattr(pipeline, "load_kind", lambda *a, **k: 99)
    monkeypatch.setattr(pipeline, "assert_all_authorities_present", lambda s: None)
    monkeypatch.setattr(pipeline, "preflight_stage", lambda c: 2)

    with pytest.raises(ValueError, match="reconciliation failed"):
        pipeline.run_ingest(object(), object(), "b", _sources(tmp_path))


def test_missing_authority_is_fatal(tmp_path):
    """A hole in the 14-cluster panel must stop the run, not warn."""
    with pytest.raises(ValueError, match="Missing analysis authorities"):
        pipeline.assert_all_authorities_present(_sources(tmp_path))


def test_all_authorities_present_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_missing_analysis_authorities", lambda codes: [])
    pipeline.assert_all_authorities_present(_sources(tmp_path))


def test_preflight_raises_when_stage_empty():
    class EmptyCursor:
        def execute(self, sql):
            return self

        def fetchall(self):
            return []

    with pytest.raises(ValueError, match="Stage is empty"):
        pipeline.preflight_stage(EmptyCursor())


def test_preflight_returns_file_count():
    class TwoFileCursor:
        def execute(self, sql):
            return self

        def fetchall(self):
            return [("raw/certificates/E06000023/certificates.csv",), ("x",)]

    assert pipeline.preflight_stage(TwoFileCursor()) == 2
```

`test_missing_authority_is_fatal` uses fixture data covering one authority out of
fourteen, so the real `_missing_analysis_authorities` reports thirteen missing.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...ingestion.pipeline'`

- [ ] **Step 3: Implement**

Create `src/ew_housing_energy_impact/ingestion/pipeline.py`:

```python
"""Wire source enumeration, S3 upload and warehouse load into one run."""

from __future__ import annotations

from ew_housing_energy_impact.ingestion.epc_source import (
    SOURCE_KINDS,
    ExtractedSource,
    _missing_analysis_authorities,
)
from ew_housing_energy_impact.ingestion.load import (
    STAGE,
    load_kind,
    local_row_count,
    read_header,
    reconcile,
)
from ew_housing_energy_impact.ingestion.s3_upload import s3_key, upload_all


def assert_all_authorities_present(sources: list[ExtractedSource]) -> None:
    """Fail unless every one of the 14 analysis authorities has data.

    A missing authority is a hole in a 14-cluster panel: every downstream
    fixed-effects estimate would be silently wrong rather than visibly broken.
    This is why it raises instead of warning.
    """
    missing = _missing_analysis_authorities({s.la_code for s in sources})
    if missing:
        raise ValueError(f"Missing analysis authorities — refusing to load: {missing}")


def preflight_stage(cursor) -> int:
    """Confirm the external stage is readable before attempting any COPY.

    Surfaces a broken storage integration or IAM trust policy as one clear
    error, rather than as a series of confusing per-table failures.
    """
    rows = cursor.execute(f"LIST @{STAGE}").fetchall()
    if not rows:
        raise ValueError(f"Stage is empty or unreadable: @{STAGE}")
    return len(rows)


def ingest_plan(sources: list[ExtractedSource], bucket: str) -> dict:
    """Describe what a run would do, without doing any of it."""
    expected: dict[str, int] = {}
    for source in sources:
        expected[source.kind] = expected.get(source.kind, 0) + local_row_count(source.path)
    return {
        "bucket": bucket,
        "planned_uploads": [s3_key(s.kind, s.la_code) for s in sources],
        "expected_rows": expected,
        "kinds": [k for k in SOURCE_KINDS if any(s.kind == k for s in sources)],
    }


def run_ingest(
    s3_client,
    cursor,
    bucket: str,
    sources: list[ExtractedSource],
    dry_run: bool = False,
) -> dict:
    """Upload sources to S3, load them into raw, and reconcile the counts."""
    plan = ingest_plan(sources, bucket)
    if dry_run:
        return {"status": "dry_run", **plan}

    if not sources:
        raise ValueError("No extracted sources found — run the download step first")

    assert_all_authorities_present(sources)
    upload_result = upload_all(s3_client, bucket, sources)
    staged_files = preflight_stage(cursor)

    actual: dict[str, int] = {}
    for kind in plan["kinds"]:
        header = read_header(next(s.path for s in sources if s.kind == kind))
        actual[kind] = load_kind(cursor, kind, header)

    reconcile(actual, plan["expected_rows"])

    return {
        "status": "ok",
        "uploads": upload_result,
        "staged_files": staged_files,
        "rows": actual,
    }
```

- [ ] **Step 4: Add the bucket setting**

In `src/ew_housing_energy_impact/settings.py`, add a field to `Settings` and populate it in `get_settings`:

```python
    s3_bucket: str
```

```python
        s3_bucket=os.getenv("EPC_S3_BUCKET", "epc-sw-pipeline-dev"),
```

- [ ] **Step 5: Add the CLI subcommand**

In `cli.py`, add the command function next to `cmd_db_load`:

```python
def cmd_ingest(dry_run: bool = False) -> None:
    from ew_housing_energy_impact.ingestion.epc_source import iter_extracted_sources
    from ew_housing_energy_impact.ingestion.pipeline import run_ingest
    from ew_housing_energy_impact.settings import get_settings

    settings = get_settings()
    sources = iter_extracted_sources()

    if dry_run:
        result = run_ingest(None, None, settings.s3_bucket, sources, dry_run=True)
    else:
        import boto3
        import snowflake.connector
        from botocore.config import Config

        # Spec §6: 3 attempts with exponential backoff on transient S3 errors.
        # botocore's standard retry mode implements exactly that, so configure it
        # rather than hand-rolling a retry loop around upload_file.
        s3_config = Config(retries={"max_attempts": 3, "mode": "standard"})

        connection = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse="epc_wh",
            database="epc",
            schema="raw",
            role="epc_dbt_role",
        )
        try:
            result = run_ingest(
                boto3.client("s3", config=s3_config),
                connection.cursor(),
                settings.s3_bucket,
                sources,
            )
            connection.commit()
        finally:
            connection.close()

    setup_logger().info("Ingest result: %s", result)
```

Add `import os` to the top-level imports if not already present. Register the parser next to the `db-load` parser:

```python
    ingest = sub.add_parser("ingest", help="Upload EPC sources to S3 and load Snowflake raw")
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads and expected row counts without touching S3 or Snowflake",
    )
```

And dispatch it next to the `db-load` branch:

```python
    elif args.cmd == "ingest":
        cmd_ingest(dry_run=args.dry_run)
```

- [ ] **Step 6: Add the dry-run to the smoke target**

In `Makefile`, append to the `smoke` target:

```make
	PYTHONPATH=src:. python -m ew_housing_energy_impact ingest --dry-run
```

- [ ] **Step 7: Run everything**

Run: `make ci && make smoke`
Expected: **96 passed**; `make smoke` prints the ingest plan without error.

Note `ingest --dry-run` returns an empty plan until the EPC data is downloaded. That is correct behaviour, not a failure.

- [ ] **Step 8: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/pipeline.py \
        src/ew_housing_energy_impact/cli.py \
        src/ew_housing_energy_impact/settings.py \
        tests/test_ingestion_pipeline.py Makefile
git commit -m "Wire ingestion behind ewhei ingest"
```

---

### Task 7: Documentation and handoff

**Files:**
- Create: `docs/ingestion.md`
- Modify: `CLAUDE.md`
- Modify: `HANDOFF.md`
- Modify: `tasks/todo.md`

- [ ] **Step 1: Write `docs/ingestion.md`**

Use exactly these sections, each answering the question in its heading:

```markdown
# Ingestion

## What it does
One paragraph: EPC zips -> S3 -> Snowflake raw, with reconciliation.

## Running it
    ewhei ingest --dry-run    # planned keys and expected row counts
    ewhei ingest              # upload, load, reconcile
One-time Snowflake and IAM setup: see snowflake/README.md.

## S3 layout
    raw/certificates/<la_code>/certificates.csv
    raw/recommendations/<la_code>/recommendations.csv
Keys carry no timestamp, so a re-run overwrites in place instead of
accumulating one copy per run.

## Why raw is all STRING
EPC ships 'NO DATA!', 'INVALID!', empty numerics and inconsistent dates.
Typing here would reject rows at load — exactly what a faithful copy of
source must not do. Staging casts with TRY_TO_*, where a failed cast is a
testable NULL rather than a lost row.

## Why sentinels are preserved
'Surveyor recorded nothing' and 'field absent from the form' are different
facts. Nulling both at load destroys that distinction irreversibly.

## Why the load is a full refresh
Snowflake's COPY load metadata expires after 64 days. Relying on its
skip-already-loaded behaviour is correct for 64 days, then silently starts
duplicating rows. EPC bulk files are complete snapshots per authority, so
TRUNCATE then COPY is both correct and deterministic.

## Why reconciliation exists
A zero exit from COPY INTO is not evidence of a correct load. Every run
compares warehouse counts against local line counts and fails on mismatch.

## Failure modes
Table of the seven cases from the spec and what each one means when you see it.
```

- [ ] **Step 2: Update `CLAUDE.md`**

Add `ewhei ingest` to the platform commands block, and add `ingestion/` to the code map with a one-line description per module.

- [ ] **Step 3: Update `HANDOFF.md` and `tasks/todo.md`**

Record Phase A as complete, the verification baseline as the new test count, and Phase B (dbt) as the next action needing its own spec.

- [ ] **Step 4: Final verification**

Run: `make ci && make smoke`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add docs/ingestion.md CLAUDE.md HANDOFF.md tasks/todo.md
git commit -m "Document ingestion layer"
```

---

## Manual verification (requires credentials — not part of CI)

Once `.env` exists and the data is downloaded, in order:

1. `ewhei download` — 14 SW LA zips, ~200–300 MB
2. Confirm `data/raw/ew_extracted/*/recommendations.csv` exists for at least some authorities
3. `aws s3 mb s3://epc-sw-pipeline-dev --region eu-west-2`
4. Apply `snowflake/01`–`03` per `snowflake/README.md`
5. `ewhei ingest --dry-run` — check the planned keys and expected row counts
6. `ewhei ingest` — must succeed **and** reconcile
7. `ewhei ingest` a second time — row counts must be **identical, not doubled**

Step 7 is the real test of D8b. If counts double, the truncate did not run.

## Definition of done (from spec §8)

1. `make ci` passes.
2. All 14 analysis authorities present; a missing one fails the run.
3. `LIST @epc_raw_stage` shows both kinds for all 14 authorities.
4. Row counts reconcile per table.
5. A second run gives identical counts.
6. `HANDOFF.md` reflects the state.
