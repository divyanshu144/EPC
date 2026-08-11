# EPC Warehouse Ingestion (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land EPC certificates and recommendations in S3 and Snowflake's `raw` schema as a faithful bronze layer, keyed by period, with quote-aware reconciliation proving the load.

**Architecture:** Split the reusable mechanics out of `scripts/pipeline/epc_download_merge_fast.py` into a new `ingestion` subpackage; add a period-keyed S3 uploader and a Snowflake loader that binds columns by name and truncates before copying. All modules take their external client as an argument so they are testable with no network.

**Tech Stack:** Python 3.11+, boto3, snowflake-connector-python, pytest, Snowflake, AWS S3 (eu-west-2).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md`, **as amended 2026-08-11 (§0)**. Decision IDs D1–D16 refer to it.
- **All new unit tests run offline.** CI has no AWS, Snowflake, or EPC credentials. Integration tests go behind `@pytest.mark.integration`, deselected by default.
- **The existing 68 tests must keep passing at every commit**, except the two explicitly retired in Task 1.
- Raw tables are typed **all `STRING`** (D7) and take **every column** (D6): certificates 93, recommendations 7.
- Columns bind **by name** — `PARSE_HEADER = TRUE` + `MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE` (D15). `SKIP_HEADER` and `ERROR_ON_COLUMN_COUNT_MISMATCH` are **removed**.
- Row counting is **quote-aware** via `csv.reader` opened with `newline=""` (D16). Never count lines.
- `ON_ERROR = ABORT_STATEMENT` (D9). Never `CONTINUE`.
- `NULL_IF` stays minimal — `NO DATA!` / `INVALID!` preserved verbatim (D10).
- S3 keys are **period-keyed**, no run timestamps (D8). Raw load is `TRUNCATE` then `COPY INTO` (D8b).
- Auth is `Bearer` from `EPC_BEARER_TOKEN`, read via `settings.get_settings()` (D14).
- Import `ARTIFACTS_DIR` / `REGISTRY_PATH` from `config.py`. No fourth `sys.path` insert.
- Lint/format: `ruff`, line-length 100. Run `make ci` before every commit.
- Branch: `feat/warehouse-ingestion`. Never commit on `main`.

## Available test data

`data copy/raw/ew_bulk/` holds 9 national monthly zips, `domestic-2025-01.zip` … `domestic-2025-09.zip`, each containing `certificates.csv` (93 cols), `recommendations.csv` (7 cols), and metadata. Headers are byte-identical across all nine. **Certificates for 2008–2024 do not exist locally** and need the bearer token plus the yearly endpoint (spec R8/R9) — out of scope here.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ew_housing_energy_impact/ingestion/__init__.py` | Package marker |
| `src/ew_housing_energy_impact/ingestion/epc_source.py` | Bearer auth, download, zip validation, extraction, period parsing, source enumeration |
| `src/ew_housing_energy_impact/ingestion/s3_upload.py` | Period-keyed upload, skip unchanged via stored MD5 |
| `src/ew_housing_energy_impact/ingestion/load.py` | Raw DDL, header diff, `TRUNCATE`/`COPY INTO`, quote-aware counting, reconciliation |
| `src/ew_housing_energy_impact/ingestion/pipeline.py` | Wiring, stage preflight, post-load authority coverage |
| `snowflake/01_setup.sql`, `02_storage_integration.sql`, `03_stage_and_format.sql`, `README.md` | Hand-applied DDL and runbook |
| `tests/test_epc_source.py`, `test_s3_upload.py`, `test_load.py`, `test_ingestion_pipeline.py` | Offline tests |
| `tests/test_geography.py` | **Modify** — imports only; delete 2 tests covering retired functions |
| `scripts/pipeline/epc_download_merge_fast.py` | **Modify** — shim to moved functions, keep merge half working |
| `src/ew_housing_energy_impact/cli.py`, `settings.py`, `Makefile` | **Modify** — `ewhei ingest`, bucket + token settings, smoke dry-run |
| `docs/ingestion.md` | Explains the decisions to a future reader |

---

### Task 1: Ingestion package, bearer auth, period parsing

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/__init__.py`, `epc_source.py`
- Modify: `scripts/pipeline/epc_download_merge_fast.py`, `tests/test_geography.py`
- Test: `tests/test_epc_source.py`

**Interfaces:**
- Consumes: `config.RAW_BULK_DIR`, `config.RAW_EXTRACT_DIR`, `settings.get_settings`
- Produces: `auth_headers() -> dict`, `period_from_name(name: str) -> str | None`, `_is_valid_zip(path) -> bool`, `download_one_file(args) -> str`, `download_all(headers, candidates) -> None`, `extract_all_zips() -> None`, `ensure_dirs() -> None`, `API_BASE: str`

**Retired, not moved (D5/D13):** `pick_la_files`, `_matched_codes`, and Basic-auth `auth_headers`. There are no per-authority files to select and the auth scheme changed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_epc_source.py`:

```python
"""Offline tests for the EPC ingestion source module."""

from __future__ import annotations

import zipfile

import pytest

from ew_housing_energy_impact.ingestion import epc_source


def test_auth_headers_uses_bearer_token(monkeypatch):
    monkeypatch.setenv("EPC_BEARER_TOKEN", "tok123")
    assert epc_source.auth_headers() == {
        "Authorization": "Bearer tok123",
        "Accept": "application/json",
    }


def test_auth_headers_raises_when_token_missing(monkeypatch):
    monkeypatch.delenv("EPC_BEARER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="EPC_BEARER_TOKEN"):
        epc_source.auth_headers()


def test_api_base_points_at_the_new_service():
    """The old host 301-redirects to an HTML page; see spec section 0."""
    assert "api.get-energy-performance-data.communities.gov.uk" in epc_source.API_BASE
    assert "opendatacommunities" not in epc_source.API_BASE


@pytest.mark.parametrize(
    "name,expected",
    [
        ("domestic-2025-09", "2025-09"),
        ("domestic-2025-09.zip", "2025-09"),
        ("domestic-2024", "2024"),
        ("domestic-2024.zip", "2024"),
        ("readme", None),
        ("domestic", None),
        ("domestic-notayear", None),
    ],
)
def test_period_from_name(name, expected):
    assert epc_source.period_from_name(name) == expected


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

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_epc_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ew_housing_energy_impact.ingestion'`

- [ ] **Step 3: Create the package**

Create `src/ew_housing_energy_impact/ingestion/__init__.py`:

```python
"""EPC ingestion: source download, S3 upload, and warehouse load."""
```

Create `src/ew_housing_energy_impact/ingestion/epc_source.py`. **Move verbatim** from `scripts/pipeline/epc_download_merge_fast.py`: `_is_valid_zip`, `download_one_file`, `download_all`, `extract_all_zips`, and `ensure_dirs` (dropping its `EPC_MERGED_CSV.parent` line — the merged CSV is Phase B's concern). Then add the new header:

```python
"""Download, validate and extract EPC bulk source archives.

The EPC service moved to GOV.UK in 2025: the old opendatacommunities host
301-redirects to an HTML landing page, so a request there returns 200 with a
body that is not JSON. Auth changed from Basic to Bearer at the same time, and
the bulk endpoint now serves one national archive split by period rather than
one archive per local authority. See spec section 0.
"""

from __future__ import annotations

import concurrent.futures as futures
import glob
import os
import re
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import RAW_BULK_DIR, RAW_EXTRACT_DIR

API_BASE = "https://api.get-energy-performance-data.communities.gov.uk"
DOMESTIC_CSV_ENDPOINT = f"{API_BASE}/api/files/domestic/csv"
DOMESTIC_CSV_INFO_ENDPOINT = f"{API_BASE}/api/files/domestic/csv/info"

TIMEOUT = 120
MAX_WORKERS = 4

# 'domestic-2025-09' or 'domestic-2024', with or without a .zip suffix.
_PERIOD_RE = re.compile(r"^domestic-(\d{4}(?:-\d{2})?)(?:\.zip)?$")


def auth_headers() -> dict:
    """Bearer auth for the rebuilt EPC service.

    The token comes from the account page on the service website; there is no
    programmatic way to mint one. EPC_EMAIL/EPC_API_KEY are obsolete.
    """
    from ew_housing_energy_impact.settings import get_settings

    token = get_settings().epc_bearer_token
    if not token:
        raise ValueError("Missing EPC_BEARER_TOKEN — get one from your EPC account page")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def period_from_name(name: str) -> str | None:
    """Extract the period from an archive or extracted-folder name.

    'domestic-2025-09' -> '2025-09'; 'domestic-2024.zip' -> '2024'.
    Returns None for anything that is not a domestic archive.
    """
    match = _PERIOD_RE.match(name)
    return match.group(1) if match else None
```

- [ ] **Step 4: Make extraction failure loud**

Replace the `except zipfile.BadZipFile` branch in `extract_all_zips`. It currently prints a warning, deletes the zip and continues, which degrades into a silently missing period discovered much later:

```python
        except zipfile.BadZipFile as exc:
            # The archive passed download validation but will not extract. Delete
            # it so the next run re-downloads, and fail now rather than leaving a
            # hole to be discovered after the load.
            os.remove(zip_path)
            raise RuntimeError(
                f"Corrupt archive removed, re-run to re-download: {zip_path}"
            ) from exc
```

- [ ] **Step 5: Keep the old module importable**

In `scripts/pipeline/epc_download_merge_fast.py`, delete the moved bodies and the now-dead `pick_la_files`, `_matched_codes` and Basic-auth `auth_headers`. Replace with:

```python
from ew_housing_energy_impact.ingestion.epc_source import (  # noqa: F401
    _is_valid_zip,
    download_all,
    download_one_file,
    ensure_dirs,
    extract_all_zips,
)
```

Keep `merge_core_columns()`. `main()` must now call `os.makedirs(EPC_MERGED_CSV.parent, exist_ok=True)` itself, and its LA-mode branch and `strict` parameter are removed — there is no per-authority selection any more.

- [ ] **Step 6: Update and prune `test_geography.py`**

Change the import on line 13 to:

```python
from ew_housing_energy_impact.ingestion import epc_source as dl  # noqa: F401
```

**Delete** these two tests — they cover functions retired in Step 5:
- `test_pick_la_files_selects_by_district_code`
- `test_missing_analysis_authorities_detection`

Change **nothing else**. The remaining seven assertions must pass unmodified; if one needs changing, the split went wrong — stop and investigate.

- [ ] **Step 7: Add the shim identity test**

Append to `tests/test_epc_source.py`:

```python
def test_compat_shim_exposes_the_same_objects():
    """Guards against the shim silently re-implementing instead of re-exporting."""
    from scripts.pipeline import epc_download_merge_fast as legacy

    assert legacy.extract_all_zips is epc_source.extract_all_zips
    assert legacy._is_valid_zip is epc_source._is_valid_zip
```

- [ ] **Step 8: Add the token setting**

In `settings.py`, add `epc_bearer_token: str` to `Settings` and populate it:

```python
        epc_bearer_token=os.getenv("EPC_BEARER_TOKEN", ""),
```

- [ ] **Step 9: Run the suite**

Run: `make ci`
Expected: ruff clean, **79 passed** — 68 existing, minus 2 deleted, plus 13 new
(`test_period_from_name` is parametrized over 7 cases, so it contributes 7).

- [ ] **Step 10: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/ src/ew_housing_energy_impact/settings.py \
        tests/test_epc_source.py tests/test_geography.py \
        scripts/pipeline/epc_download_merge_fast.py
git commit -m "Move ingestion mechanics into package, switch to bearer auth"
```

---

### Task 2: Enumerate extracted sources by period

**Files:**
- Modify: `src/ew_housing_energy_impact/ingestion/epc_source.py`
- Test: `tests/test_epc_source.py`

**Interfaces:**
- Produces: `SOURCE_KINDS: tuple[str, ...]`, `ExtractedSource` (NamedTuple: `period: str`, `kind: str`, `path: Path`), `iter_extracted_sources(root: Path | None = None) -> list[ExtractedSource]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_epc_source.py`:

```python
def test_iter_extracted_sources_finds_both_kinds(tmp_path):
    d = tmp_path / "domestic-2025-09"
    d.mkdir()
    (d / "certificates.csv").write_text("LMK_KEY\n1\n")
    (d / "recommendations.csv").write_text("LMK_KEY\n1\n")
    (d / "columns.csv").write_text("ignored\n")
    (d / "schema.json").write_text("{}")

    found = epc_source.iter_extracted_sources(tmp_path)

    assert {(s.period, s.kind) for s in found} == {
        ("2025-09", "certificates"),
        ("2025-09", "recommendations"),
    }


def test_iter_extracted_sources_skips_metadata_only_folders(tmp_path):
    """2008-2024 folders in `data copy` hold metadata but no certificates."""
    d = tmp_path / "domestic-2015"
    d.mkdir()
    (d / "columns.csv").write_text("x\n")
    (d / "schema.json").write_text("{}")

    assert epc_source.iter_extracted_sources(tmp_path) == []


def test_iter_extracted_sources_ignores_non_domestic_folders(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "certificates.csv").write_text("LMK_KEY\n1\n")

    assert epc_source.iter_extracted_sources(tmp_path) == []


def test_iter_extracted_sources_is_sorted_by_period(tmp_path):
    for period in ("2025-09", "2025-01", "2025-05"):
        d = tmp_path / f"domestic-{period}"
        d.mkdir()
        (d / "certificates.csv").write_text("LMK_KEY\n1\n")

    assert [s.period for s in epc_source.iter_extracted_sources(tmp_path)] == [
        "2025-01",
        "2025-05",
        "2025-09",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_epc_source.py -k extracted -v`
Expected: FAIL — `AttributeError: ... has no attribute 'iter_extracted_sources'`

- [ ] **Step 3: Implement**

Add to `epc_source.py` (add `from typing import NamedTuple` to imports):

```python
SOURCE_KINDS = ("certificates", "recommendations")


class ExtractedSource(NamedTuple):
    """One extracted CSV belonging to one source period."""

    period: str
    kind: str
    path: Path


def iter_extracted_sources(root: Path | None = None) -> list[ExtractedSource]:
    """List every certificates/recommendations CSV under the extract root.

    Folders holding only metadata (columns.csv, schema.json, LICENCE.txt) yield
    nothing — that is the state of the 2008-2024 folders locally, where the CSVs
    were deleted to reclaim disk.
    """
    base = Path(root) if root is not None else Path(RAW_EXTRACT_DIR)
    sources: list[ExtractedSource] = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        period = period_from_name(folder.name)
        if period is None:
            continue
        for kind in SOURCE_KINDS:
            csv_path = folder / f"{kind}.csv"
            if csv_path.exists():
                sources.append(ExtractedSource(period, kind, csv_path))
    return sources
```

- [ ] **Step 4: Run tests**

Run: `make ci`
Expected: **83 passed**.

- [ ] **Step 5: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/epc_source.py tests/test_epc_source.py
git commit -m "Enumerate extracted sources by period"
```

---

### Task 3: Period-keyed S3 uploader

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/s3_upload.py`
- Test: `tests/test_s3_upload.py`

**Interfaces:**
- Produces: `s3_key(kind: str, period: str) -> str`, `local_md5(path: Path) -> str`, `needs_upload(client, bucket, key, path) -> bool`, `upload_source(client, bucket, source: ExtractedSource) -> bool`, `upload_all(client, bucket, sources) -> dict[str, int]`

**Why stored metadata rather than ETag:** S3 sets ETag to the MD5 only for single-part uploads. boto3 goes multipart above 8 MB and the ETag becomes `"<hash>-<parts>"`. Certificate files are ~122 MB, so an ETag comparison would re-upload everything every run. We store our own MD5 in object metadata, which is correct at any size.

- [ ] **Step 1: Write the failing test**

Create `tests/test_s3_upload.py`:

```python
"""Offline tests for the S3 uploader using a fake client."""

from __future__ import annotations

from pathlib import Path

import pytest

from ew_housing_energy_impact.ingestion import s3_upload
from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource


class FakeClientError(Exception):
    """Stands in for botocore.exceptions.ClientError."""


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
    return ExtractedSource("2025-09", kind, p)


def test_s3_key_is_period_keyed_with_no_timestamp():
    assert s3_upload.s3_key("certificates", "2025-09") == (
        "raw/certificates/2025-09/certificates.csv"
    )
    assert s3_upload.s3_key("recommendations", "2024") == (
        "raw/recommendations/2024/recommendations.csv"
    )


def test_uploads_when_object_absent(tmp_path):
    client = FakeS3()
    assert s3_upload.upload_source(client, "b", _source(tmp_path)) is True
    assert client.uploads == [("b", "raw/certificates/2025-09/certificates.csv")]


def test_skips_when_hash_matches(tmp_path):
    src = _source(tmp_path)
    key = s3_upload.s3_key(src.kind, src.period)
    client = FakeS3(existing={key: s3_upload.local_md5(src.path)})

    assert s3_upload.upload_source(client, "b", src) is False
    assert client.uploads == []


def test_reuploads_when_content_changed(tmp_path):
    src = _source(tmp_path)
    key = s3_upload.s3_key(src.kind, src.period)
    client = FakeS3(existing={key: "stale-hash"})

    assert s3_upload.upload_source(client, "b", src) is True
    assert client.uploads == [("b", key)]


def test_upload_all_counts_both_outcomes(tmp_path):
    a = _source(tmp_path, "certificates")
    b = _source(tmp_path, "recommendations")
    key_a = s3_upload.s3_key(a.kind, a.period)
    client = FakeS3(existing={key_a: s3_upload.local_md5(a.path)})

    assert s3_upload.upload_all(client, "b", [a, b]) == {"uploaded": 1, "skipped": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_s3_upload.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...ingestion.s3_upload'`

- [ ] **Step 3: Implement**

Create `src/ew_housing_energy_impact/ingestion/s3_upload.py`:

```python
"""Upload extracted EPC sources to S3 at stable, period-keyed paths."""

from __future__ import annotations

import hashlib
from pathlib import Path

from botocore.exceptions import ClientError

from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource

_READ_BLOCK = 8 << 20  # 8 MB
_MD5_METADATA_KEY = "content-md5"


def s3_key(kind: str, period: str) -> str:
    """Stable key for one period's source file.

    The period is part of the data's identity; a run timestamp is not. Keys
    therefore carry no timestamp, so re-running overwrites in place rather than
    accumulating one copy per run.
    """
    return f"raw/{kind}/{period}/{kind}.csv"


def local_md5(path: Path) -> str:
    """MD5 of a file, read in blocks so a 122 MB certificate file is not loaded whole."""
    digest = hashlib.md5()  # noqa: S324 - content addressing, not security
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def needs_upload(client, bucket: str, key: str, path: Path) -> bool:
    """True when the object is absent or its stored hash differs from the file."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return True
    return head.get("Metadata", {}).get(_MD5_METADATA_KEY) != local_md5(path)


def upload_source(client, bucket: str, source: ExtractedSource) -> bool:
    """Upload one source unless S3 already holds identical content."""
    key = s3_key(source.kind, source.period)
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
        result["uploaded" if upload_source(client, bucket, source) else "skipped"] += 1
    return result
```

- [ ] **Step 4: Add boto3**

Add `"boto3>=1.35"` to `dependencies` in `pyproject.toml` and to `requirements.txt`.

- [ ] **Step 5: Run tests**

Run: `pip install -e . && make ci`
Expected: **88 passed**.

- [ ] **Step 6: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/s3_upload.py tests/test_s3_upload.py \
        pyproject.toml requirements.txt
git commit -m "Add period-keyed S3 uploader with content-hash skip"
```

---

### Task 4: Snowflake DDL and runbook

**Files:**
- Create: `snowflake/01_setup.sql`, `02_storage_integration.sql`, `03_stage_and_format.sql`, `README.md`

Applied by hand in the Snowflake console, in numeric order. Not executed by tests.

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

-- Replace with the Snowflake login the loader and dbt will use.
GRANT ROLE epc_dbt_role TO USER IDENTIFIER('<your_snowflake_user>');
```

- [ ] **Step 2: Write `snowflake/02_storage_integration.sql`**

```sql
-- Single quotes only: Snowflake rejects double-quoted literals here.
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE STORAGE INTEGRATION epc_s3_int
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = 'S3'
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::${AWS_ACCOUNT_ID}:role/epc-snowflake-s3-role'
  STORAGE_ALLOWED_LOCATIONS = ('s3://epc-sw-pipeline-dev/raw/');

GRANT USAGE ON INTEGRATION epc_s3_int TO ROLE epc_dbt_role;

-- Copy STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID into the IAM
-- role's trust policy.
DESCRIBE INTEGRATION epc_s3_int;
```

- [ ] **Step 3: Write `snowflake/03_stage_and_format.sql`**

```sql
USE ROLE ACCOUNTADMIN;
USE DATABASE epc;
USE SCHEMA raw;

-- PARSE_HEADER is what allows MATCH_BY_COLUMN_NAME on CSV, so columns bind by
-- name rather than position. It is mutually exclusive with SKIP_HEADER.
-- Positional binding would be actively dangerous here: every column is STRING,
-- so a reordered export would land postcodes in the rating column silently.
--
-- NULL_IF is deliberately empty. EPC sentinels 'NO DATA!' and 'INVALID!' are
-- preserved verbatim: "surveyor recorded nothing" and "field absent from the
-- form" are different facts, and collapsing both to NULL here would destroy
-- that distinction irreversibly. Staging decides what each sentinel means.
CREATE OR REPLACE FILE FORMAT epc.raw.csv_format
  TYPE = CSV
  COMPRESSION = AUTO
  FIELD_DELIMITER = ','
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  PARSE_HEADER = TRUE
  EMPTY_FIELD_AS_NULL = TRUE
  NULL_IF = ()
  TRIM_SPACE = FALSE;

CREATE OR REPLACE STAGE epc.raw.epc_raw_stage
  STORAGE_INTEGRATION = epc_s3_int
  URL = 's3://epc-sw-pipeline-dev/raw/'
  FILE_FORMAT = epc.raw.csv_format;

-- Preflight. Must list files before any COPY INTO is attempted.
LIST @epc.raw.epc_raw_stage;
```

- [ ] **Step 4: Write `snowflake/README.md`**

````markdown
# Snowflake setup

Run in numeric order. Raw table DDL and `COPY INTO` are generated at runtime by
`ingestion/load.py` from the CSV header — certificates have 93 columns and the
list is not worth hand-maintaining.

## Order of operations

1. **`01_setup.sql`** — replace `<your_snowflake_user>` first.
2. **AWS: create the bucket, policy and role.**
   ```bash
   aws s3 mb s3://epc-sw-pipeline-dev --region eu-west-2
   ```
   Policy `epc-s3-read` — least privilege, this bucket only. Note there is no
   `GetObjectVersion`: we never read a specific version, and leaving versioning
   off keeps stable keys from accumulating noncurrent copies.
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject"],
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
   Role `epc-snowflake-s3-role` with that policy attached, any placeholder trust
   policy for now.
3. **`02_storage_integration.sql`**, then read the `DESCRIBE` output.
4. **AWS: replace the role's trust policy** with the two values from `DESCRIBE`:
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
anyone knowing that user's ARN could write a trust policy naming it and have
Snowflake — a trusted third party — read a bucket they are not entitled to.
That is the confused-deputy problem; the external ID is the shared secret that
scopes the assume-role to one specific integration.

## Region

Snowflake must be on **AWS / eu-west-2** to match the bucket, or every load pays
cross-region egress.
````

- [ ] **Step 5: Verify no double quotes in SQL**

Run: `grep -n '"' snowflake/*.sql`
Expected: no output. Double-quoted literals are the most common Snowflake setup error.

- [ ] **Step 6: Commit**

```bash
git add snowflake/
git commit -m "Add Snowflake DDL and setup runbook"
```

---

### Task 5: Raw DDL, header diff, loader, quote-aware reconciliation

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/load.py`
- Test: `tests/test_load.py`

**Interfaces:**
- Produces: `raw_table(kind) -> str`, `read_header(path) -> list[str]`, `generate_raw_ddl(table, columns) -> str`, `copy_into_sql(table, kind) -> str`, `local_row_count(path) -> int`, `table_columns(cursor, table) -> list[str]`, `header_diff(existing, incoming) -> dict[str, list[str]]`, `load_kind(cursor, kind, columns, allow_schema_change=False) -> int`, `reconcile(actual, expected) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_load.py`:

```python
"""Offline tests for raw DDL, loading, and quote-aware reconciliation."""

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

    def fetchall(self):
        return self._last or []


def test_read_header_returns_columns(tmp_path):
    p = tmp_path / "certificates.csv"
    p.write_text("LMK_KEY,POSTCODE,TENURE\nA,B,C\n")
    assert load.read_header(p) == ["LMK_KEY", "POSTCODE", "TENURE"]


def test_generate_raw_ddl_types_every_column_as_string():
    ddl = load.generate_raw_ddl("epc.raw.certificates", ["LMK_KEY", "TOTAL_FLOOR_AREA"])
    assert "CREATE TABLE IF NOT EXISTS epc.raw.certificates" in ddl
    assert '"LMK_KEY" STRING' in ddl
    assert '"TOTAL_FLOOR_AREA" STRING' in ddl
    assert "NUMBER" not in ddl and "FLOAT" not in ddl


def test_generate_raw_ddl_rejects_empty_columns():
    with pytest.raises(ValueError, match="no columns"):
        load.generate_raw_ddl("epc.raw.certificates", [])


def test_copy_into_binds_by_name_and_aborts_on_error():
    sql = load.copy_into_sql("epc.raw.certificates", "certificates")
    assert "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE" in sql
    assert "ON_ERROR = ABORT_STATEMENT" in sql
    assert "CONTINUE" not in sql
    assert "SKIP_HEADER" not in sql
    assert "@epc.raw.epc_raw_stage/certificates/" in sql


def test_local_row_count_excludes_header(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("H1,H2\na,b\nc,d\n")
    assert load.local_row_count(p) == 2


def test_local_row_count_handles_header_only(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("H1,H2\n")
    assert load.local_row_count(p) == 0


def test_local_row_count_respects_quoted_embedded_newline(tmp_path):
    """Regression test for counting newlines instead of CSV records.

    D6 keeps every column including free-text *_DESCRIPTION fields, which can
    carry newlines inside quoted values. A naive line count reports 3 here, and
    reconciliation would then fail a load that was actually correct.
    """
    p = tmp_path / "c.csv"
    p.write_text(
        'LMK_KEY,WALLS_DESCRIPTION\n'
        '1,"Cavity wall, as built,\ninsulated (assumed)"\n'
        '2,"Solid brick"\n'
    )
    assert load.local_row_count(p) == 2


def test_local_row_count_handles_crlf_with_quoted_newline(tmp_path):
    p = tmp_path / "c.csv"
    p.write_bytes(b'A,B\r\n1,"x\r\ny"\r\n2,"z"\r\n')
    assert load.local_row_count(p) == 2


def test_header_diff_detects_added_and_removed():
    diff = load.header_diff(["A", "B"], ["B", "C"])
    assert diff == {"added": ["C"], "removed": ["A"]}


def test_header_diff_is_empty_when_identical_ignoring_order():
    assert load.header_diff(["A", "B"], ["B", "A"]) == {"added": [], "removed": []}


def test_load_kind_fails_on_header_change():
    cursor = FakeCursor(results=[None, [("A",), ("B",)]])
    with pytest.raises(ValueError, match="Header changed"):
        load.load_kind(cursor, "certificates", ["A", "C"])


def test_load_kind_allows_header_change_when_overridden():
    cursor = FakeCursor(results=[None, [("A",), ("B",)], None, None, None, (7,)])
    assert load.load_kind(cursor, "certificates", ["A", "C"], allow_schema_change=True) == 7


def test_load_kind_truncates_before_copying():
    cursor = FakeCursor(results=[None, [("A",)], None, None, (5,)])
    rows = load.load_kind(cursor, "certificates", ["A"])

    joined = " ".join(cursor.statements)
    assert "TRUNCATE TABLE epc.raw.certificates" in joined
    truncate_at = next(i for i, s in enumerate(cursor.statements) if "TRUNCATE" in s)
    copy_at = next(i for i, s in enumerate(cursor.statements) if "COPY INTO" in s)
    assert truncate_at < copy_at
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

    Generated rather than hand-written because raw takes every column EPC ships
    (93 for certificates) and the list is only knowable from the data.

    Raw is a bronze layer: EPC ships 'NO DATA!', 'INVALID!', empty numerics and
    inconsistent dates, so typing here would reject rows at load — exactly what
    a faithful copy of source must not do. Casting happens in staging with
    TRY_TO_*, where a failed cast becomes a testable NULL rather than a lost row.
    """
    if not columns:
        raise ValueError(f"Cannot generate DDL for {table}: no columns found in header")
    body = ",\n".join(f'  "{name}" STRING' for name in columns)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n{body}\n)"


def copy_into_sql(table: str, kind: str) -> str:
    """COPY INTO binding columns by name, failing loudly on any malformed row.

    MATCH_BY_COLUMN_NAME requires PARSE_HEADER on the file format. Positional
    binding is unsafe here: every column is STRING, so a reordered export would
    load successfully with values in the wrong columns.
    """
    return (
        f"COPY INTO {table}\n"
        f"FROM @{STAGE}/{kind}/\n"
        f"FILE_FORMAT = (FORMAT_NAME = '{FILE_FORMAT}')\n"
        f"MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE\n"
        f"PATTERN = '.*{kind}[.]csv'\n"
        f"ON_ERROR = ABORT_STATEMENT"
    )


def local_row_count(path: Path) -> int:
    """Data rows in a CSV, excluding the header.

    Parses rather than counting newlines. Free-text columns can contain newlines
    inside quoted values, and a line count would then exceed the true record
    count and fail reconciliation on a load that was correct. `newline=""` is
    required: without it, universal-newline translation breaks the csv module's
    own quoted-newline handling.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def table_columns(cursor, table: str) -> list[str]:
    """Column names of an existing table, empty if it does not exist yet."""
    rows = cursor.execute(f"SHOW COLUMNS IN TABLE {table}").fetchall()
    return [row[0] for row in rows] if rows else []


def header_diff(existing: list[str], incoming: list[str]) -> dict[str, list[str]]:
    """Set difference between an existing table's columns and a file's header."""
    return {
        "added": sorted(set(incoming) - set(existing)),
        "removed": sorted(set(existing) - set(incoming)),
    }


def load_kind(
    cursor,
    kind: str,
    columns: list[str],
    allow_schema_change: bool = False,
) -> int:
    """Create if absent, truncate, load one raw table. Returns its row count.

    TRUNCATE precedes COPY INTO deliberately. Snowflake's COPY load metadata
    expires after 64 days, so relying on skip-already-loaded is correct for 64
    days and silently duplicates rows afterwards. EPC archives are complete
    snapshots per period, so a full refresh is both correct and deterministic.
    TRUNCATE also clears the load metadata, so COPY reloads without FORCE.

    A header change fails rather than being absorbed silently: on a full-refresh
    pipeline a dropped source column would vanish from raw and resurface later as
    a confusing staging error.
    """
    table = raw_table(kind)
    cursor.execute(generate_raw_ddl(table, columns))

    existing = table_columns(cursor, table)
    if existing:
        diff = header_diff(existing, columns)
        if (diff["added"] or diff["removed"]) and not allow_schema_change:
            raise ValueError(
                f"Header changed for {table} — added={diff['added']} "
                f"removed={diff['removed']}. Re-run with --allow-schema-change "
                f"to accept this deliberately."
            )

    cursor.execute(f"TRUNCATE TABLE {table}")
    cursor.execute(copy_into_sql(table, kind))
    row = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def reconcile(actual: dict[str, int], expected: dict[str, int]) -> None:
    """Fail unless warehouse counts match local counts exactly.

    A zero exit from COPY INTO is not evidence of a correct load. This compares
    an independent local count against the warehouse; Snowflake's own
    rows_loaded would be Snowflake checking itself and could not detect a
    truncated upload.
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
Expected: **103 passed**.

- [ ] **Step 5: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/load.py tests/test_load.py
git commit -m "Add raw DDL, by-name loading and quote-aware reconciliation"
```

---

### Task 6: Wire it together behind `ewhei ingest`

**Files:**
- Create: `src/ew_housing_energy_impact/ingestion/pipeline.py`
- Modify: `src/ew_housing_energy_impact/cli.py`, `settings.py`, `Makefile`
- Test: `tests/test_ingestion_pipeline.py`

**Interfaces:**
- Produces: `ingest_plan(sources, bucket) -> dict`, `preflight_stage(cursor) -> int`, `loaded_authority_codes(cursor) -> set[str]`, `assert_authority_coverage(cursor) -> int`, `run_ingest(s3_client, cursor, bucket, sources, dry_run=False, allow_schema_change=False) -> dict`

**Authority coverage moved after the load (spec §6, amended).** With national period files there is no per-authority file whose absence we can detect, so coverage is checked against the loaded data.

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
        ExtractedSource("2025-09", "certificates", c),
        ExtractedSource("2025-09", "recommendations", r),
    ]


class StageCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql):
        return self

    def fetchall(self):
        return self._rows


def test_dry_run_touches_nothing(tmp_path):
    result = pipeline.run_ingest(None, None, "b", _sources(tmp_path), dry_run=True)
    assert result["status"] == "dry_run"
    assert result["planned_uploads"] == [
        "raw/certificates/2025-09/certificates.csv",
        "raw/recommendations/2025-09/recommendations.csv",
    ]


def test_plan_groups_expected_rows_by_kind(tmp_path):
    plan = pipeline.ingest_plan(_sources(tmp_path), "b")
    assert plan["expected_rows"] == {"certificates": 1, "recommendations": 1}
    assert plan["periods"] == ["2025-09"]


def test_preflight_raises_when_stage_empty():
    with pytest.raises(ValueError, match="Stage is empty"):
        pipeline.preflight_stage(StageCursor([]))


def test_preflight_returns_file_count():
    assert pipeline.preflight_stage(StageCursor([("a",), ("b",)])) == 2


def test_authority_coverage_fails_when_authorities_missing():
    """Two South West codes only: the other twelve clusters are absent."""
    cursor = StageCursor([("E07000041",), ("E06000023",)])
    with pytest.raises(ValueError, match="Missing analysis authorities"):
        pipeline.assert_authority_coverage(cursor)


def test_authority_coverage_passes_with_all_fourteen(monkeypatch):
    monkeypatch.setattr(pipeline, "_missing_analysis_authorities", lambda codes: [])
    assert pipeline.assert_authority_coverage(StageCursor([("E06000023",)])) == 1


def test_run_ingest_raises_on_reconciliation_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "upload_all", lambda *a, **k: {"uploaded": 2, "skipped": 0})
    monkeypatch.setattr(pipeline, "load_kind", lambda *a, **k: 99)
    monkeypatch.setattr(pipeline, "preflight_stage", lambda c: 2)
    monkeypatch.setattr(pipeline, "assert_authority_coverage", lambda c: 14)

    with pytest.raises(ValueError, match="reconciliation failed"):
        pipeline.run_ingest(object(), object(), "b", _sources(tmp_path))


def test_run_ingest_rejects_empty_source_list():
    with pytest.raises(ValueError, match="No extracted sources"):
        pipeline.run_ingest(object(), object(), "b", [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...ingestion.pipeline'`

- [ ] **Step 3: Implement**

Create `src/ew_housing_energy_impact/ingestion/pipeline.py`:

```python
"""Wire source enumeration, S3 upload and warehouse load into one run."""

from __future__ import annotations

from config import ANALYSIS_AUTHORITIES, CODE_TO_ANALYSIS_AUTHORITY

from ew_housing_energy_impact.ingestion.epc_source import SOURCE_KINDS, ExtractedSource
from ew_housing_energy_impact.ingestion.load import (
    STAGE,
    load_kind,
    local_row_count,
    raw_table,
    read_header,
    reconcile,
)
from ew_housing_energy_impact.ingestion.s3_upload import s3_key, upload_all


def _missing_analysis_authorities(codes: set[str]) -> list[str]:
    """Analysis authorities with no matching local authority code in the data."""
    covered = {CODE_TO_ANALYSIS_AUTHORITY[c] for c in codes if c in CODE_TO_ANALYSIS_AUTHORITY}
    return sorted(set(ANALYSIS_AUTHORITIES) - covered)


def ingest_plan(sources: list[ExtractedSource], bucket: str) -> dict:
    """Describe what a run would do, without doing any of it."""
    expected: dict[str, int] = {}
    for source in sources:
        expected[source.kind] = expected.get(source.kind, 0) + local_row_count(source.path)
    return {
        "bucket": bucket,
        "planned_uploads": [s3_key(s.kind, s.period) for s in sources],
        "expected_rows": expected,
        "periods": sorted({s.period for s in sources}),
        "kinds": [k for k in SOURCE_KINDS if any(s.kind == k for s in sources)],
    }


def preflight_stage(cursor) -> int:
    """Confirm the external stage is readable before attempting any COPY.

    Surfaces a broken storage integration or IAM trust policy as one clear
    error rather than as a series of confusing per-table failures.
    """
    rows = cursor.execute(f"LIST @{STAGE}").fetchall()
    if not rows:
        raise ValueError(f"Stage is empty or unreadable: @{STAGE}")
    return len(rows)


def loaded_authority_codes(cursor) -> set[str]:
    """Distinct LOCAL_AUTHORITY GSS codes present in the loaded certificates."""
    table = raw_table("certificates")
    rows = cursor.execute(f'SELECT DISTINCT "LOCAL_AUTHORITY" FROM {table}').fetchall()
    return {row[0] for row in rows if row and row[0]}


def assert_authority_coverage(cursor) -> int:
    """Fail unless all 14 analysis authorities appear in the loaded data.

    A missing authority is a hole in a 14-cluster panel: every downstream
    fixed-effects estimate would be silently wrong rather than visibly broken.
    Checked after the load because national period files give us no per-authority
    file whose absence we could detect beforehand.
    """
    codes = loaded_authority_codes(cursor)
    missing = _missing_analysis_authorities(codes)
    if missing:
        raise ValueError(f"Missing analysis authorities in loaded data: {missing}")
    return len(codes)


def run_ingest(
    s3_client,
    cursor,
    bucket: str,
    sources: list[ExtractedSource],
    dry_run: bool = False,
    allow_schema_change: bool = False,
) -> dict:
    """Upload sources to S3, load them into raw, and prove the load."""
    plan = ingest_plan(sources, bucket)
    if dry_run:
        return {"status": "dry_run", **plan}

    if not sources:
        raise ValueError("No extracted sources found — run the download step first")

    upload_result = upload_all(s3_client, bucket, sources)
    staged_files = preflight_stage(cursor)

    actual: dict[str, int] = {}
    for kind in plan["kinds"]:
        header = read_header(next(s.path for s in sources if s.kind == kind))
        actual[kind] = load_kind(cursor, kind, header, allow_schema_change=allow_schema_change)

    reconcile(actual, plan["expected_rows"])
    authorities = assert_authority_coverage(cursor)

    return {
        "status": "ok",
        "uploads": upload_result,
        "staged_files": staged_files,
        "rows": actual,
        "authority_codes": authorities,
    }
```

- [ ] **Step 4: Add the bucket setting**

In `settings.py`, add `s3_bucket: str` to `Settings` and populate it:

```python
        s3_bucket=os.getenv("EPC_S3_BUCKET", "epc-sw-pipeline-dev"),
```

- [ ] **Step 5: Add the CLI subcommand**

In `cli.py`, add next to `cmd_db_load`:

```python
def cmd_ingest(dry_run: bool = False, allow_schema_change: bool = False) -> None:
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

        # Spec section 6: 3 attempts with exponential backoff on transient S3
        # errors. botocore's standard retry mode implements exactly that, so
        # configure it rather than hand-rolling a loop around upload_file.
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
                allow_schema_change=allow_schema_change,
            )
            connection.commit()
        finally:
            connection.close()

    setup_logger().info("Ingest result: %s", result)
```

Add `import os` to the top-level imports if absent. Register the parser next to `db-load`:

```python
    ingest = sub.add_parser("ingest", help="Upload EPC sources to S3 and load Snowflake raw")
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads and expected row counts without touching S3 or Snowflake",
    )
    ingest.add_argument(
        "--allow-schema-change",
        action="store_true",
        help="Accept a changed source header instead of failing",
    )
```

And dispatch:

```python
    elif args.cmd == "ingest":
        cmd_ingest(dry_run=args.dry_run, allow_schema_change=args.allow_schema_change)
```

- [ ] **Step 6: Add to the smoke target**

Append to `smoke` in the `Makefile`:

```make
	PYTHONPATH=src:. python -m ew_housing_energy_impact ingest --dry-run
```

- [ ] **Step 7: Run everything**

Run: `make ci && make smoke`
Expected: **111 passed**; `make smoke` prints the ingest plan without error.

An empty plan is correct until `data copy/raw/ew_extracted/` is pointed at or repopulated — not a failure.

- [ ] **Step 8: Commit**

```bash
git add src/ew_housing_energy_impact/ingestion/pipeline.py \
        src/ew_housing_energy_impact/cli.py src/ew_housing_energy_impact/settings.py \
        tests/test_ingestion_pipeline.py Makefile
git commit -m "Wire ingestion behind ewhei ingest"
```

---

### Task 7: Documentation and handoff

**Files:**
- Create: `docs/ingestion.md`
- Modify: `CLAUDE.md`, `HANDOFF.md`, `tasks/todo.md`, `tasks/lessons.md`

- [ ] **Step 1: Write `docs/ingestion.md`**

Use exactly these sections:

```markdown
# Ingestion

## What it does
EPC archives -> S3 -> Snowflake raw, with reconciliation proving the load.

## Running it
    ewhei ingest --dry-run    # planned keys and expected row counts
    ewhei ingest              # upload, load, reconcile, check authorities
One-time Snowflake and IAM setup: see snowflake/README.md.

## S3 layout
    raw/certificates/<period>/certificates.csv
    raw/recommendations/<period>/recommendations.csv
Period is 2025-09 or 2024, taken from the archive name. Keys carry no run
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
Table of the cases in spec section 6 and what each means when you see it.
```

- [ ] **Step 2: Record the API move in `tasks/lessons.md`**

Append an entry covering: `ewhei download` failing with a JSON decode error; the root cause being a 301 from `epc.opendatacommunities.org` to a GOV.UK HTML page, so `raise_for_status()` passes and `.json()` fails; the three consequences (bearer auth, national-not-per-LA, period keys); and the lesson that `raise_for_status()` alone does not prove a JSON response — check `Content-Type`, and treat a cross-host redirect as a signal the endpoint moved.

- [ ] **Step 3: Update `CLAUDE.md`**

Add `ewhei ingest` to the platform commands block and `ingestion/` to the code map. Correct the Download modes table: LA mode no longer exists, and the `--mode year` warning is obsolete — the API serves one national archive.

- [ ] **Step 4: Update `HANDOFF.md` and `tasks/todo.md`**

Record Phase A complete, the new test count as the verification baseline, and Phase B (dbt) as next, needing its own spec. Carry R8/R9 forward: the 2008–2024 backfill still needs the bearer token.

- [ ] **Step 5: Final verification**

Run: `make ci && make smoke`

- [ ] **Step 6: Commit**

```bash
git add docs/ingestion.md CLAUDE.md HANDOFF.md tasks/todo.md tasks/lessons.md
git commit -m "Document ingestion layer"
```

---

## Manual verification (requires credentials — not part of CI)

1. Point `RAW_EXTRACT_DIR` at extracted 2025 monthly data, or extract from `data copy/raw/ew_bulk/`
2. `aws s3 mb s3://epc-sw-pipeline-dev --region eu-west-2`
3. Apply `snowflake/01`–`03` per `snowflake/README.md`
4. `ewhei ingest --dry-run` — check planned keys and expected row counts
5. `ewhei ingest` — must succeed, reconcile, **and** find all 14 authorities
6. `ewhei ingest` again — row counts **identical, not doubled**

Step 6 is the real test of D8b. If counts double, the truncate did not run.

## Definition of done

Spec §8, items 1–8.
