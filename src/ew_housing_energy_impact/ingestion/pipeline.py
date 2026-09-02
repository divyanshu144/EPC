"""Wire source enumeration, S3 upload and warehouse load into one run."""

from __future__ import annotations

import re

from ew_housing_energy_impact.ingestion.epc_source import (
    SOURCE_KINDS,
    ExtractedSource,
    _missing_analysis_authorities,
)
from ew_housing_energy_impact.ingestion.load import (
    STAGE,
    load_kind,
    local_row_count,
    raw_table,
    read_header,
    reconcile,
)
from ew_housing_energy_impact.ingestion.s3_upload import s3_key, upload_all


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


_PERIOD_SEGMENT_RE = re.compile(r"^\d{4}(?:-\d{2})?$")


def _period_from_stage_path(path: str) -> str | None:
    """Extract the period segment from one staged object's path.

    Stage paths look like 's3://bucket/raw/certificates/2025-09/certificates.csv'
    — the period is the directory immediately above the filename. Returns None
    for anything that does not fit that shape: a row this cannot confidently
    parse should be skipped, not crash the run.
    """
    parts = path.rstrip("/").split("/")
    if len(parts) < 2:
        return None
    candidate = parts[-2]
    return candidate if _PERIOD_SEGMENT_RE.match(candidate) else None


def assert_local_covers_stage(cursor, sources: list[ExtractedSource]) -> None:
    """Fail if the stage holds a period that local sources do not cover.

    `load_kind`'s COPY INTO has no period filter: it loads every period file
    ever uploaded to the stage's S3 prefix, across all past runs — S3 keys are
    stable and nothing deletes them, by design (spec D8). `reconcile()` then
    compares that cumulative warehouse count against `expected_rows`, which
    sums local row counts over only the sources passed into *this* run.

    `iter_extracted_sources` documents that local extracts get pruned to
    reclaim disk. Once that happens, the stage holds more periods than local
    disk does, and `reconcile()` raises a baffling count mismatch for a load
    that is actually correct. Catch that divergence here, with an explanation,
    before it ever reaches `reconcile()` — reconciliation is only meaningful
    when local sources mirror what is staged.
    """
    rows = cursor.execute(f"LIST @{STAGE}").fetchall()
    staged_periods: set[str] = set()
    for row in rows:
        if not row:
            continue
        period = _period_from_stage_path(str(row[0]))
        if period is not None:
            staged_periods.add(period)

    local_periods = {source.period for source in sources}
    missing = sorted(staged_periods - local_periods)
    if missing:
        raise ValueError(
            f"Stage holds periods not present in local sources: {missing}. "
            "Reconciliation compares local row counts against a cumulative "
            "warehouse load — COPY INTO has no period filter, so every period "
            "ever uploaded to the stage is loaded every run (spec D8). Every "
            "staged period must therefore also be present locally for that "
            "comparison to be valid. Re-extract the missing period(s) locally "
            "before running ingest again."
        )


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

    # Upload runs before preflight_stage even though checking the stage first
    # would fail faster on a broken storage integration. LIST @stage on a first
    # run — before anything has ever been uploaded — returns empty too, and
    # would be indistinguishable from a broken integration; there is no way to
    # tell those apart without uploading first. A re-run after fixing a broken
    # integration re-uploads nothing, since upload_all (s3_upload.py) skips any
    # file whose stored content-md5 already matches.
    upload_result = upload_all(s3_client, bucket, sources)
    staged_files = preflight_stage(cursor)
    assert_local_covers_stage(cursor, sources)

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
