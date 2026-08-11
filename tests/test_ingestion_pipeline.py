"""Offline tests for the ingestion pipeline wiring."""

from __future__ import annotations

import re

import pytest

import config
from ew_housing_energy_impact.ingestion import pipeline
from ew_housing_energy_impact.ingestion.epc_source import ExtractedSource


def _one_code_per_analysis_authority() -> dict[str, str]:
    """One real GSS code per analysis authority, derived from config's SSOT map.

    Deriving from CODE_TO_ANALYSIS_AUTHORITY (rather than hardcoding 14 codes)
    means this test cannot silently drift from the mapping it is meant to guard.
    """
    codes: dict[str, str] = {}
    for code, authority in config.CODE_TO_ANALYSIS_AUTHORITY.items():
        codes.setdefault(authority, code)
    return codes


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


def test_authority_coverage_passes_with_all_fourteen():
    """A complete 14-authority load passes — the case that actually matters."""
    codes = _one_code_per_analysis_authority()
    assert len(codes) == 14
    cursor = StageCursor([(code,) for code in codes.values()])
    assert pipeline.assert_authority_coverage(cursor) == 14


def test_authority_coverage_fails_when_one_authority_dropped():
    """Dropping exactly one of the fourteen authorities is caught, and named."""
    codes = _one_code_per_analysis_authority()
    dropped_authority, _dropped_code = next(iter(codes.items()))
    remaining_rows = [
        (code,) for authority, code in codes.items() if authority != dropped_authority
    ]
    cursor = StageCursor(remaining_rows)
    with pytest.raises(ValueError, match=re.escape(dropped_authority)):
        pipeline.assert_authority_coverage(cursor)


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
