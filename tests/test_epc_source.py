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


def test_download_one_file_refuses_with_migration_message():
    """The download path still targets the retired opendatacommunities host —
    it must refuse loudly rather than fail deep inside with a JSONDecodeError."""
    with pytest.raises(NotImplementedError, match="R8/R9"):
        epc_source.download_one_file(("domestic-2025-09.zip", {}))


def test_list_available_files_refuses_with_migration_message():
    from scripts.pipeline import epc_download_merge_fast as legacy

    with pytest.raises(NotImplementedError, match="R8/R9"):
        legacy.list_available_files({})


def test_compat_shim_exposes_the_same_objects():
    """Guards against the shim silently re-implementing instead of re-exporting."""
    from scripts.pipeline import epc_download_merge_fast as legacy

    assert legacy.extract_all_zips is epc_source.extract_all_zips
    assert legacy._is_valid_zip is epc_source._is_valid_zip


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
