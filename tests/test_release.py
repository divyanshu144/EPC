from __future__ import annotations

import json

import pandas as pd
import pytest

from config import SOUTH_WEST_EPC_AUTHORITIES
from scripts.pipeline.build_release import build_release
from scripts.pipeline.schemas import CORE_COLS
from scripts.pipeline.verify_release import verify_release


def source_rows():
    authorities = {}
    for code, meta in SOUTH_WEST_EPC_AUTHORITIES.items():
        authorities.setdefault(meta["analysis_authority"], (code, meta["epc_label"]))
    rows = []
    for i, (code, label) in enumerate(authorities.values()):
        row = dict.fromkeys(CORE_COLS, "")
        row.update(
            LMK_KEY=f"cert-{i}",
            BUILDING_REFERENCE_NUMBER=f"property-{i}",
            LOCAL_AUTHORITY=code,
            LOCAL_AUTHORITY_LABEL=label,
            LODGEMENT_DATE="2008-10-01",
            CURRENT_ENERGY_EFFICIENCY="60",
            POTENTIAL_ENERGY_EFFICIENCY="80",
            CO2_EMISSIONS_CURRENT='{"value": 2.4}',
            PROPERTY_TYPE="House",
            MAIN_FUEL="electricity (not community)",
            TRANSACTION_TYPE="new dwelling",
            CONSTRUCTION_AGE_BAND="2005",
        )
        rows.append(row)
    return rows


def write_source(root, year, rows):
    path = root / f"domestic-{year}" / "certificates.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_release_retains_missing_2008_area_and_accounts_for_2026(tmp_path):
    rows = source_rows()
    write_source(tmp_path / "raw", 2008, rows)
    newer = {**rows[0], "LMK_KEY": "future", "LODGEMENT_DATE": "2026-01-01"}
    write_source(tmp_path / "raw", 2026, [newer])
    output = tmp_path / "release"
    report = build_release(tmp_path / "raw", output, chunk_size=3)
    data = pd.read_parquet(output / "certificates.parquet")
    assert len(data) == 14
    assert data.YEAR.eq(2008).all()
    assert data.TOTAL_FLOOR_AREA.isna().all()
    assert data.CO2_EMISSIONS_CURRENT.eq(2.4).all()
    assert data.MAIN_FUEL_STD.eq("electricity").all()
    assert data.NEW_BUILD_SHARE.eq(1).all()
    assert report["counts"]["outside_2008_2025"] == 1
    assert report["unique_source_certificates"] == 15
    assert report["unique_identified_properties"] == 14
    assert report["missing_authorities"] == []
    assert not (output / "INCOMPLETE").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    assert verify_release(output, expected_years=[2008])["ok"]
    report["rows"] += 1
    (output / "manifest.json").write_text(json.dumps(report))
    assert not verify_release(output, expected_years=[2008])["ok"]


def test_duplicate_across_files_prevents_complete_release(tmp_path):
    rows = source_rows()
    write_source(tmp_path / "raw", 2008, rows)
    write_source(tmp_path / "raw", 2009, [rows[0]])
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Duplicate certificate"):
        build_release(tmp_path / "raw", output, chunk_size=3)
    assert (output / "INCOMPLETE").exists()
    assert not (output / "manifest.json").exists()


def test_release_does_not_overwrite_existing_directory(tmp_path):
    write_source(tmp_path / "raw", 2008, source_rows())
    output = tmp_path / "release"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_release(tmp_path / "raw", output)


def test_filters_reconcile_and_missing_keys_are_not_identified(tmp_path):
    rows = source_rows()
    rows[0]["BUILDING_REFERENCE_NUMBER"] = ""
    bad = {**rows[1], "LMK_KEY": "bad-area", "TOTAL_FLOOR_AREA": "-10"}
    foreign = {
        **rows[1],
        "LMK_KEY": "foreign",
        "LOCAL_AUTHORITY": "X",
        "LOCAL_AUTHORITY_LABEL": "X",
    }
    write_source(tmp_path / "raw", 2008, rows + [bad, foreign])
    report = build_release(tmp_path / "raw", tmp_path / "release", chunk_size=3)
    assert report["counts"]["outside_bounds_TOTAL_FLOOR_AREA"] == 1
    assert report["counts"]["outside_south_west"] == 1
    assert report["rows"] == 14
    assert report["unidentified_property_rows"] == 1
    assert report["unique_identified_properties"] == 13
