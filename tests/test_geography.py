"""Tests for the South West geography mapping and clean-step rollup.

Covers the structural fix that moves EPC download selection to lower-tier/unitary
authorities and rolls them up to the dissertation's 14 analysis authorities.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from scripts.pipeline import epc_download_merge_fast as dl
from scripts.pipeline.clean_epc import roll_up_to_analysis_authority, clean_chunk


# --------------------------------------------------------------------------- #
# Mapping-level invariants
# --------------------------------------------------------------------------- #

def test_no_county_e10_codes_used_for_download():
    """EPC has no bulk file for county councils (E10xxxxxx); none may be used."""
    e10 = [c for c in config.SOUTH_WEST_EPC_CODES if c.startswith("E10")]
    assert e10 == [], f"County (E10) codes must not be used for download: {e10}"
    # And only E06 (unitary) / E07 (lower-tier district) codes are present.
    bad = [c for c in config.SOUTH_WEST_EPC_CODES if not c.startswith(("E06", "E07"))]
    assert bad == [], f"Unexpected non-unitary/non-district codes: {bad}"


def test_bedford_not_present():
    assert "E06000055" not in config.SOUTH_WEST_EPC_CODES


def test_exactly_14_analysis_authorities():
    assert len(config.ANALYSIS_AUTHORITIES) == 14
    assert set(config.ANALYSIS_AUTHORITIES) == config._EXPECTED_ANALYSIS_AUTHORITIES


@pytest.mark.parametrize(
    "code,label,expected",
    [
        # Devon districts -> Devon
        ("E07000041", "Exeter", "Devon"),
        ("E07000040", "East Devon", "Devon"),
        ("E07000047", "West Devon", "Devon"),
        # Gloucestershire districts -> Gloucestershire
        ("E07000078", "Cheltenham", "Gloucestershire"),
        ("E07000081", "Gloucester", "Gloucestershire"),
        # Somerset (pre-2023) districts -> Somerset
        ("E07000187", "Mendip", "Somerset"),
        ("E07000189", "South Somerset", "Somerset"),
        ("E07000246", "Somerset West and Taunton", "Somerset"),
        ("E07000190", "Taunton Deane", "Somerset"),  # pre-2019
        ("E06000066", "Somerset", "Somerset"),       # 2023+ unitary
        # Dorset (pre-2019) districts -> Dorset
        ("E07000050", "North Dorset", "Dorset"),
        ("E07000052", "West Dorset", "Dorset"),
        ("E06000059", "Dorset", "Dorset"),           # 2019+ unitary
        # BCP transition: Bournemouth/Poole (pre-2019) + Christchurch -> BCP
        ("E06000028", "Bournemouth", "Bournemouth, Christchurch and Poole"),
        ("E06000029", "Poole", "Bournemouth, Christchurch and Poole"),
        ("E07000048", "Christchurch", "Bournemouth, Christchurch and Poole"),
        ("E06000058", "Bournemouth, Christchurch and Poole",
         "Bournemouth, Christchurch and Poole"),
        # Plain unitaries
        ("E06000023", "Bristol, City of", "Bristol, City of"),
        ("E06000052", "Cornwall", "Cornwall"),
    ],
)
def test_rollup_by_code_and_label(code, label, expected):
    # By code
    assert config.analysis_authority_for(code=code) == expected
    # By label fallback (code unknown)
    assert config.analysis_authority_for(label=label) == expected


def test_non_south_west_returns_none():
    assert config.analysis_authority_for(code="E09000001") is None       # City of London
    assert config.analysis_authority_for(label="Birmingham") is None


# --------------------------------------------------------------------------- #
# Download selection
# --------------------------------------------------------------------------- #

def test_pick_la_files_selects_by_district_code():
    files = {
        "domestic-E07000041-Exeter.zip": {},                 # Devon district
        "domestic-E07000078-Cheltenham.zip": {},             # Glos district
        "domestic-E06000023-Bristol-City-of.zip": {},        # unitary
        "domestic-E10000008-Devon.zip": {},                  # county -> must NOT match
        "non-domestic-E06000023.zip": {},                    # wrong prefix
        "readme.txt": {},
    }
    picked = dl.pick_la_files(files)
    assert "domestic-E07000041-Exeter.zip" in picked
    assert "domestic-E07000078-Cheltenham.zip" in picked
    assert "domestic-E06000023-Bristol-City-of.zip" in picked
    assert "domestic-E10000008-Devon.zip" not in picked
    assert "non-domestic-E06000023.zip" not in picked


def test_missing_analysis_authorities_detection():
    # Only Devon + Bristol covered -> 12 of 14 missing.
    matched = {"E07000041", "E06000023"}
    missing = dl._missing_analysis_authorities(matched)
    assert "Devon" not in missing
    assert "Bristol, City of" not in missing
    assert "Gloucestershire" in missing
    assert len(missing) == 12


# --------------------------------------------------------------------------- #
# Clean-step rollup (district rows must NOT be dropped)
# --------------------------------------------------------------------------- #

def _raw_row(code: str, label: str) -> dict:
    """A minimal raw EPC certificate row that survives clean_chunk's filters."""
    return {
        "LMK_KEY": "k",
        "BUILDING_REFERENCE_NUMBER": "1",
        "POSTCODE": "EX1 1AA",
        "LOCAL_AUTHORITY": code,
        "LOCAL_AUTHORITY_LABEL": label,
        "LODGEMENT_DATE": "2020-06-01",
        "CURRENT_ENERGY_RATING": "D",
        "CURRENT_ENERGY_EFFICIENCY": 60,
        "POTENTIAL_ENERGY_RATING": "C",
        "POTENTIAL_ENERGY_EFFICIENCY": 75,
        "ENERGY_CONSUMPTION_CURRENT": 200,
        "CO2_EMISSIONS_CURRENT": 5,
        "TOTAL_FLOOR_AREA": 80,
        "PROPERTY_TYPE": "House",
        "BUILT_FORM": "Detached",
        "CONSTRUCTION_AGE_BAND": "England and Wales: 2007 onwards",
        "MAIN_FUEL": "mains gas",
        "MAINS_GAS_FLAG": "Y",
        "TENURE": "owner-occupied",
        "TRANSACTION_TYPE": "marketed sale",
    }


def test_clean_chunk_keeps_district_rows_and_rolls_up():
    raw = pd.DataFrame([
        _raw_row("E07000041", "Exeter"),                 # -> Devon
        _raw_row("E07000078", "Cheltenham"),             # -> Gloucestershire
        _raw_row("E07000187", "Mendip"),                 # -> Somerset
        _raw_row("E07000050", "North Dorset"),           # -> Dorset
        _raw_row("E07000048", "Christchurch"),           # -> BCP
        _raw_row("E06000023", "Bristol, City of"),       # -> Bristol
        _raw_row("E09000001", "City of London"),         # -> dropped (not SW)
    ])
    out = clean_chunk(raw)

    # The non-SW row is dropped; all six SW rows survive.
    assert len(out) == 6
    assert "City of London" not in set(out["LOCAL_AUTHORITY_LABEL_RAW"])

    # Rollup correctness + raw preservation.
    by_raw = out.set_index("LOCAL_AUTHORITY_LABEL_RAW")
    assert by_raw.loc["Exeter", "ANALYSIS_AUTHORITY"] == "Devon"
    assert by_raw.loc["Cheltenham", "ANALYSIS_AUTHORITY"] == "Gloucestershire"
    assert by_raw.loc["Mendip", "ANALYSIS_AUTHORITY"] == "Somerset"
    assert by_raw.loc["North Dorset", "ANALYSIS_AUTHORITY"] == "Dorset"
    assert by_raw.loc["Christchurch", "ANALYSIS_AUTHORITY"] == "Bournemouth, Christchurch and Poole"

    # LOCAL_AUTHORITY (the FE entity key) is overwritten with the rollup, so the
    # panel resolves to the 14 analysis authorities, not ~30 districts.
    assert by_raw.loc["Exeter", "LOCAL_AUTHORITY"] == "Devon"
    assert by_raw.loc["Exeter", "LOCAL_AUTHORITY_RAW"] == "E07000041"


def test_clean_chunk_label_fallback_when_code_missing():
    # Code blank/unknown but label is a known SW district -> still rolled up.
    row = _raw_row("", "Exeter")
    out = clean_chunk(pd.DataFrame([row]))
    assert len(out) == 1
    assert out.iloc[0]["ANALYSIS_AUTHORITY"] == "Devon"
