"""Tests for mapping the full-load archive schema onto CORE_COLS.

The national full-load archive uses a different column vocabulary from the
monthly archives it replaced. Verified against the real 2020 and 2025 files:

* every name is lowercase, where the monthly archives were uppercase
* ``LMK_KEY`` is now ``certificate_number``
* ``BUILDING_REFERENCE_NUMBER`` is gone; ``uprn`` is the nearest equivalent
* ``COUNTY`` is gone entirely (``region``/``country`` exist instead)
* ``LOW_ENERGY_FIXED_LIGHT_COUNT`` is now
  ``low_energy_fixed_lighting_outlets_count``

The rename that matters is uprn. ``BUILDING_REFERENCE_NUMBER`` is the key
behind the leakage-safe train/test split, so if the mapping is wrong the split
silently stops protecting anything. Evidence gathered before choosing it: 1.7%
null in a 500k sample, 4,037 UPRNs repeating within one year (so it identifies a
dwelling, not a certificate), and 1,884 UPRNs present in both 2012 and 2020 (so
it links the same dwelling across years, which is the property the split needs).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.pipeline.full_load_schema import (
    FULL_LOAD_RENAMES,
    normalise_full_load_columns,
    resolve_usecols,
)
from scripts.pipeline.schemas import CORE_COLS


class TestColumnResolution:
    def test_every_core_col_resolves_or_is_known_absent(self):
        yearly = [c.lower() for c in _yearly_header()]
        for column in CORE_COLS:
            resolved = column.lower() in yearly or column in FULL_LOAD_RENAMES or column == "COUNTY"
            assert resolved, f"{column} has no mapping into the full-load schema"

    def test_renames_are_the_two_expected_ones(self):
        assert FULL_LOAD_RENAMES == {
            "LMK_KEY": "certificate_number",
            "BUILDING_REFERENCE_NUMBER": "uprn",
        }

    def test_resolve_usecols_returns_source_names(self):
        header = ["certificate_number", "uprn", "postcode", "local_authority", "region"]
        usecols, rename_map = resolve_usecols(
            ["LMK_KEY", "BUILDING_REFERENCE_NUMBER", "POSTCODE", "LOCAL_AUTHORITY"], header
        )
        assert set(usecols) == {"certificate_number", "uprn", "postcode", "local_authority"}
        assert rename_map["certificate_number"] == "LMK_KEY"
        assert rename_map["uprn"] == "BUILDING_REFERENCE_NUMBER"
        assert rename_map["postcode"] == "POSTCODE"

    def test_absent_columns_are_omitted_not_invented(self):
        """COUNTY does not exist in the full load. Requesting it must not fail,
        and must not silently map to region, which is a different thing."""
        header = ["certificate_number", "postcode", "region"]
        usecols, rename_map = resolve_usecols(["LMK_KEY", "POSTCODE", "COUNTY"], header)
        assert "region" not in usecols
        assert "COUNTY" not in rename_map.values()


class TestNormalisation:
    def test_frame_columns_come_back_uppercase_and_renamed(self):
        df = pd.DataFrame(
            {
                "certificate_number": ["c1", "c2"],
                "uprn": ["100", "200"],
                "postcode": ["EX1 1AA", "EX2 2BB"],
                "local_authority": ["E07000041", "E07000041"],
            }
        )
        out = normalise_full_load_columns(df)

        assert list(out.columns) == [
            "LMK_KEY",
            "BUILDING_REFERENCE_NUMBER",
            "POSTCODE",
            "LOCAL_AUTHORITY",
        ]
        assert out["LMK_KEY"].tolist() == ["c1", "c2"]
        assert out["BUILDING_REFERENCE_NUMBER"].tolist() == ["100", "200"]

    def test_already_uppercase_frames_pass_through(self):
        """The monthly archives are already in the target vocabulary and must
        not be disturbed, so both archive shapes can coexist."""
        df = pd.DataFrame({"LMK_KEY": ["a"], "POSTCODE": ["EX1 1AA"]})
        out = normalise_full_load_columns(df)
        assert list(out.columns) == ["LMK_KEY", "POSTCODE"]


class TestMissingPropertyIdentifier:
    """1.7% of rows have no UPRN. They must remain usable without ever being
    grouped together — treating them as one property would create leakage
    rather than prevent it."""

    def test_blank_uprns_get_distinct_synthetic_keys(self):
        df = pd.DataFrame(
            {
                "certificate_number": ["c1", "c2", "c3"],
                "uprn": ["100", "", None],
            }
        )
        out = normalise_full_load_columns(df)
        keys = out["BUILDING_REFERENCE_NUMBER"].tolist()

        assert keys[0] == "100"
        assert keys[1] != keys[2], "two unidentified dwellings must not share a key"
        assert all(k for k in keys), "no key may be blank"

    def test_synthetic_keys_are_marked_as_such(self):
        df = pd.DataFrame({"certificate_number": ["c1"], "uprn": [""]})
        out = normalise_full_load_columns(df)
        assert out["BUILDING_REFERENCE_NUMBER"].iloc[0].startswith("NOUPRN:")

    def test_synthetic_keys_never_collide_with_real_uprns(self):
        df = pd.DataFrame({"certificate_number": ["c1", "c2"], "uprn": ["NOUPRN:c1", ""]})
        out = normalise_full_load_columns(df)
        assert out["BUILDING_REFERENCE_NUMBER"].nunique() == 2


def _yearly_header() -> list[str]:
    path = "data/raw/ew_extracted/domestic-2020/certificates.csv"
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return [c.strip() for c in handle.readline().strip().split(",")]
    except FileNotFoundError:
        pytest.skip("full-load archive not extracted in this environment")
