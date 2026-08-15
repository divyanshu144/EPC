"""Tests for construction-age-band normalisation.

Two defects observed on real data (125,111 South West rows, 2025 archives):

1. ``norm_age_band`` passed bare years through verbatim, so ``AGE_BAND_STD`` had
   40 levels of which 27 were bare years covering 13,338 rows (10.7%). Those
   sparse levels fragment the categorical and are the reason three test rows
   carried categories unseen in training — see tests/test_preprocessor.py.
2. ``simplify_age_band`` matched ``"1900"`` as a substring before checking
   ``"1900-1929"``, so 9,737 rows of 1900-1929 were relabelled "Pre-1900" in
   every EDA figure. The 1900-1929 branch was unreachable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.pipeline.clean_epc import norm_age_band
from scripts.pipeline.report_eda import simplify_age_band


class TestBareYearsMapToBands:
    """A bare construction year must resolve to the band containing it."""

    @pytest.mark.parametrize(
        "year,expected",
        [
            ("1997", "England and Wales: 1996-2002"),
            ("1999", "England and Wales: 1996-2002"),
            ("2005", "England and Wales: 2003-2006"),
            ("2009", "England and Wales: 2007-2011"),
            ("2011", "England and Wales: 2007-2011"),
            ("2012", "England and Wales: 2012-2021"),
            ("2021", "England and Wales: 2012-2021"),
            ("1980", "England and Wales: 1976-1982"),
            ("1990", "England and Wales: 1983-1990"),
            ("1930", "England and Wales: 1930-1949"),
        ],
    )
    def test_year_maps_into_its_band(self, year, expected):
        assert norm_age_band(year) == expected

    @pytest.mark.parametrize("year", ["1800", "1830", "1850", "1899"])
    def test_years_before_1900_collapse(self, year):
        assert norm_age_band(year) == "England and Wales: before 1900"

    @pytest.mark.parametrize("year", ["2022", "2023", "2024", "2025"])
    def test_years_after_the_last_closed_band(self, year):
        """2022-2025 covered 12,367 rows and had no band at all — the vocabulary
        stopped at 2012-2021, so these stayed as bare years."""
        assert norm_age_band(year) == "England and Wales: 2022 onwards"

    def test_boundary_years_land_on_the_correct_side(self):
        assert norm_age_band("1900") == "England and Wales: 1900-1929"
        assert norm_age_band("1929") == "England and Wales: 1900-1929"
        assert norm_age_band("1966") == "England and Wales: 1950-1966"
        assert norm_age_band("1967") == "England and Wales: 1967-1975"


class TestExistingBehaviourPreserved:
    """The original two rewrites and the null handling must not regress."""

    def test_onwards_variants_still_close(self):
        assert norm_age_band("England and Wales: 2012 onwards") == "England and Wales: 2012-2021"
        assert norm_age_band("England and Wales: 2007 onwards") == "England and Wales: 2007-2011"

    @pytest.mark.parametrize(
        "band",
        [
            "England and Wales: before 1900",
            "England and Wales: 1950-1966",
            "England and Wales: 2012-2021",
        ],
    )
    def test_canonical_bands_pass_through_unchanged(self, band):
        assert norm_age_band(band) == band

    def test_missing_becomes_unknown(self):
        assert norm_age_band(None) == "Unknown"
        assert norm_age_band(pd.NA) == "Unknown"

    def test_whitespace_is_stripped(self):
        assert norm_age_band("  1997  ") == "England and Wales: 1996-2002"

    def test_unrecognised_text_survives_rather_than_being_discarded(self):
        """An unknown label is passed through, not silently bucketed. Losing it
        would hide a genuine change in the source vocabulary."""
        assert norm_age_band("Scotland: 1965-1975") == "Scotland: 1965-1975"

    def test_non_year_digits_are_not_treated_as_years(self):
        assert norm_age_band("12345") == "12345"
        assert norm_age_band("197") == "197"


class TestSimplifyAgeBand:
    """Regression tests for the substring-matching bug."""

    def test_1900_1929_is_not_swallowed_by_pre_1900(self):
        """The original checked `"1900" in s` first, so this returned Pre-1900
        for 9,737 rows."""
        assert simplify_age_band("England and Wales: 1900-1929") == "1900-1929"

    def test_before_1900_still_maps_to_pre_1900(self):
        assert simplify_age_band("England and Wales: before 1900") == "Pre-1900"

    @pytest.mark.parametrize(
        "band,expected",
        [
            ("England and Wales: 1930-1949", "1930-1949"),
            ("England and Wales: 1950-1966", "1950-1966"),
            ("England and Wales: 1967-1975", "1967-1975"),
            ("England and Wales: 1976-1982", "1976-1982"),
            ("England and Wales: 1983-1990", "1983-1990"),
            ("England and Wales: 1991-1995", "1991-1995"),
            ("England and Wales: 1996-2002", "1996-2002"),
            ("England and Wales: 2003-2006", "2003-2006"),
            ("England and Wales: 2007-2011", "2007-2011"),
            ("England and Wales: 2012-2021", "2012-2021"),
        ],
    )
    def test_each_band_maps_to_itself(self, band, expected):
        assert simplify_age_band(band) == expected

    def test_post_2021_band_is_grouped(self):
        assert simplify_age_band("England and Wales: 2022 onwards") == "2022+"


class TestLevelReduction:
    """The point of the fix: fewer, denser categorical levels."""

    def test_bare_years_no_longer_create_distinct_levels(self):
        raw = [
            "1997",
            "1999",
            "2005",
            "2023",
            "2025",
            "England and Wales: 1996-2002",
            "England and Wales: 2003-2006",
        ]
        normalised = {norm_age_band(v) for v in raw}
        assert normalised == {
            "England and Wales: 1996-2002",
            "England and Wales: 2003-2006",
            "England and Wales: 2022 onwards",
        }
        assert len(normalised) < len(set(raw))
