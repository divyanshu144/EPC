"""Tests for JSON-encoded numeric values in the full-load archive.

Three CO2 columns in the full-load archive carry a legacy representation in
early years: instead of ``1.5`` the cell holds ``{"value": 1.5, "quantity":
"tonnes per year"}``. Measured on the real archive:

    2008  23.9%    2012  4.7%    2016+  0%

The distribution is what makes this dangerous. It is concentrated in 2008-2012,
the Pre-GreenDeal baseline that every policy coefficient is measured against.
Coercing with ``errors="coerce"`` would turn a quarter of the baseline's CO2
readings into NaN while later periods lost none, PanelOLS would drop those rows,
and every CO2 policy estimate would be biased by non-random missingness
correlated with the time dimension — with the model running cleanly throughout.

The value is recoverable, so it is parsed rather than discarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pipeline.full_load_schema import (
    JSON_VALUED_COLUMNS,
    coerce_json_valued_numeric,
)


class TestParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"value": 1.5, "quantity": "tonnes per year"}', 1.5),
            ('{"value": 0.9, "quantity": ""}', 0.9),
            ('{"value": 12, "quantity": "kg"}', 12.0),
            ("2.4", 2.4),
            (" 2.4 ", 2.4),
        ],
    )
    def test_values_are_recovered(self, raw, expected):
        out = coerce_json_valued_numeric(pd.Series([raw]))
        assert out.iloc[0] == pytest.approx(expected)

    def test_plain_numbers_are_untouched(self):
        """Later years are already plain; they must not be disturbed."""
        series = pd.Series(["2.4", "3.1", "0.0"])
        out = coerce_json_valued_numeric(series)
        assert out.tolist() == [2.4, 3.1, 0.0]

    def test_mixed_representations_in_one_column(self):
        """A single merged file spans years, so both forms appear together."""
        series = pd.Series(['{"value": 1.5, "quantity": "x"}', "2.4", None])
        out = coerce_json_valued_numeric(series)
        assert out.iloc[0] == pytest.approx(1.5)
        assert out.iloc[1] == pytest.approx(2.4)
        assert pd.isna(out.iloc[2])


class TestGenuinelyMissing:
    """Real gaps must still become NaN. The point is not to salvage everything,
    only to stop discarding values that are present."""

    @pytest.mark.parametrize("raw", ["", "   ", None, np.nan, "NO DATA!", "INVALID!"])
    def test_non_values_become_nan(self, raw):
        out = coerce_json_valued_numeric(pd.Series([raw]))
        assert pd.isna(out.iloc[0])

    def test_malformed_json_becomes_nan_rather_than_raising(self):
        out = coerce_json_valued_numeric(pd.Series(['{"value": ', '{"no_value": 3}']))
        assert out.isna().all()


class TestNoSilentLoss:
    def test_json_rows_are_not_dropped_the_way_coercion_would_drop_them(self):
        """The regression this exists to prevent.

        pd.to_numeric(errors='coerce') nulls every JSON row. On 2008 that is
        23.9% of the baseline period.
        """
        series = pd.Series(['{"value": 1.5, "quantity": "x"}'] * 24 + ["2.4"] * 76)

        naive = pd.to_numeric(series, errors="coerce")
        fixed = coerce_json_valued_numeric(series)

        assert naive.isna().sum() == 24, "sanity: naive coercion loses the JSON rows"
        assert fixed.isna().sum() == 0, "the fix must recover them"
        assert fixed.mean() == pytest.approx((1.5 * 24 + 2.4 * 76) / 100)


class TestAffectedColumns:
    def test_the_three_co2_columns_are_declared(self):
        assert JSON_VALUED_COLUMNS == frozenset(
            {
                "CO2_EMISSIONS_CURRENT",
                "CO2_EMISSIONS_POTENTIAL",
                "CO2_EMISS_CURR_PER_FLOOR_AREA",
            }
        )
