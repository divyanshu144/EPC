"""Sensitivity tests for numeric drift detection.

Finding 2 (silent): ``numeric_drift`` computed

    denominator = max(abs(ref_mean), 1.0)

so any column whose mean is below 1 had its *relative* shift computed as an
*absolute* one. Every rate and proportion in the project is such a column,
including BELOW_C — one of the two model targets.

Reproduced: BELOW_C drifting 0.40 -> 0.55 is a 37.5% relative change and was
reported as 0.150 against a 0.20 threshold, raising no alert. The component
whose entire job is noticing that data changed could not notice.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ew_housing_energy_impact.monitoring.drift import numeric_drift


def _rate_frame(rate: float, n: int = 1000) -> pd.DataFrame:
    ones = int(round(rate * n))
    return pd.DataFrame({"BELOW_C_INT": [1] * ones + [0] * (n - ones)})


class TestProportionColumns:
    def test_meaningful_drift_on_a_rate_is_detected(self):
        """0.40 -> 0.55 is a 37.5% relative move on a key target."""
        result = numeric_drift(_rate_frame(0.40), _rate_frame(0.55), ["BELOW_C_INT"])[0]

        assert result["relative_mean_shift"] == pytest.approx(0.375, abs=0.01)
        assert result["alert"] is True
        assert "numeric_relative_mean_shift" in result["triggered_checks"]

    def test_small_drift_on_a_rate_does_not_alert(self):
        """The fix must not make everything alert. 0.40 -> 0.42 is 5%."""
        result = numeric_drift(_rate_frame(0.40), _rate_frame(0.42), ["BELOW_C_INT"])[0]

        assert result["relative_mean_shift"] == pytest.approx(0.05, abs=0.01)
        assert result["alert"] is False

    def test_relative_shift_is_scale_invariant(self):
        """A 50% move must report 0.5 whether the column is a rate or a score.
        The 1.0 floor made this depend on the column's magnitude."""
        rate = numeric_drift(_rate_frame(0.20), _rate_frame(0.30), ["BELOW_C_INT"])[0]

        big_ref = pd.DataFrame({"SCORE": [20.0] * 100})
        big_cur = pd.DataFrame({"SCORE": [30.0] * 100})
        score = numeric_drift(big_ref, big_cur, ["SCORE"])[0]

        assert rate["relative_mean_shift"] == pytest.approx(score["relative_mean_shift"], abs=0.01)


class TestLargeMagnitudeColumnsUnchanged:
    """Columns with means well above 1 behaved correctly before and must still."""

    def test_efficiency_score_drift(self):
        ref = pd.DataFrame({"CURRENT_ENERGY_EFFICIENCY": [60.0] * 100})
        cur = pd.DataFrame({"CURRENT_ENERGY_EFFICIENCY": [78.0] * 100})
        result = numeric_drift(ref, cur, ["CURRENT_ENERGY_EFFICIENCY"])[0]

        assert result["relative_mean_shift"] == pytest.approx(0.30, abs=0.01)
        assert result["alert"] is True

    def test_no_drift_reports_zero(self):
        ref = pd.DataFrame({"CURRENT_ENERGY_EFFICIENCY": [60.0] * 100})
        result = numeric_drift(ref, ref.copy(), ["CURRENT_ENERGY_EFFICIENCY"])[0]

        assert result["relative_mean_shift"] == pytest.approx(0.0)
        assert result["alert"] is False


class TestZeroBaseline:
    """A zero reference mean makes relative change undefined; it must not crash
    or silently pass."""

    def test_zero_to_nonzero_alerts(self):
        ref = pd.DataFrame({"FLAG": [0.0] * 100})
        cur = pd.DataFrame({"FLAG": [1.0] * 100})
        result = numeric_drift(ref, cur, ["FLAG"])[0]

        assert result["alert"] is True

    def test_zero_to_zero_does_not_alert(self):
        ref = pd.DataFrame({"FLAG": [0.0] * 100})
        result = numeric_drift(ref, ref.copy(), ["FLAG"])[0]

        assert result["alert"] is False

    def test_all_missing_column_does_not_crash(self):
        ref = pd.DataFrame({"X": [None] * 10})
        cur = pd.DataFrame({"X": [None] * 10})
        result = numeric_drift(ref, cur, ["X"])[0]

        assert result["relative_mean_shift"] is None
