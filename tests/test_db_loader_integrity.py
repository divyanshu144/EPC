"""Integrity tests for the analytical database loaders.

Two defects found by review of a15b3c7, both reproduced here before fixing.

Finding 1 (silent corruption): ``groupby(..., dropna=False)`` keeps NaN groups,
and ``str(record["LOCAL_AUTHORITY"])`` then turns a missing authority into the
literal string ``"nan"``. That row is written to the serving database and
surfaces in /authorities and on the dashboard as though it were a real local
authority. Nothing fails.

Finding 6 (opaque crash): the same NaN groups reach ``int(record["YEAR"])``,
which raises ``ValueError: cannot convert float NaN to integer`` — naming no
column, no row, and giving no hint that the cause is missing data.

Both mean the geography rollup or the year derivation upstream has failed. The
loader must say so, not invent an authority or die anonymously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ew_housing_energy_impact.db.loaders import load_authority_year_summary
from ew_housing_energy_impact.db.models import AuthorityYearSummary, Base


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _frame(authorities, years) -> pd.DataFrame:
    n = len(authorities)
    return pd.DataFrame(
        {
            "LOCAL_AUTHORITY": authorities,
            "YEAR": years,
            "CURRENT_ENERGY_EFFICIENCY": [60.0] * n,
            "CO2_EMISSIONS_CURRENT": [2.0] * n,
            "BELOW_C": [True] * n,
            "LMK_KEY": [f"k{i}" for i in range(n)],
        }
    )


class TestMissingGroupingKeys:
    def test_missing_authority_is_rejected_not_written_as_nan(self, session):
        """The corruption case: a NaN authority became the string 'nan'."""
        df = _frame(["Devon", np.nan], [2020, 2020])

        with pytest.raises(ValueError, match="LOCAL_AUTHORITY"):
            load_authority_year_summary(session, df)

        assert session.query(AuthorityYearSummary).count() == 0

    def test_missing_year_error_names_the_column(self, session):
        """The crash case: previously 'cannot convert float NaN to integer'."""
        df = _frame(["Devon", "Devon"], [2020, np.nan])

        with pytest.raises(ValueError, match="YEAR"):
            load_authority_year_summary(session, df)

    def test_error_reports_how_many_rows_are_affected(self, session):
        """A count tells you whether this is one bad row or a broken rollup."""
        df = _frame(["Devon", np.nan, np.nan, np.nan], [2020, 2020, 2020, 2020])

        with pytest.raises(ValueError) as exc:
            load_authority_year_summary(session, df)

        assert "3" in str(exc.value), f"row count missing from message: {exc.value}"

    def test_no_authority_named_nan_can_reach_the_database(self, session):
        """Belt and braces: whatever the loader does, 'nan' must never persist."""
        df = _frame(["Devon", np.nan], [2020, 2020])
        with pytest.raises(ValueError):
            load_authority_year_summary(session, df)

        names = {row.authority for row in session.query(AuthorityYearSummary).all()}
        assert "nan" not in names
        assert "None" not in names


class TestValidDataStillLoads:
    """The fix must not make the loader stricter about anything else."""

    def test_clean_data_loads_normally(self, session):
        df = _frame(["Devon", "Devon", "Cornwall"], [2020, 2021, 2020])
        count = load_authority_year_summary(session, df)
        session.commit()

        assert count == 3
        assert session.query(AuthorityYearSummary).count() == 3

    def test_missing_measure_values_are_still_tolerated(self, session):
        """Only the grouping keys are mandatory. A missing CO2 reading is a
        legitimate gap and must still load as NULL."""
        df = _frame(["Devon", "Devon"], [2020, 2021])
        df.loc[0, "CO2_EMISSIONS_CURRENT"] = np.nan

        count = load_authority_year_summary(session, df)
        session.commit()

        assert count == 2
        assert session.query(AuthorityYearSummary).count() == 2
