"""Robustness tests for the API fallback path and the Prefect no-op fallback.

Finding 3 (silent): ``orchestration/tasks.py`` falls back to a no-op ``@task``
decorator when Prefect is absent. Prefect is a declared dependency, so that only
happens when the install is broken — and when it does, ``ewhei orchestrate``
runs as a plain sequential loop with no retries and no observability while
reporting success. Nothing announced the downgrade.

Finding 4 (silent): ``_read_feature_csv`` filtered requested columns to those
present. If none matched, ``pd.read_csv(usecols=[])`` returned a (0, 0) frame,
so a mistyped column name produced "no data" rather than an error.

Finding 5 (performance): ``_csv_columns`` re-opened a 56 MB file on every call
with no caching.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestPrefectFallbackIsVisible:
    def test_module_reports_whether_prefect_is_active(self):
        """Callers must be able to tell orchestration from a for-loop."""
        from ew_housing_energy_impact.orchestration import tasks

        assert hasattr(tasks, "PREFECT_AVAILABLE"), (
            "tasks.py must expose PREFECT_AVAILABLE so a degraded run is detectable"
        )
        assert isinstance(tasks.PREFECT_AVAILABLE, bool)

    def test_flow_result_carries_the_orchestration_mode(self):
        """A dry run should say which engine actually executed it."""
        from ew_housing_energy_impact.orchestration.flows import epc_pipeline_flow

        result = epc_pipeline_flow(dry_run=True)
        assert "prefect_available" in result, (
            "flow result must record whether Prefect was in use, so a degraded "
            "run is visible in the output rather than silent"
        )


class TestApiColumnHandling:
    def test_all_requested_columns_missing_raises(self, tmp_path, monkeypatch):
        """A mistyped column name must error, not return an empty result set."""
        import api

        csv = tmp_path / "features.csv"
        pd.DataFrame({"REAL_COLUMN": [1, 2, 3]}).to_csv(csv, index=False)
        monkeypatch.setattr(api, "DATA_PATH", csv)
        api._csv_columns_for.cache_clear()

        with pytest.raises(Exception) as exc:
            api._read_feature_csv(["NO_SUCH_COLUMN", "ALSO_MISSING"])

        message = str(getattr(exc.value, "detail", exc.value))
        assert "NO_SUCH_COLUMN" in message, f"error should name the missing column: {message}"

    def test_partially_missing_columns_still_return_what_exists(self, tmp_path, monkeypatch):
        """Tolerating a genuinely optional column is correct; only a total miss
        is an error."""
        import api

        csv = tmp_path / "features.csv"
        pd.DataFrame({"REAL_COLUMN": [1, 2, 3]}).to_csv(csv, index=False)
        monkeypatch.setattr(api, "DATA_PATH", csv)
        api._csv_columns_for.cache_clear()

        frame = api._read_feature_csv(["REAL_COLUMN", "OPTIONAL_MISSING"])
        assert list(frame.columns) == ["REAL_COLUMN"]
        assert len(frame) == 3


class TestApiColumnCaching:
    def test_repeated_column_lookups_do_not_reread_the_file(self, tmp_path, monkeypatch):
        """The header was re-read from a 56 MB file on every request."""
        import api

        csv = tmp_path / "features.csv"
        pd.DataFrame({"A": [1], "B": [2]}).to_csv(csv, index=False)
        monkeypatch.setattr(api, "DATA_PATH", csv)
        api._csv_columns_for.cache_clear()

        reads = {"n": 0}
        original = pd.read_csv

        def counting_read_csv(*args, **kwargs):
            reads["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", counting_read_csv)

        api._csv_columns()
        api._csv_columns()
        api._csv_columns()

        assert reads["n"] == 1, f"header was read {reads['n']} times; expected 1 (cached)"

    def test_cache_invalidates_when_the_file_changes(self, tmp_path, monkeypatch):
        """A pipeline re-run inside one process must not serve stale columns."""
        import time

        import api

        csv = tmp_path / "features.csv"
        pd.DataFrame({"A": [1]}).to_csv(csv, index=False)
        monkeypatch.setattr(api, "DATA_PATH", csv)
        api._csv_columns_for.cache_clear()

        assert api._csv_columns() == {"A"}

        time.sleep(0.01)
        pd.DataFrame({"A": [1], "NEW_COLUMN": [2]}).to_csv(csv, index=False)

        assert "NEW_COLUMN" in api._csv_columns(), "cache did not invalidate on file change"
