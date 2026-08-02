from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from ew_housing_energy_impact import cli
from ew_housing_energy_impact.monitoring.drift import build_drift_report
from ew_housing_energy_impact.monitoring.reporting import run_monitoring
from ew_housing_energy_impact.monitoring.retraining import recommend_retraining
from ew_housing_energy_impact.orchestration import tasks as orchestration_tasks
from ew_housing_energy_impact.orchestration.flows import planned_task_names


def _reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CURRENT_ENERGY_EFFICIENCY": [70, 72, 71, 73, 70],
            "CO2_EMISSIONS_CURRENT": [2.0, 2.1, 2.2, 2.0, 2.1],
            "BELOW_C_INT": [0, 0, 0, 0, 0],
            "LOG_FLOOR_AREA": [4.1, 4.2, 4.1, 4.3, 4.2],
            "MAIN_FUEL_STD": ["gas", "gas", "gas", "electric", "gas"],
            "ANALYSIS_AUTHORITY": ["Devon", "Devon", "Somerset", "Somerset", "Dorset"],
            "YEAR": [2020, 2020, 2021, 2021, 2021],
        }
    )


def test_numeric_drift_detected():
    reference = _reference_df()
    current = reference.copy()
    current["LOG_FLOOR_AREA"] = [8.0, 8.2, 8.1, 8.3, 8.2]

    report = build_drift_report(reference, current)
    floor_area = next(
        item for item in report["numeric_features"] if item["column"] == "LOG_FLOOR_AREA"
    )

    assert floor_area["alert"] is True
    assert "numeric_relative_mean_shift" in floor_area["triggered_checks"]


def test_categorical_drift_detected():
    reference = _reference_df()
    current = reference.copy()
    current["MAIN_FUEL_STD"] = ["electric", "electric", "electric", "electric", "electric"]

    report = build_drift_report(reference, current)
    fuel = next(
        item for item in report["categorical_features"] if item["column"] == "MAIN_FUEL_STD"
    )

    assert fuel["alert"] is True
    assert "categorical_distribution_shift" in fuel["triggered_checks"]


def test_missingness_drift_detected():
    reference = _reference_df()
    current = reference.copy()
    current.loc[[0, 1, 2], "LOG_FLOOR_AREA"] = None

    report = build_drift_report(reference, current)
    floor_area = next(
        item for item in report["numeric_features"] if item["column"] == "LOG_FLOOR_AREA"
    )

    assert floor_area["alert"] is True
    assert "missingness_shift" in floor_area["triggered_checks"]


def test_target_drift_detected():
    reference = _reference_df()
    current = reference.copy()
    current["CURRENT_ENERGY_EFFICIENCY"] = [35, 36, 34, 37, 35]

    report = build_drift_report(reference, current)
    target = next(
        item for item in report["target_drift"] if item["column"] == "CURRENT_ENERGY_EFFICIENCY"
    )

    assert target["alert"] is True
    assert "numeric_relative_mean_shift" in target["triggered_checks"]


def test_coverage_and_below_c_balance_are_reported_and_recommended():
    reference = _reference_df()
    current = reference.copy()
    current = current[current["ANALYSIS_AUTHORITY"] != "Dorset"].reset_index(drop=True)
    current["YEAR"] = 2025
    current["BELOW_C_INT"] = 1

    report = build_drift_report(reference, current)
    recommendation = recommend_retraining(report)

    assert report["coverage"]["alert"] is True
    assert "authority_coverage_change" in report["coverage"]["triggered_checks"]
    assert "year_range_change" in report["coverage"]["triggered_checks"]
    assert report["coverage"]["below_c_class_balance_shift"] == 1.0
    assert recommendation["should_retrain"] is True
    assert "below_c_class_balance_shift" in recommendation["triggered_checks"]


def test_missing_optional_columns_do_not_crash_monitoring():
    reference = pd.DataFrame({"score": [1.0, 1.1], "fuel": ["gas", "electric"]})
    current = pd.DataFrame({"score": [1.2, 1.3], "fuel": ["gas", "gas"]})

    report = build_drift_report(reference, current)
    recommendation = recommend_retraining(report)

    assert report["summary"]["target_columns_checked"] == 0
    assert "reference_year_min" not in report["coverage"]
    assert recommendation["severity"] in {"low", "medium", "high"}


def test_no_drift_low_severity_no_retraining():
    reference = _reference_df()
    report = build_drift_report(reference, reference.copy())
    recommendation = recommend_retraining(report)

    assert recommendation["should_retrain"] is False
    assert recommendation["severity"] == "low"


def test_retraining_recommendation_triggers_for_multiple_alerts():
    reference = _reference_df()
    current = reference.copy()
    current["CURRENT_ENERGY_EFFICIENCY"] = [40, 41, 39, 42, 40]
    current["LOG_FLOOR_AREA"] = [8.0, 8.2, 8.1, 8.3, 8.2]
    current["MAIN_FUEL_STD"] = ["electric", "electric", "electric", "electric", "electric"]

    report = build_drift_report(reference, current)
    recommendation = recommend_retraining(report, max_feature_alerts=2)

    assert recommendation["should_retrain"] is True
    assert recommendation["severity"] in {"medium", "high"}
    assert recommendation["triggered_checks"]


def test_monitoring_json_and_markdown_reports_are_generated(tmp_path):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "drift_report.json"
    _reference_df().to_csv(reference, index=False)
    changed = _reference_df()
    changed["MAIN_FUEL_STD"] = ["electric"] * len(changed)
    changed.to_csv(current, index=False)

    result = run_monitoring(reference, current, output)

    assert result["status"] == "completed"
    assert output.exists()
    assert output.with_suffix(".md").exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "retraining_recommendation" in payload


def test_monitoring_dry_run_does_not_read_or_write(monkeypatch, tmp_path):
    output = tmp_path / "drift_report.json"

    def fail_read_csv(*args, **kwargs):
        raise AssertionError("dry-run should not read CSV files")

    monkeypatch.setattr(pd, "read_csv", fail_read_csv)

    result = run_monitoring(
        tmp_path / "missing_reference.csv",
        tmp_path / "missing_current.csv",
        output,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert not output.exists()
    assert not output.with_suffix(".md").exists()


def test_monitoring_skips_mlflow_safely_when_disabled(monkeypatch, tmp_path):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    output = tmp_path / "drift_report.json"
    _reference_df().to_csv(reference, index=False)
    _reference_df().to_csv(current, index=False)
    monkeypatch.setenv("ENABLE_MLFLOW", "false")

    result = run_monitoring(reference, current, output)

    assert result["status"] == "completed"
    assert output.exists()


def test_prefect_monitoring_task_skips_when_inputs_are_absent(monkeypatch):
    calls = []

    def fake_run_monitoring_report(**kwargs):
        calls.append(kwargs)
        return {"status": "skipped_missing_inputs"}

    monkeypatch.setattr(
        orchestration_tasks,
        "run_monitoring_report",
        fake_run_monitoring_report,
    )

    result = orchestration_tasks.run_monitoring()

    assert result == "skipped_missing_inputs"
    assert calls == [{"require_inputs": False}]


def test_cli_monitor_dry_run(monkeypatch):
    calls = []

    def fake_run_monitoring(**kwargs):
        calls.append(kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr(
        "ew_housing_energy_impact.monitoring.reporting.run_monitoring",
        fake_run_monitoring,
    )
    monkeypatch.setattr(sys, "argv", ["ewhei", "monitor", "--dry-run"])

    cli.main()

    assert calls == [
        {
            "reference_path": None,
            "current_path": None,
            "output_path": None,
            "dry_run": True,
        }
    ]


def test_orchestration_monitoring_skip_flag_changes_plan():
    plan = planned_task_names(skip_monitoring=True)

    assert "run_monitoring" not in plan


def test_cli_monitor_help_exposes_options(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ewhei", "monitor", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "--reference" in captured.out
    assert "--current" in captured.out
    assert "--output" in captured.out
