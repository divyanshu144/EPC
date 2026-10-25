from __future__ import annotations

import sys

import pytest


def test_orchestration_flow_imports():
    from ew_housing_energy_impact.orchestration.flows import epc_pipeline_flow

    assert callable(epc_pipeline_flow)


def test_dry_run_returns_expected_plan():
    from ew_housing_energy_impact.orchestration.flows import epc_pipeline_flow

    result = epc_pipeline_flow(dry_run=True)

    assert result["status"] == "dry_run"
    assert result["mode"] == "la"
    assert result["planned_tasks"] == [
        "download_or_merge_epc",
        "clean_south_west",
        "build_features",
        "validate_clean_dataset",
        "generate_eda",
        "run_policy_models",
        "train_ml_models",
        "generate_quality_report",
        "run_monitoring",
        "write_manifest",
        "register_artifacts",
    ]


def test_skip_flags_remove_optional_tasks_from_plan():
    from ew_housing_energy_impact.orchestration.flows import epc_pipeline_flow

    result = epc_pipeline_flow(
        dry_run=True,
        skip_download=True,
        skip_training=True,
        skip_policy=True,
        skip_quality=True,
        skip_monitoring=True,
    )

    assert "download_or_merge_epc" not in result["planned_tasks"]
    assert "train_ml_models" not in result["planned_tasks"]
    assert "run_policy_models" not in result["planned_tasks"]
    assert "generate_quality_report" not in result["planned_tasks"]
    assert "run_monitoring" not in result["planned_tasks"]
    assert "clean_south_west" in result["planned_tasks"]
    assert "validate_clean_dataset" in result["planned_tasks"]


def test_expected_task_names_exist():
    from ew_housing_energy_impact.orchestration import tasks

    for task_name in tasks.PIPELINE_TASK_NAMES:
        assert hasattr(tasks, task_name)


def test_cli_exposes_orchestrate_help(monkeypatch, capsys):
    from ew_housing_energy_impact import cli

    monkeypatch.setattr(sys, "argv", ["ewhei", "orchestrate", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--dry-run" in captured.out
    assert "--skip-download" in captured.out
    assert "--skip-training" in captured.out
    assert "--skip-monitoring" in captured.out


def test_cli_exposes_legacy_tracking_and_mlflow_info(monkeypatch, capsys):
    from ew_housing_energy_impact import cli

    monkeypatch.setattr(sys, "argv", ["ewhei", "legacy-train", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--track" in captured.out

    monkeypatch.setattr(sys, "argv", ["ewhei", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "mlflow-info" in captured.out


def test_cli_accepts_orchestrate_skip_flags(monkeypatch):
    from ew_housing_energy_impact import cli
    from ew_housing_energy_impact.orchestration import flows

    calls = {}

    def fake_flow(**kwargs):
        calls.update(kwargs)
        return {"status": "dry_run", "planned_tasks": []}

    monkeypatch.setattr(flows, "epc_pipeline_flow", fake_flow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ewhei",
            "orchestrate",
            "--mode",
            "la",
            "--skip-download",
            "--skip-training",
            "--skip-monitoring",
            "--dry-run",
        ],
    )

    cli.main()

    assert calls["mode"] == "la"
    assert calls["skip_download"] is True
    assert calls["skip_training"] is True
    assert calls["dry_run"] is True
    assert calls["skip_eda"] is False
    assert calls["skip_policy"] is False
    assert calls["skip_quality"] is False
    assert calls["skip_monitoring"] is True


def test_airflow_placeholder_is_explicitly_non_functional():
    import dags.etl_epc_pipeline as airflow_placeholder

    assert airflow_placeholder.AIRFLOW_DAG_IMPLEMENTED is False
    assert airflow_placeholder.ORCHESTRATOR == "prefect"
    assert not hasattr(airflow_placeholder, "dag")


def test_cli_accepts_db_load_dry_run(monkeypatch):
    from ew_housing_energy_impact import cli
    from ew_housing_energy_impact.db import loaders

    calls = {}

    def fake_refresh_analytics_database(**kwargs):
        calls.update(kwargs)
        return {"planned": ["authority_year_summary"]}

    monkeypatch.setattr(loaders, "refresh_analytics_database", fake_refresh_analytics_database)
    monkeypatch.setattr(sys, "argv", ["ewhei", "db-load", "--dry-run"])

    cli.main()

    assert calls["dry_run"] is True
