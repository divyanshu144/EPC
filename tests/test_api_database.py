from __future__ import annotations

import json
import logging

import pandas as pd
from fastapi.testclient import TestClient

import api
from ew_housing_energy_impact.db.engine import create_tables, get_engine, session_scope
from ew_housing_energy_impact.db.loaders import (
    load_authority_year_summary,
    load_model_metrics,
    load_policy_period_summary,
)


def test_api_health_and_metadata_include_optional_systems(monkeypatch):
    monkeypatch.setenv("ENABLE_DATABASE", "false")

    client = TestClient(api.app)
    health = client.get("/health")
    metadata = client.get("/metadata")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "csv_fallback_available" in health.json()
    assert "paths_checked" in health.json()
    assert metadata.status_code == 200
    assert metadata.json()["data_source_mode"] == "csv"
    assert "optional_systems" in metadata.json()


def test_api_csv_fallback_still_works(tmp_path, monkeypatch):
    data_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "YEAR": [2020, 2020],
            "CURRENT_ENERGY_EFFICIENCY": [60.0, 80.0],
            "CO2_EMISSIONS_CURRENT": [5.0, 3.0],
            "POLICY_PERIOD": ["MEES", "MEES"],
            "BELOW_C": [True, False],
        }
    ).to_csv(data_path, index=False)

    monkeypatch.setattr(api, "DATA_PATH", data_path)
    monkeypatch.setenv("ENABLE_DATABASE", "false")

    client = TestClient(api.app)
    response = client.get("/trends")

    assert response.status_code == 200
    assert response.json() == [{"year": 2020, "mean_epc": 70.0, "mean_co2": 4.0}]


def test_api_csv_authorities_and_authority_summary(tmp_path, monkeypatch):
    data_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "ANALYSIS_AUTHORITY": ["Devon", "Devon", "Somerset"],
            "YEAR": [2020, 2021, 2021],
            "CURRENT_ENERGY_EFFICIENCY": [60.0, 80.0, 75.0],
            "CO2_EMISSIONS_CURRENT": [5.0, 3.0, 4.0],
            "BELOW_C_INT": [1, 0, 0],
            "POLICY_PERIOD": ["MEES", "Post-Strategy", "Post-Strategy"],
            "BELOW_C": [True, False, False],
        }
    ).to_csv(data_path, index=False)

    monkeypatch.setattr(api, "DATA_PATH", data_path)
    monkeypatch.setenv("ENABLE_DATABASE", "false")

    client = TestClient(api.app)
    authorities = client.get("/authorities")
    summary = client.get("/authority/devon/summary")
    trends = client.get("/trends?authority=Devon")
    policy = client.get("/policy-summary?authority=Somerset")
    missing = client.get("/authority/Unknown/summary")

    assert authorities.status_code == 200
    assert authorities.json() == {
        "count": 2,
        "source": "csv",
        "authorities": ["Devon", "Somerset"],
    }
    assert summary.status_code == 200
    assert summary.json()["authority"] == "Devon"
    assert summary.json()["source"] == "csv"
    assert summary.json()["year_min"] == 2020
    assert summary.json()["year_max"] == 2021
    assert summary.json()["certificate_count"] == 2
    assert summary.json()["latest_mean_epc"] == 80.0
    assert trends.status_code == 200
    assert trends.json() == [
        {"year": 2020, "mean_epc": 60.0, "mean_co2": 5.0},
        {"year": 2021, "mean_epc": 80.0, "mean_co2": 3.0},
    ]
    assert policy.status_code == 200
    assert policy.json() == [
        {
            "POLICY_PERIOD": "Post-Strategy",
            "mean_epc": 75.0,
            "mean_co2": 4.0,
            "below_c_share": 0.0,
        }
    ]
    assert missing.status_code == 404


def test_api_database_metadata_and_reads(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'analytics.db'}"
    engine = get_engine(db_url)
    create_tables(engine)
    features = pd.DataFrame(
        {
            "LOCAL_AUTHORITY": ["Devon"],
            "YEAR": [2020],
            "POLICY_PERIOD": ["MEES"],
            "CURRENT_ENERGY_EFFICIENCY": [70.0],
            "CO2_EMISSIONS_CURRENT": [4.0],
            "BELOW_C": [False],
            "LMK_KEY": ["a"],
        }
    )
    with session_scope(engine) as session:
        load_authority_year_summary(session, features)
        load_policy_period_summary(session, features)

    monkeypatch.setenv("ENABLE_DATABASE", "true")
    monkeypatch.setenv("DATABASE_URL", db_url)

    client = TestClient(api.app)

    metadata = client.get("/metadata")
    trends = client.get("/trends")
    policy = client.get("/policy-summary")
    authorities = client.get("/authorities")
    summary = client.get("/authority/Devon/summary")

    assert metadata.status_code == 200
    assert metadata.json()["database"]["enabled"] is True
    assert metadata.json()["database"]["reachable"] is True
    assert trends.json() == [{"year": 2020, "mean_epc": 70.0, "mean_co2": 4.0}]
    assert policy.status_code == 200
    assert policy.json()[0]["POLICY_PERIOD"] == "MEES"
    assert authorities.json() == {"count": 1, "source": "database", "authorities": ["Devon"]}
    assert summary.status_code == 200
    assert summary.json()["source"] == "database"
    assert summary.json()["certificate_count"] == 1


def test_api_database_failure_falls_back_with_warning(tmp_path, monkeypatch, caplog):
    data_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "YEAR": [2020],
            "CURRENT_ENERGY_EFFICIENCY": [70.0],
            "CO2_EMISSIONS_CURRENT": [4.0],
            "POLICY_PERIOD": ["MEES"],
            "BELOW_C": [False],
        }
    ).to_csv(data_path, index=False)

    monkeypatch.setattr(api, "DATA_PATH", data_path)
    monkeypatch.setenv("ENABLE_DATABASE", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}")

    client = TestClient(api.app)
    with caplog.at_level(logging.WARNING):
        response = client.get("/trends")

    assert response.status_code == 200
    assert response.json() == [{"year": 2020, "mean_epc": 70.0, "mean_co2": 4.0}]
    assert "falling back to CSV" in caplog.text


def test_api_validation_artifacts_models_drift_and_retraining_fallbacks(tmp_path, monkeypatch):
    validation_path = tmp_path / "validation_report.json"
    registry_path = tmp_path / "registry.jsonl"
    regression_path = tmp_path / "regression_model_performance.csv"
    classification_path = tmp_path / "classification_model_performance.csv"
    drift_path = tmp_path / "drift_report.json"

    validation_path.write_text(
        json.dumps({"ok": True, "issues": [], "summary": {"rows": 3}}),
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "kind": "table",
                "path": str(tmp_path / "table.csv"),
                "sha256": "abc",
                "metadata": {"task": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"model": "Ridge", "rmse_test": 1.2, "r2_test": 0.5}]).to_csv(
        regression_path, index=False
    )
    pd.DataFrame([{"model": "LogisticRegression", "accuracy": 0.8, "roc_auc": 0.9}]).to_csv(
        classification_path, index=False
    )
    drift_path.write_text(
        json.dumps(
            {
                "summary": {"should_retrain": True, "severity": "medium"},
                "retraining_recommendation": {
                    "should_retrain": True,
                    "severity": "medium",
                    "reasons": ["feature drift"],
                },
                "drift": {"numeric_features": []},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ENABLE_DATABASE", "false")
    monkeypatch.setattr(api, "VALIDATION_REPORT_PATH", validation_path)
    monkeypatch.setattr(api, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(api, "REGRESSION_METRICS_PATH", regression_path)
    monkeypatch.setattr(api, "CLASSIFICATION_METRICS_PATH", classification_path)
    monkeypatch.setattr(api, "DRIFT_REPORT_PATH", drift_path)

    client = TestClient(api.app)

    validation = client.get("/validation/latest")
    artifacts = client.get("/artifacts?limit=1")
    models = client.get("/model/metadata")
    drift = client.get("/drift/latest")
    retraining = client.get("/retraining/status")

    assert validation.status_code == 200
    assert validation.json()["source"] == "artifact"
    assert validation.json()["validation"]["ok"] is True
    assert artifacts.status_code == 200
    assert artifacts.json()["source"] == "artifact"
    assert artifacts.json()["count"] == 1
    assert artifacts.json()["artifacts"][0]["path"] == "table.csv"
    assert models.status_code == 200
    assert models.json()["source"] == "artifact"
    assert len(models.json()["models"]) == 2
    assert drift.json()["summary"]["severity"] == "medium"
    assert retraining.json()["automatic_retraining"] is False
    assert retraining.json()["recommendation"]["should_retrain"] is True


def test_api_empty_states_for_missing_optional_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_DATABASE", "false")
    monkeypatch.setattr(api, "VALIDATION_REPORT_PATH", tmp_path / "missing_validation.json")
    monkeypatch.setattr(api, "REGISTRY_PATH", tmp_path / "missing_registry.jsonl")
    monkeypatch.setattr(api, "REGRESSION_METRICS_PATH", tmp_path / "missing_regression.csv")
    monkeypatch.setattr(api, "CLASSIFICATION_METRICS_PATH", tmp_path / "missing_classification.csv")
    monkeypatch.setattr(api, "DRIFT_REPORT_PATH", tmp_path / "missing_drift.json")

    client = TestClient(api.app)

    assert client.get("/validation/latest").json()["status"] == "unavailable"
    assert client.get("/artifacts").json()["artifacts"] == []
    assert client.get("/model/metadata").json()["status"] == "unavailable"
    assert client.get("/drift/latest").json()["status"] == "unavailable"
    assert client.get("/retraining/status").json()["automatic_retraining"] is False


def test_api_artifact_limit_validation(monkeypatch):
    monkeypatch.setenv("ENABLE_DATABASE", "false")
    client = TestClient(api.app)

    response = client.get("/artifacts?limit=0")

    assert response.status_code == 422


def test_api_model_metadata_from_database(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'analytics.db'}"
    engine = get_engine(db_url)
    create_tables(engine)
    metrics_path = tmp_path / "regression_model_performance.csv"
    pd.DataFrame([{"model": "Ridge", "rmse_test": 1.2}]).to_csv(metrics_path, index=False)
    with session_scope(engine) as session:
        load_model_metrics(session, regression_metrics_path=metrics_path)

    monkeypatch.setenv("ENABLE_DATABASE", "true")
    monkeypatch.setenv("DATABASE_URL", db_url)

    client = TestClient(api.app)
    response = client.get("/model/metadata")

    assert response.status_code == 200
    assert response.json()["source"] == "database"
    assert response.json()["models"][0]["metrics"] == {"rmse_test": 1.2}
