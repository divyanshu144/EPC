from __future__ import annotations

import json

import pandas as pd

from ew_housing_energy_impact.db.engine import create_tables, get_engine, session_scope
from ew_housing_energy_impact.db.loaders import (
    load_authority_year_summary,
    load_policy_period_summary,
    refresh_analytics_database,
)
from ew_housing_energy_impact.db.queries import get_policy_summary, get_trends


def _sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LOCAL_AUTHORITY": ["Devon", "Devon", "Bristol, City of"],
            "YEAR": [2020, 2020, 2021],
            "POLICY_PERIOD": ["MEES", "MEES", "Post-Strategy"],
            "CURRENT_ENERGY_EFFICIENCY": [60.0, 80.0, 70.0],
            "CO2_EMISSIONS_CURRENT": [5.0, 3.0, 4.0],
            "BELOW_C": [True, False, False],
            "LMK_KEY": ["a", "b", "c"],
            "HAS_MAINS_GAS": [True, False, True],
            "NEW_BUILD_SHARE": [0.2, 0.2, 0.5],
        }
    )


def test_table_creation_and_summary_queries():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)

    with session_scope(engine) as session:
        assert load_authority_year_summary(session, _sample_features()) == 2
        assert load_policy_period_summary(session, _sample_features()) == 4

    with session_scope(engine) as session:
        trends = get_trends(session, authority="Devon")
        policy = get_policy_summary(session)

    assert len(trends) == 1
    assert trends[0]["authority"] == "Devon"
    assert trends[0]["year"] == 2020
    assert trends[0]["certificate_count"] == 2
    assert trends[0]["mean_current_energy_efficiency"] == 70.0
    assert len(policy) == 2


def test_refresh_analytics_database_dry_run_does_not_require_db():
    result = refresh_analytics_database(dry_run=True)

    assert "planned" in result
    assert any("authority_year_summary" in item for item in result["planned"])


def test_refresh_analytics_database_loads_existing_small_outputs(tmp_path):
    root = tmp_path
    features_dir = root / "data" / "processed" / "ew_epc_core_clean_sw"
    artifacts_dir = root / "reports" / "artifacts"
    tables_dir = root / "reports" / "tables"
    features_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)

    _sample_features().to_csv(features_dir / "ew_epc_core_clean_sw_imputed.csv", index=False)
    (artifacts_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "issues": [],
                "summary": {"rows": 3, "geography": {"present": ["Devon"]}},
            }
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "registry.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "kind": "dataset",
                "path": "data.csv",
                "sha256": None,
                "metadata": {"stage": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"model": "LinearRegression", "rmse_test": 1.2}]).to_csv(
        tables_dir / "regression_model_performance.csv", index=False
    )
    pd.DataFrame([{"model": "LogisticRegression", "roc_auc": 0.8}]).to_csv(
        tables_dir / "classification_model_performance.csv", index=False
    )

    engine = get_engine("sqlite:///:memory:")
    result = refresh_analytics_database(engine=engine, root=root)

    assert result["authority_year_summary"] == 2
    assert result["policy_period_summary"] == 4
    assert result["validation_runs"] == 1
    assert result["artifact_registry"] == 1
    assert result["model_metrics"] == 2
