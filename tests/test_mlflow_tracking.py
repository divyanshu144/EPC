from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd

from ew_housing_energy_impact.ml import tracking
from scripts.pipeline import train_models


def test_mlflow_disabled_noops(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_MLFLOW", "false")

    assert tracking.is_mlflow_enabled() is False
    assert tracking.setup_mlflow() is None
    assert (
        tracking.log_training_run(
            task_type="regression",
            model_name="LinearRegression",
            metrics={"rmse_test": 1.0},
            artifact_paths=[tmp_path / "missing.txt"],
        )
        is False
    )


def test_mlflow_enabled_logs_with_mocked_module(monkeypatch, tmp_path):
    calls: dict[str, list] = {
        "tracking_uri": [],
        "experiment": [],
        "runs": [],
        "params": [],
        "metrics": [],
        "artifacts": [],
        "models": [],
    }

    @contextmanager
    def fake_start_run(**kwargs):
        calls["runs"].append(kwargs)
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nrmse,1.0\n", encoding="utf-8")

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls["tracking_uri"].append(uri),
        set_experiment=lambda name: calls["experiment"].append(name),
        start_run=fake_start_run,
        log_params=lambda params: calls["params"].append(params),
        log_metrics=lambda metrics: calls["metrics"].append(metrics),
        log_artifact=lambda path, artifact_path=None: calls["artifacts"].append(
            (path, artifact_path)
        ),
        sklearn=SimpleNamespace(
            log_model=lambda model, artifact_path: calls["models"].append((model, artifact_path))
        ),
    )

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("ENABLE_MLFLOW", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns-test")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")

    logged = tracking.log_training_run(
        task_type="classification",
        model_name="XGBoost",
        params={"target": "BELOW_C_INT"},
        metrics={"roc_auc": 0.9},
        artifact_paths=[artifact],
        model=object(),
    )

    assert logged is True
    assert calls["tracking_uri"][-1] == "file:///tmp/mlruns-test"
    assert calls["experiment"][-1] == "test-experiment"
    assert calls["params"]
    assert calls["metrics"][-1] == {"roc_auc": 0.9}
    assert calls["artifacts"][-1] == (str(artifact), "classification")
    assert calls["models"]


def test_train_models_tracking_helper_calls_tracking(monkeypatch, tmp_path):
    calls = []

    def fake_log_training_run(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(train_models.tracking, "log_training_run", fake_log_training_run)

    reg_results = pd.DataFrame(
        [{"model": "LinearRegression", "rmse_test": 1.2, "mae_test": 0.8, "r2_test": 0.5}]
    )
    clf_results = pd.DataFrame([{"model": "LogisticRegression", "accuracy": 0.7, "roc_auc": 0.8}])
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("x,y\n", encoding="utf-8")

    train_models.log_mlflow_training_outputs(
        reg_results_df=reg_results,
        clf_results_df=clf_results,
        feature_cols=["PROPERTY_TYPE", "LOG_FLOOR_AREA"],
        target_reg="CURRENT_ENERGY_EFFICIENCY",
        target_clf="BELOW_C_INT",
        train_rows=10,
        test_rows=3,
        model_lookup={},
        artifact_lookup={"LinearRegression": [artifact], "LogisticRegression": [artifact]},
        force_enable=True,
    )

    assert len(calls) == 2
    assert calls[0]["task_type"] == "regression"
    assert calls[0]["model_name"] == "LinearRegression"
    assert calls[0]["metrics"]["rmse_test"] == 1.2
    assert calls[1]["task_type"] == "classification"
    assert calls[1]["params"]["split_strategy"] == "BUILDING_REFERENCE_NUMBER property-level split"
