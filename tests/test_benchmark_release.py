from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from ew_housing_energy_impact.ml.evaluation import PROPERTY_KEY
from ew_housing_energy_impact.registry import sha256_file
from scripts.pipeline import benchmark_release as benchmark
from scripts.pipeline.verify_benchmark import verify_benchmark


def test_production_protocol_parameters_serialize_and_disable_internal_early_stopping():
    for models in benchmark.candidates().values():
        for name, (estimator, _) in models.items():
            json.dumps(benchmark.parameter_values(estimator.get_params()), allow_nan=False)
            if "MLP" in name:
                assert estimator.early_stopping is False


def fixture_release(root):
    root.mkdir()
    n = 600
    rng = np.random.default_rng(42)
    area = rng.normal(4, 0.4, n)
    score = 60 + area * 2 + rng.normal(0, 5, n)
    data = pd.DataFrame(
        {
            "LMK_KEY": [f"cert-{i}" for i in range(n)],
            PROPERTY_KEY: [f"property-{i // 2}" for i in range(n)],
            "YEAR": [2020] * n,
            "ANALYSIS_AUTHORITY": ["Devon"] * n,
            "PROPERTY_TYPE": ["House"] * n,
            "AGE_BAND_STD": ["old"] * n,
            "TENURE_STD": ["Unknown"] * n,
            "MAIN_FUEL_STD": ["gas"] * n,
            "LOG_FLOOR_AREA": area,
            "CURRENT_ENERGY_EFFICIENCY": score,
            "BELOW_C_INT": (score < 69).astype(int),
        }
    )
    data.to_parquet(root / "certificates.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": n,
                "dataset_sha256": sha256_file(root / "certificates.parquet"),
            }
        )
    )
    return data


def test_actual_local_mlflow_benchmark_integration(tmp_path, monkeypatch):
    """Exercise real split/preprocess/refit/prediction/artifact/MLflow persistence."""
    fixture_release(tmp_path / "release")
    monkeypatch.setattr(
        benchmark,
        "candidates",
        lambda: {
            "regression": {
                "LinearRegression": (LinearRegression(), {"model__fit_intercept": [True, False]}),
                "DummyMean": (DummyRegressor(strategy="mean"), {}),
                "DummyMedian": (DummyRegressor(strategy="median"), {}),
            },
            "classification": {
                "LogisticRegression": (LogisticRegression(), {}),
                "DummyPrior": (DummyClassifier(strategy="prior"), {}),
            },
        },
    )
    old_uri = mlflow.get_tracking_uri()
    try:
        result = benchmark.run_benchmark(
            tmp_path / "release",
            tmp_path / "run",
            "property",
            200,
            80,
            5,
            tracking_root=tmp_path / "mlflow",
        )
        run = mlflow.get_run(result["run_id"])
        assert run.info.status == "FINISHED"
        assert "test_regression_LinearRegression_rmse" in run.data.metrics
        assert result["status"] == "complete"
        assert not (tmp_path / "run/INCOMPLETE").exists()
        selection = json.loads((tmp_path / "run/selection.json").read_text())
        assert selection["test_metrics_accessed"] is False
        for task in ["regression", "classification"]:
            prediction = pd.read_parquet(tmp_path / f"run/test_predictions_{task}.parquet")
            assert len(prediction) > 0
            assert prediction.LMK_KEY.is_unique
            assert (tmp_path / f"run/selected_{task}.joblib").exists()
        for name, digest in result["artifacts"].items():
            assert sha256_file(tmp_path / "run" / name) == digest
        verification = verify_benchmark(tmp_path / "run", tmp_path / "release")
        assert verification["ok"], verification["checks"]
        # Exercise actual fitted pipelines through the HTTP boundary.
        from fastapi.testclient import TestClient

        from ew_housing_energy_impact.ml.serving import create_app

        verification_path = tmp_path / "verified.json"
        verification_path.write_text(json.dumps(verification))
        client = TestClient(create_app(tmp_path / "run", verification_path))
        response = client.post(
            "/predict",
            json={
                "properties": [
                    {
                        "property_type": "House",
                        "age_band": "old",
                        "tenure": "Unknown",
                        "main_fuel": "gas",
                        "floor_area": 100,
                    },
                    {},
                ]
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["run_id"] == result["run_id"]
        assert len(payload["predictions"]) == 2
        assert 0 <= payload["predictions"][0]["below_c_probability"] <= 1
        assert "floor_area" in payload["predictions"][1]["missing_inputs"]
        assert client.post("/predict", json={"properties": [{"floor_area": -1}]}).status_code == 422
        assert client.post("/predict", json={"properties": [{}] * 101}).status_code == 422
        assert (
            client.post("/predict", json={"properties": [{"model_path": "/tmp/a"}]}).status_code
            == 422
        )
        assert (
            client.get("/model").json()["dataset_sha256"]
            == json.loads((tmp_path / "release/manifest.json").read_text())["dataset_sha256"]
        )
        with (tmp_path / "run/selected_regression.joblib").open("ab") as handle:
            handle.write(b"tamper")
        with pytest.raises(ValueError, match="Model hash mismatch"):
            create_app(tmp_path / "run", verification_path)
    finally:
        mlflow.set_tracking_uri(old_uri)


def test_benchmark_refuses_tampered_release(tmp_path):
    fixture_release(tmp_path / "release")
    with (tmp_path / "release/certificates.parquet").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash validation"):
        benchmark.run_benchmark(tmp_path / "release", tmp_path / "run", "property", 200, 80, 5)
    assert not (tmp_path / "run").exists()


def test_property_bootstrap_preserves_paired_baseline_comparison():
    data = pd.DataFrame(
        {
            PROPERTY_KEY: ["a", "a", "b", "b"],
            "actual": [1.0, 2.0, 3.0, 4.0],
            "prediction": [1.0, 2.0, 3.0, 4.0],
            "baseline_prediction": [2.0, 3.0, 4.0, 5.0],
        }
    )
    result = benchmark.bootstrap_regression(data, 20)
    assert result["intervals"]["rmse"] == [0.0, 0.0]
    assert result["intervals"]["baseline_rmse_minus_selected_rmse"] == [1.0, 1.0]
