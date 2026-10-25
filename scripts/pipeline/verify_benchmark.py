"""Verify a completed benchmark against its release, saved assignments and models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ew_housing_energy_impact.ml.evaluation import (
    FEATURES,
    PROPERTY_KEY,
    assert_property_separation,
    predictive_inputs,
)
from ew_housing_energy_impact.registry import sha256_file


def verify_benchmark(run: Path, release: Path, verify_models: bool = True) -> dict:
    result = json.loads((run / "result.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    selected = json.loads((run / "selection.json").read_text())["selected"]
    dataset = release / "certificates.parquet"
    checks = {
        "complete": result["status"] == "complete" and not (run / "INCOMPLETE").exists(),
        "dataset_hash": sha256_file(dataset) == protocol["dataset_sha256"],
        "release_manifest_hash": sha256_file(release / "manifest.json")
        == protocol["release_manifest_sha256"],
        "protocol_hash": sha256_file(run / "protocol.json") == result["protocol_sha256"],
        "artifact_hashes": all(
            sha256_file(run / name) == digest for name, digest in result["artifacts"].items()
        ),
        "code_snapshot_hashes": all(
            sha256_file(run / "code" / name) == digest
            for name, digest in protocol["code_sha256"].items()
        ),
        "fixed_feature_list": protocol["features"] == FEATURES,
        "internal_early_stopping_disabled": protocol["internal_early_stopping"] is False,
    }
    assignments = pd.read_parquet(run / "split_assignments.parquet")
    checks["unique_split_certificates"] = assignments.LMK_KEY.is_unique
    audit = assert_property_separation(assignments, assignments.partition)
    checks["split_audit"] = audit == protocol["split_audit"]
    sample = pd.read_parquet(run / "development_sample.parquet")
    joined = sample.merge(assignments, on="LMK_KEY", how="left", validate="one_to_one")
    checks["development_roles_match"] = bool(joined.role.eq(joined.partition).all())
    checks["development_counts"] = (
        int(joined.role.eq("train").sum()) == protocol["train_rows"]
        and int(joined.role.eq("validation").sum()) == protocol["validation_rows"]
    )
    folds = pd.read_parquet(run / "cv_assignments.parquet")
    checks["cv_certificate_coverage"] = folds.LMK_KEY.is_unique and set(folds.LMK_KEY) == set(
        sample.loc[sample.role.eq("train"), "LMK_KEY"]
    )
    checks["cv_property_separation"] = bool(
        folds.groupby(PROPERTY_KEY).validation_fold.nunique().eq(1).all()
    )
    checks["cv_fold_count"] = folds.validation_fold.nunique() == protocol["cv_folds"]
    if protocol["strategy"] == "chronological":
        ranges = assignments.groupby("partition").YEAR.agg(["min", "max"])
        checks["chronology"] = (
            ranges.loc["train", "max"] < ranges.loc["validation", "min"]
            and ranges.loc["validation", "max"] < ranges.loc["test", "min"]
        )
    validation = pd.read_csv(run / "validation_metrics.csv")
    expected = {
        "regression": validation.query("task == 'regression'").sort_values("rmse").iloc[0]["model"],
        "classification": validation.query("task == 'classification'")
        .sort_values("roc_auc", ascending=False)
        .iloc[0]["model"],
    }
    checks["validation_only_selection"] = expected == selected == result["selected"]
    table = pd.read_csv(run / "test_metrics.csv")
    test_ids = set(assignments.loc[assignments.partition.eq("test"), "LMK_KEY"])
    source_columns = list(
        dict.fromkeys(
            ["LMK_KEY", PROPERTY_KEY, *FEATURES, "CURRENT_ENERGY_EFFICIENCY", "BELOW_C_INT"]
        )
    )
    source = pd.read_parquet(dataset, columns=source_columns).set_index("LMK_KEY")
    recomputed = {}
    for task, model_name in selected.items():
        predictions = pd.read_parquet(run / f"test_predictions_{task}.parquet")
        checks[f"{task}_test_coverage"] = (
            predictions.LMK_KEY.is_unique and set(predictions.LMK_KEY) == test_ids
        )
        actual_source = source.loc[predictions.LMK_KEY]
        target = "CURRENT_ENERGY_EFFICIENCY" if task == "regression" else "BELOW_C_INT"
        checks[f"{task}_labels_match_source"] = bool(
            np.array_equal(predictions.actual, actual_source[target])
        )
        checks[f"{task}_properties_match_source"] = bool(
            np.array_equal(predictions[PROPERTY_KEY], actual_source[PROPERTY_KEY])
        )
        if task == "regression":
            residual = predictions.actual - predictions.prediction
            metrics = {
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "r2": float(
                    1
                    - np.sum(residual**2)
                    / np.sum((predictions.actual - predictions.actual.mean()) ** 2)
                ),
            }
        else:
            y, pred = predictions.actual.to_numpy(), predictions.prediction.to_numpy()
            tp = int(((y == 1) & (pred == 1)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            metrics = {
                "accuracy": float(np.mean(y == pred)),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "roc_auc": float(roc_auc_score(y, predictions.probability)),
                "average_precision": float(average_precision_score(y, predictions.probability)),
            }
            checks["classification_threshold"] = bool(
                np.array_equal(pred, (predictions.probability.to_numpy() > 0.5).astype(int))
            )
        reported = table.loc[table.task.eq(task) & table.model.eq(model_name)].iloc[0]
        checks[f"{task}_metrics"] = all(
            np.isclose(v, reported[k], rtol=1e-9, atol=1e-9) for k, v in metrics.items()
        )
        recomputed[task] = metrics
        if verify_models:
            model = joblib.load(run / f"selected_{task}.joblib")
            values = model.predict(predictive_inputs(actual_source))
            checks[f"{task}_model_predictions"] = bool(
                np.allclose(values, predictions.prediction, rtol=1e-9, atol=1e-9)
            )
            if task == "classification":
                values = model.predict_proba(predictive_inputs(actual_source))[:, 1]
                checks["classification_model_probabilities"] = bool(
                    np.allclose(values, predictions.probability, rtol=1e-9, atol=1e-9)
                )
        intervals = json.loads((run / f"uncertainty_{task}.json").read_text())
        checks[f"{task}_interval_structure"] = intervals["repetitions"] == protocol[
            "bootstrap_repetitions"
        ] and all(
            len(bounds) == 2 and np.isfinite(bounds).all() and bounds[0] <= bounds[1]
            for bounds in intervals["intervals"].values()
        )
    tracking = json.loads((run / "mlflow_run.json").read_text())
    client = mlflow.MlflowClient(tracking_uri=tracking["tracking_uri"])
    tracked = client.get_run(tracking["run_id"])
    checks["mlflow_finished"] = tracked.info.status == "FINISHED"
    checks["mlflow_protocol"] = (
        tracked.data.params.get("protocol_sha256") == result["protocol_sha256"]
    )
    return {
        "ok": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "selected": selected,
        "recomputed_test_metrics": recomputed,
        "run_id": tracked.info.run_id,
        "models_reloaded": verify_models,
        "bootstrap_intervals_recomputed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify_benchmark(args.run, args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
