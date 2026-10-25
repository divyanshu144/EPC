"""Frozen development/test benchmark for a completed release.

Model selection uses grouped training CV and a separate validation partition.
Selection is persisted before ANY test prediction. All estimators, including
MLP, have internal row-level early stopping disabled. This command writes to a
fresh run directory and uses actual local MLflow tracking.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import ROOT
from ew_housing_energy_impact.ml.evaluation import (
    FEATURES,
    PROPERTY_KEY,
    SEED,
    assert_property_separation,
    build_preprocessor,
    cap_by_property,
    chronological_partitions,
    grouped_folds,
    predictive_inputs,
    property_partitions,
)
from ew_housing_energy_impact.registry import sha256_file


def candidates() -> dict:
    """Declared before test access; small, explicit grids control compute cost."""
    return {
        "regression": {
            "DummyMean": (DummyRegressor(strategy="mean"), {}),
            "DummyMedian": (DummyRegressor(strategy="median"), {}),
            "LinearRegression": (LinearRegression(), {}),
            "Ridge": (Ridge(alpha=1), {}),
            "Lasso": (Lasso(alpha=0.01, max_iter=5000), {}),
            "RandomForest": (
                RandomForestRegressor(
                    n_estimators=200, min_samples_leaf=5, n_jobs=2, random_state=SEED
                ),
                {},
            ),
            "GradientBoosting": (
                GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED),
                {},
            ),
            "MLP": (
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=300,
                    early_stopping=False,
                    random_state=SEED,
                ),
                {},
            ),
            "GBR_tuned": (
                GradientBoostingRegressor(random_state=SEED),
                {
                    "model__n_estimators": [150, 250],
                    "model__max_depth": [2, 3],
                    "model__learning_rate": [0.05],
                },
            ),
            "MLP_tuned": (
                MLPRegressor(max_iter=300, early_stopping=False, random_state=SEED),
                {
                    "model__hidden_layer_sizes": [(64, 32), (128, 64)],
                    "model__alpha": [0.001],
                },
            ),
        },
        "classification": {
            "DummyPrior": (DummyClassifier(strategy="prior"), {}),
            "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=SEED), {}),
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=5, n_jobs=2, random_state=SEED
                ),
                {},
            ),
            "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), {}),
            "XGBoost": (
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=2,
                    eval_metric="logloss",
                    random_state=SEED,
                ),
                {},
            ),
        },
    }


def score(task: str, y, prediction, probability=None) -> dict[str, float]:
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction)),
        }
    metrics = {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
    }
    if len(np.unique(y)) == 2:
        metrics.update(
            roc_auc=roc_auc_score(y, probability),
            average_precision=average_precision_score(y, probability),
        )
    return {k: float(v) for k, v in metrics.items()}


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str))


def parameter_values(value):
    """Represent NaN parameter sentinels explicitly without permitting NaN metrics."""
    if isinstance(value, dict):
        return {key: parameter_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [parameter_values(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    return value


def bootstrap_regression(df: pd.DataFrame, repetitions: int) -> dict:
    """Resample properties, retaining all certificates and paired baseline errors."""
    frame = pd.DataFrame(
        {
            "property": df[PROPERTY_KEY],
            "squared": (df.actual - df.prediction) ** 2,
            "absolute": (df.actual - df.prediction).abs(),
            "baseline_squared": (df.actual - df.baseline_prediction) ** 2,
        }
    )
    groups = frame.groupby("property").agg(
        squared=("squared", "sum"),
        absolute=("absolute", "sum"),
        baseline_squared=("baseline_squared", "sum"),
        count=("squared", "size"),
    )
    if len(groups) < 2:
        raise ValueError("At least two properties required for uncertainty estimates")
    rng = np.random.default_rng(SEED)
    estimates = []
    values = groups.to_numpy()
    for _ in range(repetitions):
        total = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        rmse = np.sqrt(total[0] / total[3])
        baseline = np.sqrt(total[2] / total[3])
        estimates.append([rmse, total[1] / total[3], baseline - rmse])
    intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "method": "paired property bootstrap percentile 95%",
        "repetitions": repetitions,
        "properties": len(groups),
        "seed": SEED,
        "intervals": {
            name: intervals[:, i].tolist()
            for i, name in enumerate(["rmse", "mae", "baseline_rmse_minus_selected_rmse"])
        },
    }


def bootstrap_classification(df: pd.DataFrame, repetitions: int) -> dict:
    codes, properties = pd.factorize(df[PROPERTY_KEY])
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(properties), size=len(properties))
        weights = np.bincount(sampled, minlength=len(properties))[codes]
        if len(np.unique(df.actual[weights > 0])) < 2:
            continue
        auc = roc_auc_score(df.actual, df.probability, sample_weight=weights)
        baseline = roc_auc_score(df.actual, df.baseline_probability, sample_weight=weights)
        estimates.append([auc, auc - baseline])
    if len(estimates) < max(2, repetitions // 2):
        raise ValueError("Too few valid bootstrap replicates for classification")
    intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "method": "paired property bootstrap percentile 95%",
        "repetitions": repetitions,
        "valid_repetitions": len(estimates),
        "properties": len(properties),
        "seed": SEED,
        "intervals": {
            name: intervals[:, i].tolist()
            for i, name in enumerate(["roc_auc", "selected_auc_minus_logistic_auc"])
        },
    }


def run_benchmark(
    release: Path,
    output: Path,
    strategy: str,
    train_cap: int,
    validation_cap: int,
    bootstrap: int,
    tracking_root: Path | None = None,
) -> dict:
    started = time.time()
    if (release / "INCOMPLETE").exists():
        raise ValueError("Release is incomplete")
    manifest = json.loads((release / "manifest.json").read_text())
    dataset = release / "certificates.parquet"
    if manifest["status"] != "complete" or sha256_file(dataset) != manifest["dataset_sha256"]:
        raise ValueError("Release status/hash validation failed")
    if bootstrap < 2:
        raise ValueError("At least two bootstrap replications required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Benchmark has not completed.\n")
    columns = list(
        dict.fromkeys(
            [
                "LMK_KEY",
                PROPERTY_KEY,
                "YEAR",
                "ANALYSIS_AUTHORITY",
                *FEATURES,
                "CURRENT_ENERGY_EFFICIENCY",
                "BELOW_C_INT",
            ]
        )
    )
    df = pd.read_parquet(dataset, columns=columns)
    if len(df) != manifest["rows"] or df.LMK_KEY.duplicated().any():
        raise ValueError("Release row count or certificate uniqueness check failed")
    partition = (
        chronological_partitions(df) if strategy == "chronological" else property_partitions(df)
    )
    split_audit = assert_property_separation(df, partition)
    assignment = df[["LMK_KEY", PROPERTY_KEY, "YEAR"]].assign(partition=partition)
    assignment.to_parquet(output / "split_assignments.parquet", index=False)
    train = cap_by_property(df.loc[partition.eq("train")], train_cap)
    validation = cap_by_property(df.loc[partition.eq("validation")], validation_cap)
    test = df.loc[partition.eq("test")].copy()
    sampled_ids = pd.concat(
        [train[["LMK_KEY"]].assign(role="train"), validation[["LMK_KEY"]].assign(role="validation")]
    )
    sampled_ids.to_parquet(output / "development_sample.parquet", index=False)
    folds = grouped_folds(train)
    # Save actual fold membership for review, using certificate IDs rather than positions alone.
    fold_rows = []
    for i, (_, heldout) in enumerate(folds):
        fold_rows.append(train.iloc[heldout][["LMK_KEY", PROPERTY_KEY]].assign(validation_fold=i))
    pd.concat(fold_rows).to_parquet(output / "cv_assignments.parquet", index=False)
    registry = candidates()
    code_paths = [
        Path(__file__),
        ROOT / "src/ew_housing_energy_impact/ml/evaluation.py",
        ROOT / "src/ew_housing_energy_impact/registry.py",
    ]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths}
    snapshot = output / "code"
    for path in code_paths:
        dest = snapshot / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    protocol = {
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "dataset_sha256": manifest["dataset_sha256"],
        "split_sha256": sha256_file(output / "split_assignments.parquet"),
        "development_sample_sha256": sha256_file(output / "development_sample.parquet"),
        "cv_sha256": sha256_file(output / "cv_assignments.parquet"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": code_hashes,
        "python": platform.python_version(),
        "features": FEATURES,
        "seed": SEED,
        "strategy": strategy,
        "split_audit": split_audit,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_cap": train_cap,
        "validation_cap": validation_cap,
        "cv_folds": 3,
        "cv_strategy": "GroupKFold by identified property",
        "selection": {
            "regression": "minimum validation RMSE",
            "classification": "maximum validation ROC-AUC",
        },
        "final_fit": "refit selected configuration and baselines on sampled train plus validation",
        "classification_threshold": 0.5,
        "bootstrap_repetitions": bootstrap,
        "internal_early_stopping": False,
        "candidates": {
            task: {
                name: {
                    "estimator": type(est).__name__,
                    "params": parameter_values(est.get_params()),
                    "grid": grid,
                }
                for name, (est, grid) in models.items()
            }
            for task, models in registry.items()
        },
    }
    write_json(output / "protocol.json", protocol)
    protocol_hash = sha256_file(output / "protocol.json")
    tracking_root = tracking_root or ROOT / "reports/evidence/mlflow"
    tracking_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_root / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name("epc-verified-benchmark")
    if experiment is None:
        mlflow.create_experiment(
            "epc-verified-benchmark", artifact_location=(tracking_root / "artifacts").as_uri()
        )
    mlflow.set_experiment("epc-verified-benchmark")
    X_train, X_val = predictive_inputs(train), predictive_inputs(validation)
    targets = {"regression": "CURRENT_ENERGY_EFFICIENCY", "classification": "BELOW_C_INT"}
    tuned = {}
    validation_rows = []
    with mlflow.start_run(run_name=output.name) as parent:
        mlflow.log_params(
            {
                "dataset_sha256": protocol["dataset_sha256"],
                "protocol_sha256": protocol_hash,
                "strategy": strategy,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "seed": SEED,
            }
        )
        write_json(
            output / "mlflow_run.json",
            {"run_id": parent.info.run_id, "tracking_uri": mlflow.get_tracking_uri()},
        )
        for task, models in registry.items():
            y = train[targets[task]]
            for name, (estimator, grid) in models.items():
                print(f"Training {task}/{name}", flush=True)
                pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
                with mlflow.start_run(run_name=f"{task}-{name}", nested=True):
                    if grid:
                        search = GridSearchCV(
                            pipe,
                            grid,
                            cv=folds,
                            n_jobs=2,
                            scoring="neg_root_mean_squared_error"
                            if task == "regression"
                            else "roc_auc",
                            error_score="raise",
                        )
                        search.fit(X_train, y)
                        fitted = search.best_estimator_
                        pd.DataFrame(search.cv_results_).to_csv(
                            output / f"cv_{name}.csv", index=False
                        )
                        mlflow.log_metric("best_cv_score", search.best_score_)
                    else:
                        fitted = pipe.fit(X_train, y)
                    tuned[(task, name)] = clone(fitted)
                    prediction = fitted.predict(X_val)
                    probability = (
                        fitted.predict_proba(X_val)[:, 1] if task == "classification" else None
                    )
                    metrics = score(task, validation[targets[task]], prediction, probability)
                    validation_rows.append({"task": task, "model": name, **metrics})
                    mlflow.log_params({"model": name, "task": task})
                    mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
                    write_json(
                        output / f"params_{task}_{name}.json",
                        parameter_values(fitted.get_params(deep=True)),
                    )
                    mlflow.log_artifact(str(output / f"params_{task}_{name}.json"))
                    del fitted
        validation_table = pd.DataFrame(validation_rows)
        validation_table.to_csv(output / "validation_metrics.csv", index=False)
        selected = {
            "regression": validation_table.query("task == 'regression'")
            .sort_values("rmse")
            .iloc[0]["model"],
            "classification": validation_table.query("task == 'classification'")
            .sort_values("roc_auc", ascending=False)
            .iloc[0]["model"],
        }
        write_json(
            output / "selection.json",
            {
                "selected": selected,
                "protocol_sha256": protocol_hash,
                "test_metrics_accessed": False,
            },
        )
        # No test predictions occur above this point. Selection is now fixed on disk.
        development = pd.concat([train, validation], ignore_index=True)
        X_dev, X_test = predictive_inputs(development), predictive_inputs(test)
        test_metrics = []
        subgroup_rows = []
        for task, winner in selected.items():
            names = list(
                dict.fromkeys(
                    [winner, "LinearRegression", "DummyMean", "DummyMedian"]
                    if task == "regression"
                    else [winner, "LogisticRegression", "DummyPrior"]
                )
            )
            predictions = test[
                [
                    "LMK_KEY",
                    PROPERTY_KEY,
                    "YEAR",
                    "ANALYSIS_AUTHORITY",
                    "TENURE_STD",
                    "AGE_BAND_STD",
                ]
            ].copy()
            predictions["actual"] = test[targets[task]].to_numpy()
            for name in names:
                print(f"Final refit and test: {task}/{name}", flush=True)
                model = tuned[(task, name)].fit(X_dev, development[targets[task]])
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1] if task == "classification" else None
                metrics = score(task, predictions.actual, pred, prob)
                test_metrics.append(
                    {"task": task, "model": name, "selected": name == winner, **metrics}
                )
                mlflow.log_metrics({f"test_{task}_{name}_{k}": v for k, v in metrics.items()})
                if name == winner:
                    predictions["prediction"] = pred
                    if prob is not None:
                        predictions["probability"] = prob
                    joblib.dump(model, output / f"selected_{task}.joblib")
                if name == ("LinearRegression" if task == "regression" else "LogisticRegression"):
                    predictions["baseline_prediction"] = pred
                    if prob is not None:
                        predictions["baseline_probability"] = prob
                del model
            predictions.to_parquet(output / f"test_predictions_{task}.parquet", index=False)
            uncertainty = (
                bootstrap_regression(predictions, bootstrap)
                if task == "regression"
                else bootstrap_classification(predictions, bootstrap)
            )
            write_json(output / f"uncertainty_{task}.json", uncertainty)
            for field in ["ANALYSIS_AUTHORITY", "TENURE_STD", "AGE_BAND_STD", "YEAR"]:
                for group, subset in predictions.groupby(field, dropna=False):
                    if len(subset) < 2:
                        continue
                    metrics = score(
                        task,
                        subset.actual,
                        subset.prediction,
                        subset.probability if task == "classification" else None,
                    )
                    subgroup_rows.append(
                        {
                            "task": task,
                            "field": field,
                            "group": group,
                            "rows": len(subset),
                            "properties": subset[PROPERTY_KEY].nunique(),
                            "small_sample": len(subset) < 200,
                            **metrics,
                        }
                    )
        pd.DataFrame(test_metrics).to_csv(output / "test_metrics.csv", index=False)
        pd.DataFrame(subgroup_rows).to_csv(output / "subgroup_metrics.csv", index=False)
        for path in output.iterdir():
            if path.is_file() and path.name != "INCOMPLETE":
                mlflow.log_artifact(str(path))
        for path in code_paths:
            if sha256_file(path) != code_hashes[str(path.relative_to(ROOT))]:
                raise RuntimeError("Benchmark implementation changed during execution")
        if sha256_file(output / "protocol.json") != protocol_hash:
            raise RuntimeError("Protocol changed during execution")
        result = {
            "status": "complete",
            "run_id": parent.info.run_id,
            "selected": selected,
            "duration_seconds": time.time() - started,
            "protocol_sha256": protocol_hash,
            "artifacts": {
                p.name: sha256_file(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "INCOMPLETE"
            },
            "limitations": [
                "UPRN linkage may miss repeat properties.",
                "Property sample is capped for model development; see protocol counts.",
                "Bootstrap intervals condition on fitted models; no retraining per replicate.",
                "Temporal evaluation is a separate run, not implied by property holdout.",
            ],
        }
        write_json(output / "result.json", result)
        mlflow.log_artifact(str(output / "result.json"))
        (output / "INCOMPLETE").unlink()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strategy", choices=["property", "chronological"], default="property")
    parser.add_argument("--train-cap", type=int, default=200_000)
    parser.add_argument("--validation-cap", type=int, default=50_000)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("run-id must be a simple directory name")
    result = run_benchmark(
        args.release,
        ROOT / "reports/benchmarks" / args.run_id,
        args.strategy,
        args.train_cap,
        args.validation_cap,
        args.bootstrap,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
