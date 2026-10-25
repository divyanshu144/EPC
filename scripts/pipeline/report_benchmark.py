"""Generate model cards and sampled explanations for verified selected pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from ew_housing_energy_impact.ml.evaluation import FEATURES, PROPERTY_KEY, SEED, predictive_inputs
from ew_housing_energy_impact.registry import sha256_file


def report_benchmark(run: Path, release: Path, verification: Path, output: Path) -> dict:
    verified = json.loads(verification.read_text())
    result = json.loads((run / "result.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    if not verified["ok"] or verified["run_id"] != result["run_id"]:
        raise ValueError("A successful verification for this exact run is required")
    for name, digest in result["artifacts"].items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"Benchmark artifact changed: {name}")
    if sha256_file(release / "certificates.parquet") != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Reporting not finished\n")
    source = pd.read_parquet(
        release / "certificates.parquet", columns=["LMK_KEY", *FEATURES]
    ).set_index("LMK_KEY")
    development = pd.read_parquet(run / "development_sample.parquet")
    background_ids = development.sample(n=min(50, len(development)), random_state=SEED).LMK_KEY
    metrics = pd.read_csv(run / "test_metrics.csv")
    subgroups = pd.read_csv(run / "subgroup_metrics.csv")
    metadata = {
        "run_id": result["run_id"],
        "protocol_sha256": result["protocol_sha256"],
        "verification_sha256": sha256_file(verification),
        "seed": SEED,
        "explanations": {},
    }
    for task, name in result["selected"].items():
        print(f"Explaining {task}/{name}", flush=True)
        model = joblib.load(run / f"selected_{task}.joblib")
        predictions = pd.read_parquet(run / f"test_predictions_{task}.parquet")
        sample = predictions.sample(n=min(5000, len(predictions)), random_state=SEED)
        X = predictive_inputs(source.loc[sample.LMK_KEY])
        importance = permutation_importance(
            model,
            X,
            sample.actual.to_numpy(),
            scoring="neg_root_mean_squared_error" if task == "regression" else "roc_auc",
            n_repeats=5,
            random_state=SEED,
            n_jobs=1,
        )
        importance_table = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        ).sort_values("importance_mean")
        importance_table.to_csv(output / f"{task}_permutation_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            importance_table.feature,
            importance_table.importance_mean,
            xerr=importance_table.importance_std,
            color="#276b8c",
        )
        ax.set_xlabel("Increase in RMSE" if task == "regression" else "Decrease in ROC-AUC")
        ax.set_title(f"{name}: held-out permutation importance (n={len(sample):,})")
        fig.tight_layout()
        fig.savefig(output / f"{task}_permutation_importance.png", dpi=180)
        plt.close(fig)

        pre = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        explain_ids = sample.LMK_KEY.iloc[:128]
        transformed = pre.transform(predictive_inputs(source.loc[explain_ids]))
        background = pre.transform(predictive_inputs(source.loc[background_ids]))
        predict = (
            estimator.predict
            if task == "regression"
            else lambda values: estimator.predict_proba(values)[:, 1]
        )
        explainer = shap.Explainer(
            predict,
            background,
            algorithm="permutation",
            feature_names=pre.get_feature_names_out(),
            seed=SEED,
        )
        explanation = explainer(transformed, max_evals=3 * (2 * transformed.shape[1] + 1))
        residual = explanation.base_values + explanation.values.sum(axis=1) - predict(transformed)
        if not np.allclose(residual, 0, atol=1e-5):
            raise ValueError("SHAP additivity check failed")
        names = pre.get_feature_names_out()
        shap_table = pd.DataFrame(
            {"feature": names, "mean_absolute_shap": np.abs(explanation.values).mean(axis=0)}
        )
        shap_table.sort_values("mean_absolute_shap", ascending=False).to_csv(
            output / f"{task}_shap.csv", index=False
        )
        top = shap_table.nlargest(15, "mean_absolute_shap").sort_values("mean_absolute_shap")
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(top.feature, top.mean_absolute_shap, color="#276b8c")
        ax.set_xlabel(
            "Mean absolute SHAP value (EPC points)"
            if task == "regression"
            else "Mean absolute SHAP value (below-C probability)"
        )
        ax.set_title(f"{name}: sampled permutation SHAP (n={len(transformed)})")
        fig.tight_layout()
        fig.savefig(output / f"{task}_shap.png", dpi=180)
        plt.close(fig)
        np.savez_compressed(
            output / f"{task}_shap_values.npz",
            values=explanation.values,
            base_values=explanation.base_values,
            transformed=transformed,
        )
        pd.DataFrame({"LMK_KEY": explain_ids}).to_parquet(
            output / f"{task}_explanation_sample.parquet", index=False
        )
        metadata["explanations"][task] = {
            "permutation_rows": len(sample),
            "permutation_repeats": 5,
            "shap_rows": len(transformed),
            "background_rows": len(background),
            "shap_algorithm": "permutation on fitted transformed features",
            "maximum_additivity_error": float(np.max(np.abs(residual))),
            "model_sha256": sha256_file(run / f"selected_{task}.joblib"),
        }
        task_metrics = metrics.loc[metrics.task.eq(task)].dropna(axis=1, how="all")
        uncertainty = json.loads((run / f"uncertainty_{task}.json").read_text())
        authorities = subgroups.loc[
            subgroups.task.eq(task) & subgroups.field.eq("ANALYSIS_AUTHORITY")
        ]
        criterion = "rmse" if task == "regression" else "roc_auc"
        subgroup_range = f"{authorities[criterion].min():.4f}–{authorities[criterion].max():.4f}"
        lines = [
            f"# Model card: {name} ({task})",
            "",
            "## Intended use",
            "",
            "Research into housing energy performance from EPC property characteristics. "
            "Not validated for funding eligibility, regulatory decisions "
            "or measured energy savings.",
            "",
            "## Data and evaluation",
            "",
            f"- Release: `{release.name}`; dataset SHA-256 `{protocol['dataset_sha256']}`.",
            f"- Actual MLflow run: `{result['run_id']}`; split strategy: `{protocol['strategy']}`.",
            f"- Development: {protocol['train_rows']:,} training and "
            f"{protocol['validation_rows']:,} validation records; "
            "selected model refitted on their union. "
            f"Final test: {protocol['test_rows']:,} records "
            f"from {predictions[PROPERTY_KEY].nunique():,} identified properties.",
            "- Hyperparameter tuning: three property-grouped folds; "
            "model selection: validation only.",
            "- Learned imputation, encoding and scaling fitted inside pipelines. "
            "Internal row-level early stopping disabled.",
            "- Predictors: property type, construction-age band, tenure, heating fuel "
            "and log floor area.",
            "",
            "## Results",
            "",
            task_metrics.to_markdown(index=False),
            "",
            f"95% paired property-bootstrap intervals ({uncertainty['repetitions']} replicates):",
            "",
            *[
                f"- {key}: {value[0]:.6f} to {value[1]:.6f}"
                for key, value in uncertainty["intervals"].items()
            ],
            "",
            f"Authority-level {criterion} range: {subgroup_range}. "
            f"Full subgroup results: `{run / 'subgroup_metrics.csv'}`.",
            "",
            "## Explanations",
            "",
            f"![Permutation importance]({task}_permutation_importance.png)",
            "",
            f"![Sampled SHAP]({task}_shap.png)",
            "",
            "Permutation importance uses 5,000 seeded test rows "
            "(or all rows if fewer), five repeats. "
            "SHAP uses up to 128 test rows and 50 development background rows. "
            "Explanations are descriptive; encoded-feature perturbations may create "
            "unrealistic categorical combinations.",
            "",
            "## Limitations",
            "",
            "- Certificates are repeated observations, not unique homes; "
            "identifier linkage may be imperfect.",
            "- Unidentified properties are excluded from the primary evaluation; "
            "performance does not generalise to them automatically.",
            "- Development data are capped for computation. Cohort bounds and "
            "missingness influence the population represented.",
            "- Bootstrap intervals condition on fitted models and do not include "
            "variability from retraining.",
            "- MLP tuning emitted a maximum-iteration warning in the property-v1 run; "
            "no convergence claim is made.",
            "- A property holdout does not establish future-year performance. "
            "Consult the separate chronological run.",
            "- No deployment, adoption, causal effect, retrofit savings "
            "or business impact is implied.",
            "",
        ]
        (output / f"{task}_model_card.md").write_text("\n".join(lines))
    metadata["artifacts"] = {
        p.name: sha256_file(p) for p in output.iterdir() if p.name != "INCOMPLETE"
    }
    (output / "report_manifest.json").write_text(json.dumps(metadata, indent=2))
    (output / "INCOMPLETE").unlink()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_benchmark(args.run, args.release, args.verification, args.output)


if __name__ == "__main__":
    main()
