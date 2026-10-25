"""Compare independently verified training runs with strict numerical tolerances.

This is a reproduction check, not a new model-selection procedure. No tolerance
is adjusted automatically when a result fails to reproduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipeline.verify_benchmark import verify_benchmark


def compare_benchmarks(reference: Path, reproduced: Path, release: Path) -> dict:
    verifications = [verify_benchmark(path, release) for path in (reference, reproduced)]
    protocols = [
        json.loads((path / "protocol.json").read_text()) for path in (reference, reproduced)
    ]
    fields = [
        "dataset_sha256",
        "release_manifest_sha256",
        "split_sha256",
        "development_sample_sha256",
        "cv_sha256",
        "code_sha256",
        "features",
        "seed",
        "strategy",
        "train_cap",
        "validation_cap",
        "candidates",
        "selection",
        "final_fit",
        "bootstrap_repetitions",
    ]
    checks = {"independently_verified": all(v["ok"] for v in verifications)}
    checks.update({f"same_{key}": protocols[0][key] == protocols[1][key] for key in fields})
    checks["same_selected_models"] = verifications[0]["selected"] == verifications[1]["selected"]
    tables = [
        pd.read_csv(path / "test_metrics.csv").set_index(["task", "model"])
        for path in (reference, reproduced)
    ]
    checks["same_tested_models"] = set(tables[0].index) == set(tables[1].index)
    differences = []
    for key in tables[0].index.intersection(tables[1].index):
        for metric in tables[0].columns.drop("selected"):
            old, new = tables[0].loc[key, metric], tables[1].loc[key, metric]
            if pd.isna(old) and pd.isna(new):
                continue
            matches = bool(np.isclose(old, new, rtol=1e-8, atol=1e-8))
            differences.append(
                {
                    "task": key[0],
                    "model": key[1],
                    "metric": metric,
                    "reference": float(old),
                    "reproduced": float(new),
                    "difference": float(new - old),
                    "matches": matches,
                }
            )
    checks["all_test_metrics_match"] = bool(differences) and all(d["matches"] for d in differences)
    prediction_differences = {}
    for task in ("regression", "classification"):
        frames = [
            pd.read_parquet(path / f"test_predictions_{task}.parquet")
            .set_index("LMK_KEY")
            .sort_index()
            for path in (reference, reproduced)
        ]
        same_ids = frames[0].index.equals(frames[1].index)
        checks[f"same_{task}_test_ids"] = same_ids
        if not same_ids:
            continue
        for column in ["actual", "prediction"] + (
            ["probability"] if task == "classification" else []
        ):
            a, b = frames[0][column].to_numpy(), frames[1][column].to_numpy()
            checks[f"same_{task}_{column}"] = bool(np.allclose(a, b, rtol=1e-8, atol=1e-8))
            prediction_differences[f"{task}_{column}"] = float(np.max(np.abs(a - b)))
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "reference": str(reference),
        "reproduced": str(reproduced),
        "tolerance": {"rtol": 1e-8, "atol": 1e-8},
        "metric_differences": differences,
        "maximum_prediction_differences": prediction_differences,
        "limitations": [
            "Same-machine isolated-environment rerun, not cross-platform reproduction.",
            "Environment files may differ in transitive dependency versions.",
            "Bootstrap replicates are not independently recalculated here.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("reproduced", type=Path)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_benchmarks(args.reference, args.reproduced, args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"ok": result["ok"], "checks": result["checks"]}, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
