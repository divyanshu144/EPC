"""Command-line interface for the EPC report pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ew_housing_energy_impact.logging_utils import setup_logger
from ew_housing_energy_impact.paths import repo_root
from ew_housing_energy_impact.validation import validate_imputed_csv, write_validation_report
from ew_housing_energy_impact.manifest import write_manifest

# Ensure repo root is on sys.path for scripts.* imports
root = repo_root()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

def cmd_run_all() -> None:
    from scripts.pipeline import (
        epc_download_merge_fast,
        clean_epc,
        build_features,
        report_eda,
        run_did,
        train_models,
    )

    epc_download_merge_fast.main()
    clean_epc.main()
    build_features.main()
    report_eda.main()
    run_did.main()
    train_models.main()


def cmd_validate() -> None:
    root = repo_root()
    data_path = root / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
    report_path = root / "reports" / "artifacts" / "validation_report.json"
    result = validate_imputed_csv(data_path)
    write_validation_report(result, report_path)
    from ew_housing_energy_impact.registry import register_artifact
    register_artifact(report_path.parent / "registry.jsonl", "validation_report", report_path, {"ok": result.ok})
    logger = setup_logger()
    if result.ok:
        logger.info("Validation OK. Report written to %s", report_path)
    else:
        logger.warning("Validation issues found. Report written to %s", report_path)


def cmd_manifest() -> None:
    root = repo_root()
    data_path = root / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
    out_path = root / "reports" / "artifacts" / "data_manifest.json"
    import pandas as pd

    df = pd.read_csv(data_path)
    write_manifest(df, out_path)
    from ew_housing_energy_impact.registry import register_artifact
    register_artifact(out_path.parent / "registry.jsonl", "manifest", out_path, {"type": "data_manifest"})
    logger = setup_logger()
    logger.info("Manifest written to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ewhei", description="EPC report pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run-all", help="Run the full report pipeline")
    sub.add_parser("download", help="Download and merge EPC data")
    sub.add_parser("clean", help="Clean EPC data for South West")
    sub.add_parser("features", help="Build features and policy periods")
    sub.add_parser("eda", help="Generate EDA figures and tables")
    sub.add_parser("fe", help="Run fixed-effects and clustering analysis")
    sub.add_parser("train", help="Run predictive modelling")
    sub.add_parser("quality", help="Generate data quality HTML report")
    sub.add_parser("validate", help="Validate imputed dataset and write report")
    sub.add_parser("manifest", help="Write data manifest")

    args = parser.parse_args()

    if args.cmd == "run-all":
        cmd_run_all()
    elif args.cmd == "download":
        from scripts.pipeline import epc_download_merge_fast
        epc_download_merge_fast.main()
    elif args.cmd == "clean":
        from scripts.pipeline import clean_epc
        clean_epc.main()
    elif args.cmd == "features":
        from scripts.pipeline import build_features
        build_features.main()
    elif args.cmd == "eda":
        from scripts.pipeline import report_eda
        report_eda.main()
    elif args.cmd == "fe":
        from scripts.pipeline import run_did
        run_did.main()
    elif args.cmd == "train":
        from scripts.pipeline import train_models
        train_models.main()
    elif args.cmd == "quality":
        from scripts.pipeline import quality_report
        quality_report.main()
    elif args.cmd == "validate":
        cmd_validate()
    elif args.cmd == "manifest":
        cmd_manifest()


if __name__ == "__main__":
    main()
