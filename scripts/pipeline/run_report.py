#!/usr/bin/env python3
"""Run the full report pipeline in order."""

from __future__ import annotations

from scripts.pipeline import epc_download_merge_fast, clean_epc, build_features, report_eda, run_did, train_models, quality_report
from ew_housing_energy_impact.cli import cmd_validate, cmd_manifest


def main() -> None:
    epc_download_merge_fast.main()
    clean_epc.main()
    build_features.main()
    report_eda.main()
    run_did.main()
    train_models.main()
    quality_report.main()
    cmd_validate()
    cmd_manifest()


if __name__ == "__main__":
    main()
