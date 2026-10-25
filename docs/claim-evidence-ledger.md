# Claim evidence ledger

Updated 2026-10-09. This ledger separates measured results from implementation,
tests, future work and external dependencies. Historical metrics must never be
attached to the new release.

| Claim | Status | Evidence and limits |
|---|---|---|
| Processed 30,153,553 national certificate records | Verified release input count | `data/releases/epc-sw-2008-2025-v1/manifest.json`, source ledger for 19 annual files (2008–2026); includes 2026 input, which is excluded from the final cohort. Source SHA-256 hashes recorded during build. |
| Released 2,776,781 South West certificates covering 2008–2025 and 14 authorities | Verified | Release Parquet, manifest, and independent `reports/evidence/release-v1-verification.json`. Dataset SHA-256: `978331eefcd30a3b482d992ab2a8b2977f5d5c0705df2d3037276c61ace12be6`. |
| Global certificate uniqueness in selected South West inputs | Verified within stated scope | Unique SQLite index accepted all 2,922,241 selected source certificates before eligibility filtering; independent verifier confirmed uniqueness of all retained records. NOT a national-wide duplicate audit. |
| Recovered missing early-year coverage | Verified | 75,193 retained 2008 certificates; prior cleaner dropped missing floor-area records. Source ledger records numeric missingness. Missing measurements retained for fold-local imputation, not fabricated. |
| Dataset has 1,905,073 unique identified properties | Verified count, imperfect linkage | Independent verifier agrees with manifest. 70,293 rows use unidentified-property fallback keys and are excluded from primary grouped benchmark. UPRN linkage accuracy is not established by counting IDs. |
| Property-separated holdout and grouped tuning | Verified on completed property benchmark | `ml/evaluation.py`, `tests/test_evaluation.py`, new benchmark runner. Direct train CLI migrated; historical run-all/Prefect paths remain legacy. |
| Fold-local imputation | Implemented and tested | Missing floor area retained; imputer/scaler inside sklearn Pipeline. Test changes held-out values and verifies training median is unchanged. |
| Actual MLflow integration | Verified locally on full benchmark and integration fixture | Finished parent run `8e0a1377e51f4cf78925f80ea91444cc`, child runs and artifacts; independent benchmark verification. Does not establish hosted deployment. |
| Property-held-out model performance | Verified | `reports/benchmarks/property-v1/test_metrics.csv`, result and independent `reports/evidence/benchmark-property-v1-verification.json`: MLP RMSE 9.429 versus linear 9.985; XGBoost ROC-AUC 0.8858 versus logistic 0.8784 on 541,309 records / 380,723 properties. Training plus validation capped at 250,000 rows. Separate 2025 evaluation also independently verified; see next row. |
| Future-year evaluation | Verified | `reports/benchmarks/chronological-v1/test_metrics.csv`, protocol and independent verification: 154,680 certificates / 151,892 properties in 2025. MLP RMSE 9.757 versus linear 10.212; XGBoost AUC 0.8508 versus logistic 0.8456. Earlier overlapping properties purged. |
| Property-bootstrap uncertainty | Generated; interval structure independently checked | 500 paired property-bootstrap replicates: linear-minus-MLP RMSE 95% interval [0.5456, 0.5672]; XGBoost-minus-logistic AUC [0.007093, 0.007587]. Conditional on fitted models; verifier does not rerun all replicates. |
| Selected-model explainability and model cards | Generated; property figures visually checked | `reports/model_reports/property-v1/report_manifest.json`, cards, permutation importance and SHAP artifacts. Additivity checks passed; encoded-feature perturbations may be unrealistic, and explanations are not causal. |
| Clean-environment reproducibility | NOT YET CLAIMABLE | Isolated pinned environment installed, pip check passed, full property-reproduce-v1 training running. No successful reproduction claim until comparison completes. |
| Completed policy bootstrap / causal effects | NOT CLAIMABLE | Saved bootstrap p-values are blank. Causal identification remains unestablished. |
| Local prediction serving | Implemented and tested | `ml/serving.py`, actual fitted-pipeline integration checks, and 50-request actual-model TestClient smoke in `reports/evidence/serving-local-smoke.json`. No deployment or production-latency claim. |
| Cloud ingestion / deployed prediction service | NOT YET CLAIMABLE | No live cloud execution or deployment evidence; examined environment and .env entries contain no required cloud credentials. |
| Stakeholder adoption / savings / business impact | NOT CLAIMABLE | Requires actual participation and measured outcomes. No outreach performed. |

## Release accounting

- National input: 30,153,553; outside selected South West geography: 27,231,312.
- Selected South West input: 2,922,241.
- Outside primary year scope: 97,873 (2026).
- Sequential remaining exclusions: current-efficiency bounds 3,163;
  potential-efficiency bounds 32,938; floor-area bounds 5,217;
  emissions bounds 6,061; negative efficiency gap 208.
- Retained: 2,776,781. These mutually exclusive exclusions reconcile exactly.
- Missing retained floor area: 197,236; missing raw tenure: 757,816.
- Bounds are analysis-cohort decisions, not universal definitions of invalid EPCs.

## Open measurement requirements

Finish chronological report visual QA;
clean-environment reproduction (full training rerun running); actual CI run; policy sensitivity and
bootstrap; verified cloud integration; deployed API and measured operational
behaviour; stakeholder pilot evaluation. No completion claim until each is checked.
