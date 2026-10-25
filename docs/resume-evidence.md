# Ranked EPC resume candidates

Based on inspected implementation and verified outputs, 2026-10-09. These are
current supported claims, not a declaration that the entire credibility programme
is complete. AI = AI Engineer; Applied = Applied AI Engineer; DS = Data Scientist.

## 1. Dataset engineering and validation

Processed 30.2 million EPC source records into a versioned dataset of 2.78 million South West certificates, with source hashes, cross-file duplicate checks and reconciled exclusions.

- Evidence: `scripts/pipeline/build_release.py`, `build_release`, `register_certificates`
  and `prepare_chunk`; `data/releases/epc-sw-2008-2025-v1/manifest.json`;
  `reports/evidence/release-v1-verification.json`.
- Recruiter value: demonstrates substantial ingestion, provenance and auditability
  rather than simply loading a pre-cleaned modelling table.
- Best roles: AI, Applied, DS, KTP.
- Limits: 30,153,553 is the national input count, not the training-set size.
  Duplicate checks cover 2,922,241 selected South West certificates, not all
  national input. Retained cohort is 2008–2025; partial 2026 is excluded.

## 2. Baseline-controlled predictive results

Trained and compared regression and classification models using property-grouped validation, achieving 9.43 EPC-point RMSE versus 9.99 for linear regression and 0.886 ROC-AUC versus 0.878 for logistic regression.

- Evidence: `scripts/pipeline/benchmark_release.py`, `candidates`, `run_benchmark`;
  `reports/benchmarks/property-v1/{protocol.json,selection.json,test_metrics.csv}`;
  `reports/evidence/benchmark-property-v1-verification.json`.
- Recruiter value: reports concrete, independently checked results against
  meaningful baselines, alongside appropriate repeated-property handling.
- Best roles: DS, Applied, AI; also KTP.
- Limits: 200,000 training and 50,000 validation records; final refit uses their
  union. Test contains 541,309 certificates from 380,723 identified properties.
  Classification target is below EPC band C. MLP tuning emitted iteration-limit
  warnings. Neither result establishes energy savings or deployment quality.

## 3. Temporal validation and leakage controls

Evaluated 2025 performance on 154,680 certificates using chronological splits that removed repeat-property overlap, achieving 9.76 RMSE and 0.851 ROC-AUC against 10.21 and 0.846 baseline scores.

- Evidence: `src/ew_housing_energy_impact/ml/evaluation.py`,
  `chronological_partitions`, `assert_property_separation`, `grouped_folds`,
  `build_preprocessor`; `reports/benchmarks/chronological-v1/protocol.json` and
  `test_metrics.csv`; `reports/evidence/benchmark-chronological-v1-verification.json`.
- Recruiter value: demonstrates testing on later observations, acknowledges
  weaker future-year performance and prevents repeated homes spanning partitions.
- Best roles: DS, Applied, KTP; AI where evaluation design matters.
- Limits: baselines are linear regression and logistic regression respectively.
  Training years precede 2024; validation is 2024; test is 2025. Earlier records
  of validation/test properties are purged (212,435 rows). Unknown property IDs
  are excluded (70,293 rows). Identifier errors can still affect separation;
  one later-year evaluation does not establish lasting robustness.

## 4. Experiment lineage and verification

Tracked model comparisons in MLflow with versioned data and split hashes, saved preprocessing pipelines and predictions, and independent checks of reloaded-model outputs and evaluation metrics.

- Evidence: `scripts/pipeline/benchmark_release.py`, `run_benchmark`;
  `scripts/pipeline/verify_benchmark.py`, `verify_benchmark`;
  both benchmark directories' `protocol.json`, `result.json`, `mlflow_run.json`;
  both independent verification JSON files; `tests/test_benchmark_release.py`.
- Recruiter value: makes results inspectable and models reusable; demonstrates
  an actual working tracking integration rather than configuration alone.
- Best roles: AI, Applied, DS.
- Limits: MLflow is local SQLite plus local artifact storage. A clean-environment
  training rerun is in progress; do not yet claim reproduced training results,
  hosted tracking, remote CI or production deployment. Code snapshots identify
  uncommitted implementation changes; Git HEAD alone does not contain them.

## 5. Explainability and uncertainty

Produced model cards, held-out permutation importance and SHAP explanations, with property-bootstrap confidence intervals and performance breakdowns by authority, tenure and construction age.

- Evidence: `scripts/pipeline/report_benchmark.py`, `report_benchmark`;
  `scripts/pipeline/benchmark_release.py`, `bootstrap_regression`,
  `bootstrap_classification`, `run_benchmark`;
  `reports/model_reports/property-v1/report_manifest.json` and model cards;
  `reports/benchmarks/property-v1/subgroup_metrics.csv` and `uncertainty_*.json`.
- Recruiter value: communicates model behaviour and uncertainty to technical
  and domain audiences, relevant to housing and energy research partnerships.
- Best roles: KTP, DS, Applied.
- Limits: 500 property-bootstrap replicates condition on fitted models; no
  retraining uncertainty. Permutation importance uses 5,000 held-out rows and
  five repeats; SHAP uses 128 test rows and 50 background rows. Encoded-feature
  perturbations can produce unrealistic categories. Explanations are not causal.

## 6. Data-loss correction (alternative to bullet 5)

Corrected missing-value filtering that removed early EPC records, retaining 75,193 certificates from 2008 and moving learned imputation into training-only preprocessing pipelines.

- Evidence: `build_release.prepare_chunk`, `evaluation.build_preprocessor`,
  `tests/test_release.py`, `tests/test_evaluation.py`, release manifest and
  independent release verification.
- Recruiter value: illustrates a concrete data-quality diagnosis and methodological
  correction that affects cohort coverage.
- Best roles: DS, KTP, Applied.
- Limits: retained records are not all national 2008 records. Missing floor area
  remains missing in the release; preprocessing does not recover its true value.
  Do not invent a downstream performance gain from this correction.

## 7. Prediction API (AI Engineering alternative)

Built a FastAPI prediction service for verified EPC models, with batch validation, artifact-integrity checks, model provenance and tests for invalid inputs and altered model files.

- Evidence: `src/ew_housing_energy_impact/ml/serving.py`, `create_app`,
  `PropertyInput`, `PredictionRequest`; HTTP checks in
  `tests/test_benchmark_release.py::test_actual_local_mlflow_benchmark_integration`;
  `reports/evidence/serving-local-smoke.json`; full suite: 257 passing tests in
  `reports/evidence/tests-credibility-final.xml`.
- Recruiter value: connects model artifacts to an operational interface with
  validation and provenance, moving beyond notebook-only modelling.
- Best roles: AI, Applied.
- Limits: tested locally with real fitted models. No deployment, sustained-load
  benchmark or authentication yet. Fifty sequential TestClient requests are not
  production latency evidence. The 257-test total covers the entire repository,
  not just tests newly written for serving.

## Recommendation

Use candidates 1–5 for a five-bullet DS/KTP entry, preserving the exact qualification
of the metrics. For four bullets, omit candidate 5 for AI Engineering; combine
the tracking and reporting content briefly for DS/KTP and retain temporal
validation. Candidate 6 is a concrete alternative when data quality is central
to the role. Do not repeat the full dataset count next to model training in a
way that implies all 2.78 million records were used to fit models. For AI/Applied
Engineering, replace candidate 5 with candidate 7 when serving is central to the job.

Unsafe to claim currently: production deployment, real S3/Snowflake execution,
remote CI success, clean-environment reproduction, causal policy effects,
stakeholder adoption, measured savings or business impact. An implementation
or protocol alone does not establish these outcomes.
