# EW Housing Energy Impact

An end-to-end data and ML pipeline over **30 million UK Energy Performance Certificate
(EPC) records**. It builds a versioned, hash-verified dataset of 2.78 million South West
England certificates (2008–2025) and evaluates predictive models against baselines
without letting the same home appear in both training and test data.

Built as a dissertation project on how housing energy efficiency changed across five UK
energy-policy periods, and engineered to be audited, not just run.

## Results

| | |
|---|---|
| **Scale** | 30,153,553 national records → **2,776,781** South West certificates, 14 local authorities, 1.9M identified properties |
| **Data integrity** | Source SHA-256 hashes; every exclusion reconciled to the row; release re-checked by an independent verifier |
| **Regression** (EPC score) | Tuned MLP **9.43 RMSE** vs 9.99 linear baseline (−5.6%, R² 0.62 vs 0.57) |
| **Classification** (below band C) | XGBoost **0.886 ROC-AUC** vs 0.878 logistic baseline |
| **Future-year test** (2025, 154,680 certificates) | 9.76 RMSE / 0.851 AUC, still ahead of both baselines |
| **Uncertainty** | 500 property-level bootstrap replicates: RMSE gain 95% CI [0.55, 0.57]; AUC gain [0.0071, 0.0076] |
| **Engineering** | 257 tests, 0 failures; Ruff-enforced style; GitHub Actions CI |

Held-out test set: 541,309 certificates from 380,723 properties, with **zero property
overlap** with training. The gain over the linear baselines is modest. The point is that
it is measured honestly, with baselines, grouped validation and confidence intervals.

## What makes it rigorous

- **No leakage from repeat inspections.** A property can have many certificates over
  time. Splits, tuning folds and the 2025 test all separate by property, and a check
  fails the run if any property crosses partitions.
- **Imputation inside the model pipeline.** Missing floor area is kept missing in the
  data and imputed only on training folds. An earlier cleaner silently dropped those
  rows, removing early-year records. 75,193 certificates from 2008 are now retained.
- **Every result is traceable.** Each benchmark records data, split and code hashes, logs
  to MLflow, saves its preprocessing pipeline and predictions, and is re-verified by a
  separate script that reloads the model and recomputes the metrics.
- **Reproduction checked.** A clean-environment rerun matched the splits, model
  selection and selected-model predictions bit-for-bit. Two baseline logistic-regression
  metrics differed by up to 9e-7, which fails the strict 1e-8 tolerance. The report
  states this rather than loosening the check.
- **Statistics suited to the data.** With only 14 local authorities, standard clustered
  errors are unreliable (Cameron, Gelbach & Miller 2008), so the policy models use a
  wild cluster bootstrap alongside fixed-effects panel OLS and an event study.

## Stack

Python · pandas / PyArrow · scikit-learn · XGBoost · SHAP · `linearmodels` ·
MLflow · FastAPI · Streamlit · Prefect · S3 / Snowflake · Docker Compose · pytest · Ruff

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .
make ci                          # lint + format + 257 tests; no data or credentials needed
ewhei orchestrate --dry-run      # inspect the Prefect pipeline plan
```

Rebuild and benchmark (needs the downloaded EPC archives and `EPC_BEARER_TOKEN` in `.env`):

```bash
python -m scripts.pipeline.build_release --release-id epc-sw-2008-2025-v1
ewhei train --release data/releases/epc-sw-2008-2025-v1 --output reports/benchmarks/my-run
python -m scripts.pipeline.verify_benchmark reports/benchmarks/my-run \
  --release data/releases/epc-sw-2008-2025-v1 --output reports/evidence/my-run-verification.json
```

Serve a verified model: `uvicorn ew_housing_energy_impact.ml.serving:create_app --factory`
(see `docs/api.md`).

## Limitations

- Results are predictive and observational. EPCs are lodgement records, not a stock
  census, so nothing here establishes causal policy effects, energy savings or
  deployment performance.
- Models are fitted on at most 250,000 sampled rows and evaluated on all 541,309 test
  rows. The full 2.78M are not used for training.
- The policy bootstrap output is not yet complete, so no policy-effect results are
  claimed.
- Prediction serving is tested locally only. There is no deployment, authentication or
  load testing.
- The S3 → Snowflake ingestion path is tested with mocks. It has not been run against
  live cloud services.
- Linkage of repeat properties depends on EPC identifiers, which can miss repeats. About
  70,000 rows with no usable identifier are excluded from the grouped benchmarks.

## Repository guide

```
scripts/pipeline/            release build, benchmark, verification, FE models, reports
src/ew_housing_energy_impact/
  ml/                        grouped evaluation, preprocessing, prediction serving
  ingestion/                 EPC download, S3 upload, Snowflake load
  orchestration/ monitoring/ db/   Prefect flow, drift checks, optional Postgres
tests/                       257 tests, all offline
```

More detail: [architecture](docs/architecture.md) · [ingestion](docs/ingestion.md) ·
[API](docs/api.md) · [evidence ledger](docs/claim-evidence-ledger.md) ·
[interview guide](docs/interview_guide.md)
