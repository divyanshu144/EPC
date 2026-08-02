# Demo Script

This script is designed for a recruiter screen, portfolio walkthrough, or senior
engineer interview. It avoids full EPC downloads and live infrastructure unless
you choose to start those services.

## 2-Minute Walkthrough

1. **Open with the project purpose.**  
   "This is a production-style EPC housing-energy analytics platform. It moves
   from raw EPC bulk data to validated analytical features, fixed-effects policy
   modelling, predictive ML, drift monitoring, FastAPI, Streamlit, and CI."

2. **Show the architecture diagram in `README.md`.**  
   Point out that CSV/report artifacts remain the source of truth, while
   PostgreSQL and MLflow are optional local services.

3. **Run the quality gate.**

   ```bash
   make ci
   ```

4. **Show orchestration without running heavy work.**

   ```bash
   ewhei orchestrate --dry-run
   ```

5. **Show monitoring and API readiness.**

   ```bash
   ewhei monitor --dry-run
   uvicorn api:app --reload --port 8000
   ```

6. **Use one API call.**

   ```bash
   curl http://localhost:8000/health
   ```

## 5-Minute Technical Walkthrough

1. **Data engineering path:**  
   Explain ingestion, controlled schema selection, chunked cleaning, and South
   West geography rollup.

2. **Analytics path:**  
   Explain policy-period feature engineering, fixed-effects policy-period models,
   event-study style outputs, and why the wording avoids overclaiming causality.

3. **ML path:**  
   Explain regression/classification tasks, property-level train/test split using
   `BUILDING_REFERENCE_NUMBER`, model artifacts, model cards, and optional MLflow.

4. **Platform path:**  
   Explain Prefect orchestration, validation/manifest/registry, optional
   PostgreSQL summaries, expanded FastAPI endpoints, and drift/retraining
   recommendation reports.

5. **Quality path:**  
   Explain Ruff, format checks, pytest, Makefile targets, GitHub Actions, and why
   tests avoid full EPC downloads.

## Demo Commands

Run all quality checks:

```bash
make ci
```

Inspect workflow orchestration:

```bash
ewhei orchestrate --dry-run
```

Inspect monitoring defaults:

```bash
ewhei monitor --dry-run
```

Preview database loading:

```bash
ewhei db-load --dry-run
```

Start the API:

```bash
uvicorn api:app --reload --port 8000
```

or:

```bash
make api
```

## API Curl Examples

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl http://localhost:8000/authorities
curl "http://localhost:8000/authority/Devon/summary"
curl http://localhost:8000/trends
curl http://localhost:8000/policy-summary
curl http://localhost:8000/validation/latest
curl "http://localhost:8000/artifacts?limit=10"
curl http://localhost:8000/model/metadata
curl http://localhost:8000/drift/latest
curl http://localhost:8000/retraining/status
```

## Synthetic Demo Data

Tiny synthetic examples live in `examples/demo_data/`. They are not real EPC
records and are not used as production data.

Useful local monitoring demo:

```bash
ewhei monitor \
  --reference examples/demo_data/reference_features.csv \
  --current examples/demo_data/current_features.csv \
  --output /tmp/ewhei_demo_drift_report.json
```

## What To Say During An Interview Demo

- "I kept the original analytics reproducible, then added platform layers around
  it without making local development heavy."
- "Postgres and MLflow are optional so CI and basic demos do not depend on live
  services."
- "The geography rollup is a correctness feature, because EPC bulk files are not
  published at the same level as the analysis authorities."
- "The retraining endpoint is intentionally a recommendation. It does not
  automatically retrain or deploy."
- "The API is demo-ready but still honest: auth, pagination, prediction, and model
  promotion are future work."

## Limitations To Mention

- EPC data is lodgement/event based, not a complete annual stock panel.
- Fixed-effects policy-period models are observational.
- Current PostgreSQL layer stores summaries and metadata, not full raw EPC data.
- MLflow tracking is optional; model registry promotion is not implemented.
- Drift monitoring is threshold based and should be treated as operational
  decision support.
