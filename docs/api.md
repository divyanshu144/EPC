# FastAPI Analytics Service

## Purpose

The FastAPI service exposes generated EPC analytics for dashboard, demo, and
interview use. It is intentionally lightweight and keeps the CSV/report artifacts
as a fallback source.

Run locally:

```bash
uvicorn api:app --reload --port 8000
```

Open generated docs:

```text
http://localhost:8000/docs
```

## Data Source Behaviour

The API prefers the optional analytical database when both conditions are true:

- `ENABLE_DATABASE=true`
- `DATABASE_URL` is reachable and contains the expected summary tables

If the database is disabled or a DB query fails, endpoints fall back to generated
CSV/report artifacts where possible. DB fallback warnings are logged server-side.

The API does not require:

- live Postgres in CI
- live MLflow
- trained model objects
- full raw EPC certificate data

## Endpoints

### Health And Metadata

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
```

- `/health` reports app status, database enabled/reachable state, CSV fallback
  availability, and lightweight artifact paths checked.
- `/metadata` reports active data source mode and optional systems, including
  database, MLflow configuration, and monitoring report availability.

### Trends And Policy Summaries

```bash
curl http://localhost:8000/trends
curl "http://localhost:8000/trends?authority=Devon"
curl http://localhost:8000/policy-summary
```

- `/trends` returns yearly mean EPC/SAP and CO2 values.
- `/policy-summary` returns policy-period summary statistics.

### Authorities

```bash
curl http://localhost:8000/authorities
curl "http://localhost:8000/authority/Devon/summary"
```

- `/authorities` returns authority names, count, and source: `database` or `csv`.
- `/authority/{authority_name}/summary` returns year range, certificate count,
  latest mean EPC/SAP, latest mean CO2, and latest below-C rate.
- Unknown authorities return `404`.

### Validation And Artifacts

```bash
curl http://localhost:8000/validation/latest
curl "http://localhost:8000/artifacts?limit=25"
```

- `/validation/latest` returns the latest DB validation row or
  `reports/artifacts/validation_report.json`.
- `/artifacts` returns DB artifact rows or `reports/artifacts/registry.jsonl`.
- `limit` must be between 1 and 500.

### Model Metadata

```bash
curl http://localhost:8000/model/metadata
curl "http://localhost:8000/model/metadata?task_type=classification"
```

Returns model metrics from the optional database when available. Otherwise it
falls back to:

- `reports/tables/regression_model_performance.csv`
- `reports/tables/classification_model_performance.csv`

No trained model object is loaded by this endpoint.

### Drift And Retraining Recommendation

```bash
curl http://localhost:8000/drift/latest
curl http://localhost:8000/retraining/status
```

- `/drift/latest` returns the latest `drift_report.json` summary, drift payload,
  and retraining recommendation.
- `/retraining/status` returns the latest recommendation and explicitly reports
  `automatic_retraining=false`.

The retraining response is decision support only. It does not trigger training,
deployment, or model promotion.

## Empty States

Optional artifact endpoints return clear empty states instead of failing when no
artifact exists yet. Example:

```json
{
  "status": "unavailable",
  "source": null,
  "message": "No drift report is available."
}
```

Core data endpoints such as `/trends` still return `404` if no feature dataset is
available and no database result can be served.

## Path And Input Safety

API responses avoid exposing local absolute paths where practical. Artifact paths
are shown relative to the repo when possible, otherwise only the file name is
returned.

Authority names are matched against known authorities from the database or CSV
artifact. Unknown names return `404`.

## Future Work

- authenticated API access
- pagination for artifact/model endpoints
- richer query filters for years and authorities
- prediction endpoint backed by a promoted model
- API package split once endpoint count grows further
- dashboard queries fully served from analytical tables
