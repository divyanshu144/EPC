# Synthetic Demo Data

These files are tiny synthetic examples for demos and tests. They are not real
EPC records and should not be used for analysis.

Example monitoring command:

```bash
ewhei monitor \
  --reference examples/demo_data/reference_features.csv \
  --current examples/demo_data/current_features.csv \
  --output /tmp/ewhei_demo_drift_report.json
```

The JSON and JSONL files mirror the shape of generated artifacts so API fallback
behavior is easy to discuss without requiring full EPC data.
