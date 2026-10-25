# EPC evidence programme

This programme is in progress. Existing reports are historical outputs, not evidence
for a new benchmark. Never combine their metrics with a newer dataset's row count.

## Completion requirements

1. Versioned release from hashed source files: national input counts, South West
   selection, global certificate uniqueness within that selection, exclusion
   accounting, missingness, property counts, geography and year coverage.
2. Frozen property-level development and test partitions, grouped model selection,
   train-only learned preprocessing, explicit unidentified-property exclusions,
   and a separate chronological evaluation.
3. Common regression/classification benchmark with dummy and linear baselines,
   saved predictions, property-resampled uncertainty, subgroup results and a
   model selected without consulting the final test set.
4. Actual MLflow runs and immutable run manifests connecting source/data/split
   hashes, code, environment, parameters, fitted pipelines and generated reports.
   Demonstrate a clean-environment rerun.
5. Meaningful regression and integration tests, local results and a recorded CI run.
6. Verified bootstrap and sensitivity analysis for policy associations. Causal
   effects require an independently justified identification design.
7. Actual cloud integration evidence and a reviewed deployed application with
   tested prediction serving and documented operational measurements.
8. Stakeholder pilot protocol, actual participation and measured outcomes before
   any adoption or business-impact claim. No outreach without user authorization.
9. Final claim ledger and role-specific resume bullets linked to the above evidence.

## Decisions and findings

- Primary release scope: lodgements in 2008–2025; partial 2026 is excluded and counted.
- Missing numerical measurements are retained, not converted to a plausibility
  failure. Learned imputation belongs inside model selection folds.
- Range filters are analysis-cohort choices, not claims that every excluded EPC
  is erroneous. Include missingness and filtering sensitivity in later analyses.
- Initial raw-file inspection found all 10,000 sampled 2008 floor-area values
  missing. The former floor-area filter excludes these records. Full release
  accounting must quantify this rather than infer full-year coverage from a sample.
- Existing saved model metrics predate the current 2,675,904-row feature file.
- Existing tests passed locally on the initial worktree; a fresh final result
  and CI execution are still required after implementation.
- Pre-existing untracked files, including `fast_wcb.py`, are user work and must
  not be overwritten or treated as newly verified implementation.

## Evidence status

All completion requirements remain open until their artifacts have been inspected.
The new release command writes to a fresh directory and preserves legacy outputs.
