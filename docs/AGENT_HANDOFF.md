# EPC active goal handoff

Updated: 2026-10-09. Goal tracker reports `usageLimited`; objective is not complete.

The user wants the whole credibility programme completed and wants this handoff
kept current so another coding agent can continue before a five-hour rate limit.
Do not depend on being able to observe the remaining quota. Update at milestones.

## Scope and authorization

See `docs/credibility-plan.md` for the full requirements. Implementation and tests
are authorized. Preserve legacy data/reports and pre-existing untracked work.
No messages to stakeholders without explicit authorization. Prepare concrete
deployment work before requesting necessary access or consequential approvals.
Do not claim adoption, business impact, causal effects, cloud success or complete
reproducibility without direct evidence. No subagents unless explicitly authorized.

## Environment

- Workspace: `/Users/divyanshu/Desktop/All Projects/EPC`.
- Working Python: `/opt/anaconda3/bin/python` (3.13.5); pandas, sklearn, pyarrow,
  MLflow, pytest, linearmodels, xgboost, shap and ruff are installed.
- `python3` resolves to Homebrew Python without pandas; do not use it for pipeline work.
- Python 3.11.9 under `.pyenv` has scientific libraries but lacks pytest and MLflow.
- `.venv-reproduce` created. Pinned install failed with sandbox DNS errors; clean-environment rerun remains outstanding. See next steps.
- Initial Git HEAD: `047c533623e7d69868bf1105a4b20a3e384d1125`.
- Initial user-owned untracked files: `Dissertation_report_Latest.pdf`,
  `PROJECT_DEEPDIVE.md`, `RESUME_NUMBERS.md`, `scripts/pipeline/fast_wcb.py`.

## Implemented this session

1. `scripts/pipeline/build_release.py`: new non-overwriting release builder.
   Reads annual raw files in chunks, hashes sources, selects South West, checks
   certificate uniqueness across chunks/files using SQLite, records sequential
   exclusions, retains missing measurements, excludes years outside 2008–2025,
   writes Parquet, computes authority-year new-build share for policy use only,
   records code hashes/environment/counts/missingness in manifest. Failed builds
   retain `INCOMPLETE`; final manifest is written only after checks.
2. `full_load_schema.resolve_usecols`: fixed legacy ID names not resolving when
   alias names were absent. Added regression test.
3. `src/ew_housing_energy_impact/ml/evaluation.py`: deterministic hashed property
   train/validation/test assignments; unidentified-property exclusions; purged
   chronological partitions; independent overlap assertions; deterministic
   whole-property sampling; grouped CV; fold-local imputation and encoding.
   Wired into the new benchmark; legacy CLI still needs migration.
4. `tests/test_release.py`, `tests/test_evaluation.py`: meaningful checks for
   missing 2008 floor area, excluded 2026, duplicate IDs across files, exact
   accounting, output preservation, grouping, temporal purging and train-only
   imputation. Documentation plan added.
5. `scripts/pipeline/verify_release.py`: independent Parquet checks for hashes,
   record counts, uniqueness, missingness, bounds/labels, property counts,
   full 2008–2025 coverage, authorities and exclusion reconciliation.
6. `scripts/pipeline/benchmark_release.py`: fresh-run benchmark with protocol,
   saved property and CV assignments, actual local MLflow, grouped tuning,
   validation-only model selection, selected/baseline refits, test predictions,
   subgroup metrics and paired property bootstrap. Default development caps:
   200,000 training rows and 50,000 validation rows. Full test partition retained.
   Internal MLP early stopping disabled. Eight non-dummy regression configurations
   plus mean/median dummy baselines; four non-dummy classifiers plus dummy prior.
7. `tests/test_benchmark_release.py`: real SQLite MLflow integration including
   grouped grid search, tamper rejection, early-stopping settings, strict parameter
   serialization and paired-bootstrap checks. `docs/claim-evidence-ledger.md` and
   `docs/pilot-protocol.md` created. No pilot participants or outcomes yet.
8. `scripts/pipeline/verify_benchmark.py`: independent completion audit: artifact
   and code-snapshot hashes, dataset identity, split/CV assignments, development
   samples, validation-only selection, labels, reloaded model predictions,
   recomputed metrics and actual finished MLflow status. Tested in the real
   MLflow integration fixture. It checks bootstrap interval structure but does
   not recompute all bootstrap replicates; this limitation is explicit.

## Verified so far

- Existing full pytest suite passed before edits (241 executed tests).
- Release/schema tests: 14 passed.
- Evaluation tests: 5 passed.
- Ruff checks passed for new release/evaluation modules and associated tests.
- Full current suite: **255 passed**, 0 failures/errors/skips in
  `reports/evidence/tests-credibility.xml`. This passed outside the sandbox after
  four Prefect tests could not open their SQLite database inside it. The earlier
  sandbox failure remains in `tests-credibility.log`; XML is the successful rerun.
  A subsequent targeted benchmark integration run passed all 4 tests after adding
  grouped grid-search coverage. Actual remote CI execution remains outstanding.
- Raw first 10,000 2008 records have no floor-area measurements. The old cleaner
  drops these through its numeric range filter. New release retains missingness.
- Completed full release has verified 75,193 retained 2008 South West records;
  early-year data loss is real, not merely stale reporting.
- Legacy model metrics predate the legacy current feature file. Do not reuse
  those metrics for the new release.

## Live work — check before restarting anything

Release session `70207` **finished successfully**. The release and independent
verification are complete. Source input: 30,153,553 national records; selected
South West input: 2,922,241; retained: 2,776,781; identified unique properties:
1,905,073; unidentified-property rows: 70,293. All 14 authorities and all years
2008–2025 present. 97,873 selected 2026 rows excluded. See claim ledger for bounds.

Release dataset SHA-256:
`978331eefcd30a3b482d992ab2a8b2977f5d5c0705df2d3037276c61ace12be6`.
All 6 hashed implementation files were verified and copied into release `code/`.
Independent verification: `reports/evidence/release-v1-verification.json` (`ok: true`).
Source hashes were recorded during build, not independently rehashed in verification.

**Property benchmark completed and independently verified.** Session `87158`
finished with exit 0. Outputs: `reports/benchmarks/property-v1/`.
Verification: `reports/evidence/benchmark-property-v1-verification.json` (all checks
true), including source labels, split separation, artifact hashes, recalculated
metrics, reloaded predictions and actual finished MLflow status.
Actual parent MLflow run: `8e0a1377e51f4cf78925f80ea91444cc` in
`reports/evidence/mlflow/mlflow.db`.

Frozen split: 1,625,583 train-universe records (1,144,581 properties), 539,596
validation-universe records (379,769 properties), 541,309 test records (380,723
properties). Development uses 200,000 train and 50,000 validation rows sampled
as whole properties. Excludes 70,293 unidentified-property records.

Selected tuned MLP: test RMSE 9.4291538065, MAE 6.6411770956, R² 0.6194617478;
linear baseline RMSE 9.9853299184; mean baseline RMSE 15.2853953805.
Selected XGBoost: ROC-AUC 0.8857552102, F1 0.8359568192;
logistic baseline ROC-AUC 0.8784031562; dummy ROC-AUC 0.5.
500 paired property-bootstrap replicates: MLP RMSE 95% interval
[9.3986254171, 9.4616645918]; linear-minus-MLP RMSE interval
[0.5456286935, 0.5671630913]; XGBoost AUC interval
[0.8847127232, 0.8866799921]; XGBoost-minus-logistic AUC interval
[0.0070929147, 0.0075872221]. Conditional on fitted models, not retraining uncertainty.
MLP tuning emitted maximum-iteration warnings; do not claim convergence.

`scripts/pipeline/report_benchmark.py` completed (session `66897`, exit 0).
Outputs: `reports/model_reports/property-v1/`, including model cards, held-out
permutation importance (5,000 rows, 5 repeats), SHAP permutation explanations
(128 rows, 50 development background rows), figures and hashed report manifest.
SHAP additivity checks passed. Encoded-feature perturbations can create unrealistic
combinations; these are model explanations, not causal effects. Figure visual QA
is still outstanding. Ruff checks passed for the report script.

**Chronological benchmark completed and independently verified.** Session `23970`
finished exit 0; `reports/evidence/benchmark-chronological-v1-verification.json`
is all true, including chronology and reloaded-model predictions. Actual MLflow
run `c2303c538108439ba40ac2164510a5e8`. Test: 154,680 certificates, 151,892
properties from 2025. Training before 2024, validation 2024, overlap purge 212,435
rows; 70,293 unidentified rows excluded. MLP RMSE 9.7571324574 versus linear
10.2117944177; XGBoost AUC 0.8507698382 versus logistic 0.8455843408.
Chronological model cards/explanations generated successfully in
`reports/model_reports/chronological-v1/` (report session `29092` finished).
All eight property/chronological PNG plots visually inspected and legible.

**Live full clean-environment rerun: session `27951`.**
Pinned installation succeeded with network approval (session `98485`); `pip check`
passed, freeze saved at `reports/evidence/reproduction-environment.txt`. pandas
and SHAP built from source for Python 3.13. Direct numerical versions match;
transitive dependencies differ from Anaconda. Launched:

```sh
set -o pipefail
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv-reproduce/bin/python -u -m scripts.pipeline.benchmark_release --release data/releases/epc-sw-2008-2025-v1 --run-id property-reproduce-v1 --strategy property --train-cap 200000 --validation-cap 50000 --bootstrap 500 2>&1 | tee reports/evidence/benchmark-property-reproduce-v1.log
```

Latest stage: GBR_tuned. Do NOT edit benchmark_release.py, evaluation.py or
registry.py while this run executes. Check existing process/artifacts rather than
starting a duplicate. No reproduced-training claim until complete and compared.
New `scripts/pipeline/compare_benchmarks.py` independently verifies both runs and
checks identical data/split/CV/code/configuration plus metrics and predictions at
rtol=atol=1e-8. Tolerances fixed before rerun results, not tuned after inspection.
Run it after completion; command is in `docs/verified-workflow.md`.

**New implementation:** `ewhei train` now requires explicit --release/--output
and runs the verified benchmark; old command renamed `legacy-train`. Historical
run-all/Prefect orchestration remains legacy (documented, not migrated).
`ml/serving.py` adds an independent research FastAPI factory with trusted local
model/protocol hashes, successful verification requirement, 100-item batch limit,
input validation, missing/unknown-category flags and provenance. Tests exercise
actual fitted pipelines and reject invalid inputs and tampered model files.
Local actual-model smoke: 50 sequential in-process TestClient calls passed;
`reports/evidence/serving-local-smoke.json`. Not network latency or deployment.

Eight targeted CLI/benchmark/serving tests passed (`tests-serving-cli.xml`).
Full suite initially had one obsolete test expecting --track on new train;
updated it to check legacy-train. **Full rerun session `93579` completed exit 0: 257 tests passed**,
`reports/evidence/tests-credibility-final.xml` (zero failures/errors/skips).
Previous failing full-run session `45005` may linger in Prefect teardown.

`docs/resume-evidence.md` now contains six ranked candidates and recommended
five, with code/output evidence, recruiter value, roles and caveats.
`docs/verified-workflow.md` documents train/verify/reproduce/serve commands.
Cloud credential presence check found no configured AWS/Snowflake/bucket values
in the examined environment variables or .env (no secrets printed). Other provider
credential mechanisms have not been exhausted. No outreach or deployment done.

The sandbox-failed full test session `37510` lingered during teardown and was
terminated with Ctrl-C after the successful outside-sandbox rerun completed.
It is not a live benchmark and must not be restarted as one.

## Next work, in order

1. Poll reproduction session 27951; independently compare completed runs using
   compare_benchmarks.py. Preserve original and failed results; do not loosen
   tolerances automatically. Full suite is complete: 257 passed.
2. Update ledger and resume claims only after reproduction results verify.
3. Review chronological model cards; all eight generated plots passed visual QA.
4. Complete remaining credibility-plan requirements: policy bootstrap/sensitivity,
   actual CI execution, real cloud access verification and deployment where
   authorized/available, and genuine stakeholder pilot evidence.
5. Historical run-all and Prefect workflows still require explicit migration or
   clear deprecation. Direct train command has been migrated and tested.
6. Keep handoff current. Do not mark complete while required local work or external
   evidence is missing. User requested continuation despite tracker usageLimited.

## Known risks to review

- Current new release range bounds are declared cohort choices, not proof that
  excluded source values are erroneous. Missing numeric values remain missing.
- Whole-population new-build share MUST NOT be used as a predictive model input.
- UPRN linkage quality is limited; synthetic IDs do not identify unique homes.
- Legacy `train_models.py` still has row-based CV/test-based model ranking; the new train CLI bypasses it, but old run-all/Prefect workflows still invoke it.
- Saved policy WCB p-values are blank; no completed bootstrap claim.
- Local MLflow is verified on the full property benchmark. Cloud integrations
  and deployment remain unverified. Legacy analytics DB has five empty tables.
- An asynchronous question asks whether the user has a housing/energy/university
  pilot contact. Answer was pending at handoff update. Do not treat silence as
  permission for outreach; continue independent implementation.
