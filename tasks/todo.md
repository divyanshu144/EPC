# Todo

Working checklist for the current task. Write the plan here *before* implementing.
Mark items `[x]` as each one finishes — don't batch the updates.

Archive completed sections under "Done" rather than deleting them, so the history
of what was attempted stays readable.

---

## Current task

**Warehouse migration — Phase A: COMPLETE (2026-08-11). Phase B (dbt) is next and needs its
own spec before any code.**

Spec approved and committed: `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md`
(amended 2026-08-11, §0, after the EPC API moved to GOV.UK mid-implementation).
Branch: `feat/warehouse-ingestion`

- [x] Explore project context
- [x] Assess scope, decompose into four phases
- [x] Clarifying questions (goal, data source, architecture, warehouse, accounts)
- [x] Propose approaches, A2 selected
- [x] Present design in three sections, all approved
- [x] Write and commit spec
- [x] Spec self-review (caught the 64-day `COPY INTO` duplication bug)
- [x] Discovered mid-implementation: EPC API moved to GOV.UK — amended spec §0
- [x] Task 1: ingestion package, bearer auth, period parsing
- [x] Task 2: enumerate extracted sources by period
- [x] Task 3: period-keyed S3 uploader
- [x] Task 4: Snowflake DDL and runbook (`snowflake/01`–`03`, `README.md`)
- [x] Task 5: raw DDL, header diff, quote-aware reconciliation
- [x] Task 6: wire behind `ewhei ingest`, review-fix round (deps, credential errors,
  atomicity comment, authority-coverage test rewrite) — 129 tests passing
- [x] Task 7: documentation and handoff (this task — `docs/ingestion.md`, `CLAUDE.md`,
  `HANDOFF.md`, `tasks/todo.md`, `tasks/lessons.md`)
- [ ] **Manual verification against real services — not done, needs a human with
  credentials.** See `.superpowers/sdd/2026-08-11-epc-warehouse-ingestion/task-7-brief.md`
  and `docs/ingestion.md`'s "What is and is not verified" note. All 129 tests are offline.
- [ ] Phase B (dbt staging + marts) — needs its own spec before implementation
- [ ] Phase C (Airflow cutover, Prefect retirement) — needs its own spec first
- [ ] Phase D (AI lane) — needs its own spec first; depends on Phase B landing

### Blocked on (user/human action, not code) — carried forward from Phase A

- [ ] **R8, blocking the backfill:** obtain `EPC_BEARER_TOKEN` from the EPC account page —
  cannot be done programmatically. Nothing that touches the real API works without it,
  including the manual verification above.
- [ ] **R9, still unscoped:** 2008–2024 backfill. Only 2025-01…09 monthly archives exist
  locally (`data copy/`). The full backfill needs the token plus the yearly endpoint; load
  volume into a trial Snowflake account is untested.
- [ ] Confirm the S3 bucket `epc-sw-pipeline-dev` still exists / create it in `eu-west-2`
  (confirmed available as of the 2026-08-02 spec, not yet created)
- [ ] Create the Snowflake account on **AWS / eu-west-2** to match the bucket region
- [ ] `snowflake/01`–`03` have not been run against a real account yet

### Known trap, not yet fixed (out of scope for Phase A)

- `download_one_file` (`ingestion/epc_source.py`, re-exported by `epc_download_merge_fast.py`)
  and `list_available_files` / `pick_year_from_name` (defined in
  `epc_download_merge_fast.py`) still hardcode the dead `epc.opendatacommunities.org` host.
  `ewhei download` will fail against the real API until these are updated to the new base URL
  and bearer auth already used by `ingestion/epc_source.py`'s `auth_headers()`. Nothing in
  Phase A exercises `ewhei download` end to end, so this was never forced to surface.

---

## Done

### Onboarding — 2026-08-02
- [x] STEP 1: explore codebase, map stack / entry points / test commands
- [x] Establish verification baseline (ruff, ruff format, pytest, make smoke)
- [x] Fix broken editable install (stale path from repo move)
- [x] Branch off `main` → `chore/onboarding-conventions`
- [x] Extend `CLAUDE.md` with operating rules and the five new subsystems
- [x] Create scaffolding: `tasks/`, `HANDOFF.md`, `docs/plans/`, `docs/specs/`
