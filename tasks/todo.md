# Todo

Working checklist for the current task. Write the plan here *before* implementing.
Mark items `[x]` as each one finishes — don't batch the updates.

Archive completed sections under "Done" rather than deleting them, so the history
of what was attempted stays readable.

---

## Current task

**Warehouse migration — Phase A design. PAUSED 2026-08-02 at user's request.**

Spec approved and committed: `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md`
Branch: `feat/warehouse-ingestion`

Brainstorming is complete. Nothing has been implemented — no code, no data downloaded.

- [x] Explore project context
- [x] Assess scope, decompose into four phases
- [x] Clarifying questions (goal, data source, architecture, warehouse, accounts)
- [x] Propose approaches, A2 selected
- [x] Present design in three sections, all approved
- [x] Write and commit spec
- [x] Spec self-review (caught the 64-day `COPY INTO` duplication bug)
- [ ] **RESUME HERE:** invoke `writing-plans` against the spec
- [ ] Implement Phase A
- [ ] Phases B (dbt), C (Airflow cutover), D (AI lane) — each needs its own spec first

### Blocked on (user action, not code)

- [ ] Create `.env` with `EPC_EMAIL` + `EPC_API_KEY` — nothing runs without it
- [ ] Run `ewhei download && ewhei clean && ewhei features` (~200–300 MB)
- [ ] Confirm S3 bucket name `epc-sw-pipeline-dev` is available in `eu-west-2`
- [ ] Create the Snowflake account on **AWS / eu-west-2** to match the bucket region

---

## Done

### Onboarding — 2026-08-02
- [x] STEP 1: explore codebase, map stack / entry points / test commands
- [x] Establish verification baseline (ruff, ruff format, pytest, make smoke)
- [x] Fix broken editable install (stale path from repo move)
- [x] Branch off `main` → `chore/onboarding-conventions`
- [x] Extend `CLAUDE.md` with operating rules and the five new subsystems
- [x] Create scaffolding: `tasks/`, `HANDOFF.md`, `docs/plans/`, `docs/specs/`
