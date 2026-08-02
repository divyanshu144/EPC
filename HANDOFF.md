# HANDOFF

Written so a cold-start session can resume without re-deriving context.
Update at every checkpoint: task complete · milestone done · stop signalled · blocker ·
context ~70% · 30+ min since last update.

**Last updated:** 2026-08-02

---

## Current State

Onboarding complete and pushed. No feature work in progress.

- **Branch:** `chore/onboarding-conventions`, pushed to `origin` and tracking it.
  Two commits, both ahead of `main` — **not yet merged, no PR opened.**
  - `a15b3c7` platform layers (the pre-existing uncommitted build-out)
  - `8e66fdd` working conventions + session-tracking scaffolding
- **Excluded from the repo on purpose:** `claude_onboarding.md` is still untracked.
  It is a personal operating-manual prompt, not project code, and the repo is **public**.
  Add it deliberately if you want it published.
- `CLAUDE.md` extended with operating rules, the five new subsystems, and corrected
  environment/Docker/testing facts.
- Scaffolding created: `tasks/todo.md`, `tasks/lessons.md`, `tasks/agent_memory.md`,
  `HANDOFF.md`, `docs/plans/`, `docs/specs/`.
- **Fixed:** the editable install was pointing at `/Users/divyanshu/Desktop/EPC`, a path that
  no longer exists, so `ewhei` was dead. Re-ran `pip install -e .`; CLI verified working.

**The working tree carries substantial uncommitted work that predates this session** —
24 modified files, 10 new docs, and five new subsystems (`db/`, `ml/`, `monitoring/`,
`orchestration/`, `settings.py`). This was authored before onboarding and has not been
reviewed or committed. It is all green against the quality gate.

---

## Next Action

**Open a PR for `chore/onboarding-conventions` and merge it into `main`**, or say the work
should stay on the branch. Nothing else is queued.

```
https://github.com/divyanshu144/EPC/pull/new/chore/onboarding-conventions
```

Note the platform build-out in `a15b3c7` was authored before this session and has **not
been code-reviewed** — it passes the quality gate, but passing lint and tests is not the
same as having been read by anyone. Worth a review pass before it lands on `main`.

---

## In-Flight Files

None — no edit is half-finished.

Everything modified this session is complete: `CLAUDE.md`, `tasks/*`, `HANDOFF.md`,
`docs/plans/README.md`, `docs/specs/README.md`.

---

## Open Questions

1. **Merge strategy** — PR and merge to `main`, or hold on the branch? (see Next Action)
2. **Python version split** — local is Anaconda 3.13.5, CI pins 3.11. Leave as-is and treat
   CI as authoritative, or align local to 3.11 via a dedicated venv? Currently no failures
   trace to this, but it is a latent divergence.
3. **Stale knowledge graph** — the `code-review-graph` index is 4 subsystems behind
   (built at `96ce4e3`, 75 nodes / 12 files). Rebuild before relying on impact analysis.
4. **Pre-existing dependency conflicts** — `fastmcp` wants `websockets>=15.0.1` (have 13.1),
   `typer-slim` wants `typer>=0.23.1` (have 0.17.5). Unrelated to this project and harmless
   to the suite, but they surface on every `pip install`.

---

## Verification Baseline

Last run 2026-08-02, after the editable reinstall — all green:

```
ruff check .            -> All checks passed!
ruff format --check .   -> 51 files already formatted
python -m pytest        -> 68 passed, 948 warnings in 4.81s
make smoke              -> ewhei --help + 3 dry-run paths OK
```

Reproduce with `make ci` then `make smoke`. The 948 warnings are pre-existing deprecation
noise from `shap`, `pydantic.v1`, and `prefect` — not regressions.

**Note:** a green `pytest` does not prove the CLI works — `tests/conftest.py` patches
`sys.path`. Always pair it with `make smoke`.
