# Lessons

Chronological log. After ANY correction or non-obvious discovery, append an entry:
**what broke** / **root cause** / **what to do next time**. Newest at the bottom.

---

## 2026-08-02 — `ewhei` broken by a repo move; tests hid it

**What broke.** Every `ewhei` invocation failed:

```
File "/opt/anaconda3/bin/ewhei", line 5, in <module>
    from ew_housing_energy_impact.cli import main
ModuleNotFoundError: No module named 'ew_housing_energy_impact'
```

`pip show ew-housing-energy-impact` still reported the package as installed, and
`ewhei` was still on `PATH`, so nothing looked wrong until the CLI was actually run.

**Root cause.** The editable install records an absolute source path:

```
Editable project location: /Users/divyanshu/Desktop/EPC          <- no longer exists
Actual project:            /Users/divyanshu/Desktop/All Projects/EPC
```

The repo was moved into `All Projects/` after `pip install -e .`. The
`__editable__.ew_housing_energy_impact-0.1.0.pth` finder kept pointing at the old
location, so the dist-info survived while the import path did not.

**Why it stayed hidden.** Three separate mechanisms mask it:
- `tests/conftest.py` inserts repo root + `src/` onto `sys.path` → all 68 tests pass.
- Makefile targets (`smoke`, `demo-dry-run`, `validate`) hardcode `PYTHONPATH=src:.`.
- CI installs fresh on a clean runner, so it never reproduces.

So the entire documented entry point was dead while every quality gate stayed green.

**What to do next time.**
- After moving or renaming the repo directory, re-run `pip install -e .` immediately.
- Never treat a green `pytest` as proof the CLI works. `make smoke` is the CLI check.
- When a `ModuleNotFoundError` contradicts `pip show`, check
  `pip show -f <pkg> | grep "Editable project location"` before anything else.
- Reinstalling pulled in ~20 transitive packages (prefect 2.20.25, sqlalchemy 2.0.35,
  psycopg 3.3.4) and reported pre-existing conflicts (`fastmcp` wants websockets>=15.0.1,
  have 13.1; `typer-slim` wants typer>=0.23.1, have 0.17.5). Those conflicts are unrelated
  to this project and did not affect the suite — re-verified 68 passed afterwards.

---

## 2026-08-02 — Code knowledge graph was stale enough to mislead

**What broke.** Nothing yet, but the `code-review-graph` MCP reported 75 nodes across
12 files, built at commit `96ce4e3`. The working tree has ~40 source files.

**Root cause.** The graph was built before the `db/`, `ml/`, `monitoring/`, and
`orchestration/` subsystems were added, and those changes are still uncommitted.

**What to do next time.** Check `list_graph_stats_tool` (node count, last-updated commit)
against `git log -1` before trusting the graph for impact analysis. A stale graph does not
error — it silently returns an answer that omits whole subsystems, which is worse than a
failed query. Rebuild, or fall back to Grep, when the commit doesn't match.

---

## 2026-08-11 — `ewhei download` failed with a JSON decode error; the EPC API had moved

**What broke.** `ewhei download` failed:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Root cause.** Not our code. `epc.opendatacommunities.org` now 301-redirects to
`get-energy-performance-data.communities.gov.uk`, the site root of a rebuilt GOV.UK service.
`requests` follows the redirect, lands on an HTML landing page, and gets back HTTP 200 — so
`resp.raise_for_status()` passes cleanly, and `.json()` only fails afterwards, at char 0,
because the body was never JSON to begin with. `raise_for_status()` proved the request
didn't error; it proved nothing about what came back.

**Three consequences, all of which reshaped the ingestion design:**

1. **Auth changed.** `Basic base64(email:api_key)` is replaced by `Bearer <token>`, minted
   from the account page on the new service — no programmatic way to obtain it.
   `EPC_EMAIL`/`EPC_API_KEY` are obsolete.
2. **Granularity changed.** There are no more per-local-authority archives. The bulk
   endpoint `GET /api/files/domestic/csv` serves one national archive (~2.92 GB), split by
   period, regenerated monthly — not one file per LA. S3 keys therefore became
   period-keyed (`raw/certificates/<period>/certificates.csv`) instead of LA-keyed, and
   South West filtering moved downstream into dbt staging, since there is no longer a
   per-LA file to filter at ingestion.
3. **Redirect behaviour is correct, not a bug.** `requests` strips the `Authorization`
   header across this cross-host redirect. That is the right behaviour here: per the API's
   documentation the 302 leads to a presigned S3 URL carrying its own credentials in the
   query string, and forwarding a bearer token to S3 would be both wrong and a credential
   leak. **Not independently observed** — the endpoint returns 403 without a bearer token,
   which nobody on this task has obtained yet, so this is what the docs describe rather than
   something we watched happen. The 301 from `epc.opendatacommunities.org` above, by
   contrast, was confirmed directly.

**What to do next time.**
- A passing `raise_for_status()` does not prove you got JSON back. Check `Content-Type`
  (or just try the parse and expect it might not be JSON) before trusting a 200.
- Treat a cross-host redirect as a signal that the endpoint moved, not something to follow
  blindly and paper over. It was the first real evidence something upstream had changed.
- `scripts/pipeline/epc_download_merge_fast.py`'s `list_available_files` and
  `pick_year_from_name` still hardcode the dead `epc.opendatacommunities.org` host, and
  `download_one_file` (defined in `ew_housing_energy_impact.ingestion.epc_source`,
  re-exported there) does too. Nothing in Phase A exercises `ewhei download` end to end, so
  this is a live trap for the next person who runs it expecting it to work.
