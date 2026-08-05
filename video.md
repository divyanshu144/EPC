# JobFit — Video Evidence Pack

**Generated:** 2026-08-05 · **HEAD:** `f2845a3` · **Working tree:** clean
**Status:** local-only (`.git/info/exclude`) — not tracked, will not be committed.

Every claim below carries a file:line, commit SHA, tag, or date. Items I could not
evidence are marked **UNVERIFIED** rather than inferred.

> ### ⚠️ Read this first: this session changed the repo
> Earlier today I fixed the eval severity-gating defect described in §1c and merged it
> to `main`. Consequences for anything you say on camera:
> - `reports/evals/latest.json` now reads `generated_at: 2026-08-05`, **because I ran
>   it today** — not because the suite has been running regularly. Before today it read
>   `2026-06-15`.
> - The eval module line counts below are **post-fix**. My commits added 23 lines.
> - The defect is real and historically true, but it is **fixed as of `f2845a3`**.
>   Describe it in the past tense.

---

# SECTION 1 — The three things the interviewer asked about

## 1a. Typed contracts at agent boundaries

### Where schemas are defined

All agent output models live in `backend/schemas.py`:

| Model | Lines | Agent |
|---|---|---|
| `ValidationWarning` | `schemas.py:54-62` | meta-field on 6 models |
| `JobParserOutput` | `schemas.py:65-72` | job_parser |
| `MatchScorerOutput` | `schemas.py:75-80` | match_scorer |
| `GapAnalystOutput` | `schemas.py:83-86` | gap_analyst |
| `ResourcePlannerOutput` | `schemas.py:98-101` | resource_planner |
| `CoverLetterOutput` | `schemas.py:104-108` | cover_letter |
| `ResumeTailorerOutput` | `schemas.py:111-120` | resume_tailorer |
| `ExtractedProfile` | `schemas.py:209-217` | profile_extractor |
| `ColdEmailOutput` | `schemas.py:516-518` | cold_email_agent |
| `ResumeEditorOutput` | `schemas.py:635-643` | resume_editor |
| `Stage2Result` | `services/stage2.py:24-37` | discovery relevance |

### Is it enforced, or just a type hint? — **ENFORCED**

Single enforcement point, `backend/agents/base.py:216-236`:

```python
async def _call_structured(
    self, system: str, user: str, output_cls: type[T], *, label: str
) -> T:
    raw = await self._call(system, user)
    try:
        return output_cls.model_validate(_parse_json(raw))
    except (json.JSONDecodeError, ValidationError, AgentError) as e:
        self._retry_count += 1
        await self._log_retry(label)
        correction_system = self._correction_prompt(system, raw, str(e))
        raw2 = await self._call(correction_system, user)
        try:
            return output_cls.model_validate(_parse_json(raw2))
        except (json.JSONDecodeError, ValidationError, AgentError) as e2:
            raise AgentError(f"{label}: {e2}") from e2
```

`model_validate` at `:227` and `:234` are hard gates. `T = TypeVar("T", bound=BaseModel)`
(`base.py:33`) means `output_cls` must be a Pydantic model. `_parse_json` (`base.py:72-80`)
raises `AgentError` at `:78` if no JSON object is present.

### What happens on schema failure — full path

1. Failure raised at `base.py:78` / `:79` / `:227`
2. Caught `base.py:228` → `_retry_count += 1` (`:229`) → `PipelineEvent(kind="retry")` written
   fail-open by `_log_retry` (`base.py:189-201`, returns early at `:192` with no DB session)
3. Correction prompt built `base.py:231`, second call `:232`
4. Second failure → `AgentError` raised `base.py:236`. **Never a third call.**
5. Propagates to the orchestrator's per-agent `except Exception`:
   Phase 1 SSE `orchestrator.py:418-433`, Phase 1 non-SSE `:271-284`, Phase 2 `:559-580`
6. `to_user_error()` (`services/pipeline_errors.py:37-65`) maps `AgentError` →
   `code="invalid_output"` (`:61-62`)
7. SSE emits `pipeline_error` (`orchestrator.py:429-432`), `analysis.partial = True`
   (`:421`, persisted `:436`), error-bearing `JobResult` upserted

**Net behaviour: exception → caught → degraded/partial result. The pipeline continues
to the next agent; it does not abort.**

Test proof: `tests/test_agents/test_base_self_correction.py:46-51` asserts
`mock.await_count == 2  # hard cap — never a third call`.

### Which agents have it — all of them, with two exceptions

| Agent | File:line | Output type | Via |
|---|---|---|---|
| JobParserAgent | `job_parser.py:12-19` | `JobParserOutput` | `_call_structured` |
| MatchScorerAgent | `match_scorer.py:8-16` | `MatchScorerOutput` | `_call_structured` |
| GapAnalystAgent | `gap_analyst.py:8-14` | `GapAnalystOutput` | `_call_structured` |
| **ResourcePlannerAgent** | `resource_planner.py:16-114` | `ResourcePlannerOutput` | **direct `model_validate` — NOT `_call_structured`** (`:25`, `:52`) |
| CoverLetterAgent | `cover_letter.py:8-14` | `CoverLetterOutput` | `_call_structured` |
| ResumeTailorerAgent | `resume_tailorer.py:8-21` | `ResumeTailorerOutput` | `_call_structured` |
| ColdEmailAgent | `cold_email_agent.py:7-27` | `ColdEmailOutput` | `_call_structured` |
| ProfileExtractorAgent | `profile_extractor.py:7-18` | `ExtractedProfile` | `_call_structured` |
| ResumeEditorAgent | `resume_editor.py:8-37` | `ResumeEditorOutput` | `_call_structured` |
| **`_ResumeLatexAgent`** | `services/resume_latex.py:45-53` | **raw `str` — NO SCHEMA** | contract is "pdflatex compiles" |

**Two honest caveats:**
- `_ResumeLatexAgent` is a 10th `BaseAgent` subclass living outside `backend/agents/`.
  It returns raw text (`resume_latex.py:52-53`) with no typed contract at all.
- `ResourcePlannerAgent` validates but **has no schema self-correction** — it raises on
  the first bad parse (`resource_planner.py:22-27`). It is simultaneously the agent that
  makes the most LLM calls (up to 3).

### ⚠️ `validation_warnings` is dead in production

`exclude=True` on all six models (`schemas.py:72, 80, 86, 101, 108, 120`) strips it from
`model_dump()`. The only readers anywhere are `backend/evals/runner.py:105-119` (offline)
and `logger.warning` inside `_warn` (`validators.py:87`). **A `severity="error"` warning
during a live request has zero effect** — no retry, no rejection, no user surface.

Worse: `validate_resume_tailorer` **mutates**, it does not merely flag:
- `validators.py:312` — `output.skills = supported_skills`
- `validators.py:326` — `exp.company = None`
- `validators.py:351` — `education.degree = None`

This is the mechanism behind the §2 "validator deleted user data" bug.

---

## 1b. Self-correction retries

### Two distinct classes exist — actually four mechanisms at three levels

**Loop 1 — transient, SDK-internal.** `base.py:91-94`, `max_retries=settings.anthropic_max_retries`,
value `3` at `config.py:30`.

**Loop 2 — transient, outer tenacity.** `base.py:35-38`:
```python
# Outer retry layer for when the SDK's own retries give up: 529 (overloaded)
# and timeouts only. Everything else fails fast on the first attempt.
_RETRY_ATTEMPTS = 3
_RETRY_WAIT = wait_exponential_jitter(initial=1, max=4)
```
Predicate `_is_transient` (`base.py:45-49`) matches **only** HTTP 529 and timeouts.
**429 is NOT in this predicate** — rate limits fall through to the SDK layer only.
Enforced `base.py:173-187` with `reraise=True` (`:177`).

**Loop 3 — content/validation self-correction.** `base.py:216-236` (quoted in §1a).

**Loop 4 — content/quality, resource_planner only.** `resource_planner.py:29-63`,
`CONFIDENCE_THRESHOLD = 0.7` (`:13`). Self-scores its output, regenerates only
low-confidence gaps. Max 3 LLM calls (`:23`, `:34`, `:48`).

Separation is documented at `base.py:222-223`: *"Transient errors (rate limit, timeout,
connection) raised inside _call propagate untouched — the SDK owns those retries; we add
no layer here."* Verified by `test_base_self_correction.py:54-61` — a `RateLimitError`
gives `await_count == 1`.

### Is the error fed back to the model? — **YES, genuine self-correction**

`base.py:203-214`:
```python
@staticmethod
def _correction_prompt(system: str, prior_raw: str, error: str) -> str:
    return (
        f"{system}\n\n"
        "## CORRECTION\n"
        "Your previous response failed schema validation:\n"
        f"{error}\n"
        "Your previous response was:\n"
        f"{prior_raw[:500]}\n"
        "Return ONLY valid JSON matching the schema above.\n"
        "Fix exactly that problem; change nothing else."
    )
```

Both `str(e)` (the actual Pydantic/JSON error) **and** the first 500 chars of the
offending response are fed back as the retry's system prompt (`base.py:231-232`).
This is not a repeated identical call.

Test proof `test_base_self_correction.py:40-43`:
```python
second_system = mock.await_args_list[1].args[0]
assert "## CORRECTION" in second_system
assert "Input should be a valid integer" in second_system  # the actual error text
assert '{"x": "not-an-int"}' in second_system  # prior raw fed back
```

### Retry caps — confirmed

| Layer | Cap | Constant | Enforced at |
|---|---|---|---|
| SDK internal | 3 retries | `config.py:30` | `base.py:93` |
| Outer transient | 3 attempts | `_RETRY_ATTEMPTS` `base.py:37` | `base.py:175` |
| Content self-correction | **2 calls total** | none — structural | `base.py:225` + `:232` |
| resource_planner regen | 1 regeneration | none — structural | `resource_planner.py:46-63` |
| pdflatex self-correction | 2 attempts | none — structural | `resume_latex.py:118-128` |

⚠️ **The content cap is not a tunable constant** — it is the shape of a single nested
`try`. Saying "the retry cap is configurable" would be wrong.

⚠️ **Worst-case request multiplication:** `2 (correction) × 3 (tenacity) × 4 (SDK) = 24`
HTTP requests for one `_call_structured`, all billed. Only 529/timeout multiplies that way.

### On cap exhaustion

- **Content** → `AgentError` (`base.py:236`) → partial result, pipeline continues
- **Transient** → SDK exception reraised (`base.py:184`), `_record_failure` bumps the counter
- **resource_planner regen** → `except Exception: pass` (`:62-63`), keeps pass-1 output;
  self-check failure → `confidences = {}` (`:37-38`), treated as fully confident
- **resume_editor** → one-shot Sonnet fallback on service failure only (`resume_chat.py:67-70`);
  schema failures explicitly do **not** trigger it (`:57-59`)

### `resume_latex.py` — is the LaTeX error fed back? **YES**

`resume_latex.py:66-78`:
```python
system += (
    "\n\nYour previous .tex FAILED to compile with pdflatex. Fix the LaTeX "
    "error (keep the same tailoring intent). pdflatex log tail:\n" + correction_log
)
```
Log tail is the last 2000 chars — `_LOG_TAIL_CHARS = 2000` (`resume_latex.py:17`),
applied `:108`. Driven from `tailor_resume_pdf` (`:111-128`); exhaustion raises
`ResumeCompileError` (`:41-42`, `:128`) — hard failure, no degraded PDF.

**Two caveats worth owning:** the retry re-sends the *original* `latex_source`, not the
broken `.tex` the model produced — so the model sees the error but not the document that
caused it. And pdflatex puts its `!` error line where the error occurred, not at the end,
so a long log can truncate the actual error out of the 2000-char tail.

---

## 1c. The eval harness

### Inventory

`backend/evals/` — **549 lines** (post my fix; **526** at session start `934ec3f`):

| File | Now | At `934ec3f` | What it does |
|---|---:|---:|---|
| `__init__.py` | 13 | 13 | Docstring only. States: *"Feedback capture is built; the scorer that consumes it is not yet implemented."* |
| `dataset.py` | 58 | 53 | Fixture Pydantic models + `load_eval_cases()` |
| `faithfulness.py` | 111 | 111 | Deterministic grounding checks for the chat editor |
| `runner.py` | 274 | 264 | `run_deterministic_eval` / `evaluate_case` / report writer |
| `validators.py` | 367 | 359 | Six `validate_*` functions — also imported by live agents |

`tests/test_evals/` — **923 lines** across 8 files: `test_consistency.py` (106,
integration-marked, deselected), `test_dataset_loading.py` (29), `test_dataset_regression.py`
(57), `test_faithfulness.py` (99), `test_grounding_regression.py` (58),
`test_live_eval_guard.py` (21), `test_prompt_versions.py` (29), `__init__.py` (0).

### What faithfulness checks, and deliberately does not

Module docstring `faithfulness.py:1-13`, verbatim:

> Flag-only and NON-mutating: unlike validate_resume_tailorer (which omits unsupported
> items — correct for the automated tailorer), the chat path surfaces warnings and lets
> the user keep, dismiss, or undo. **The user is the ground truth for their own resume.**
> Zero LLM calls.
>
> Deliberately UNCHECKED at this layer: job titles, dates, degrees, and project names.
> Literal matching would false-positive on nearly every legitimate rephrasing ("SWE II"
> to "Software Engineer", "2022-2024" to "Mar 2022 - Present").

**Does check** (`faithfulness.py:64-111`): em/en dash style (`:69-74`), unsupported
employer (`:80-89`), unsupported institution (`:90-99`), unsupported skill (`:100-108`),
unsupported metric (`:110`).

The metric-check reasoning, `faithfulness.py:40-47`:
> A number the profile never mentions is the classic resume hallucination. Literal
> digit-string match: '40%' is supported by '40' anywhere in the source; 'forty percent'
> in the source does NOT support '40%'. Digit-boundary guarded so a fabricated '40%' is
> not wrongly "supported" by an unrelated longer number (e.g. '140')… Known limitation:
> comma-grouped numbers ('1,200') tokenize as separate runs… **Deterministic > clever.**

⚠️ `faithfulness.py:78` — `if not cv_text: return warnings`. **With an empty source every
grounding check silently passes.** Codified by `test_faithfulness.py:82-84`.

⚠️ **"Layer 3, the optional LLM judge" does not exist.** `resume_faithfulness_judge_enabled:
bool = False` (`config.py:55`) is the only occurrence outside the docstring. No judge code.

### Zero LLM calls — **CONFIRMED**

No `anthropic`, `httpx`, or even a single `await` anywhere in `backend/evals/`. Every
function is synchronous. `runner.py` imports only model-name constants and the prompt-hash
helper (`:11-12`), never a client. Scoring is pure string matching (`runner.py:220-250`).

### Prompt version pinning

**SHA-256.** `agents/prompt_versions.py:28-29`:
```python
def compute_prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
```
Short form is first 12 hex chars (`prompt_versions.py:47`). `PromptVersion`
(`:8-25`) carries `agent_name`, `model`, `prompt_name`, `prompt_path`, `prompt_hash`,
`prompt_version`.

Stored in three places: per LLM call in the DB (`base.py:126-132`, `:156-164` →
`LLMCall`); in the eval report (`runner.py:80` → `reports/evals/latest.json`); and
Stage-2 hashes a template constant (`services/stage2.py:43-51`).

**Model is pinned per agent — but in two places that can drift.** `runner.py:194-199`
hardcodes its own map while production values come from class attributes.
⚠️ Nothing enforces they stay in sync, **and the orchestrator mutates `agent.model` at
dispatch** (`orchestrator.py:254`) — so the model in the report is a static declaration,
not necessarily what ran.

⚠️ `prompt_path` is written as an **absolute developer-machine path**
(`/Users/divyanshu/Desktop/…`), so the report is not reproducible across machines.

### Eval cases — 5

`tests/fixtures/evals/jobfit_eval_cases.json`: `ai_software_engineer_startup`,
`applied_ai_engineer_healthtech`, `backend_engineer_platform`, `data_ml_engineer_analytics`,
`forward_deployed_engineer_ai`. Pinned by `test_dataset_loading.py:10-17`.

### ⚠️ THE DEFECT — real, and fixed today

**Before (evidence preserved from the 2026-06-15 report):**
```
generated_at: 2026-06-15T14:14:47  |  passed: True  |  failed_cases: 0
case[0] warnings: [{'agent': 'cover_letter', 'rule': 'body_too_short',
                    'detail': 'body has 86 words, minimum is 150',
                    'severity': 'error'}]
```
All five cases carried a `severity: "error"` warning while the run reported
`passed: True, failed_cases: 0`.

**Why:** `runner.py:170` computed `passed=not failures`. Warnings were collected,
serialised into the report, and **never consulted**. Severity was discarded entirely.
The suite was structurally incapable of failing on a validator finding.

**Compounding — three conflicting thresholds for one property:**

| Source | Minimum |
|---|---:|
| `validators.py:241` (production validator) | 150 |
| `dataset.py:13` (`CompletenessRules` default) | 120 |
| Fixture, per case | 75, 70, 70, 65, 70 |

Mock bodies were 86, 77, 76, 75, 75 — **each fixture threshold sat 5–10 words below its
own mock.** The bar was lowered to clear what was already there.

**Fixed 2026-08-05.** `2e3428a` promotes `severity="error"` to failures;
`312050d` collapses the three thresholds into `COVER_LETTER_MIN_WORDS = 75`;
merged `f2845a3`. Verified the gate now fires: truncating a mock to 40 words gives
4/5 and exit 1.

### CI wiring — nuanced

`.github/workflows/ci.yml:41-43` runs `pytest tests/ -v`. **No `make eval-deterministic`,
no `make eval-live`, no `scripts/run_evals.py` anywhere in CI.** Those exist only as
Makefile targets (`Makefile:20-27`), and `make check` does not include them either.

**However** the deterministic eval *does* execute transitively:
`tests/test_evals/test_dataset_regression.py:15-33` calls `run_deterministic_eval` and
is collected by `pytest tests/`. So the eval **logic** is CI-gated; the Makefile targets
and the committed report artifact are not. `test_consistency.py` is genuinely excluded
(integration-marked, `pyproject.toml:8`).

### Last actual run

`2026-08-05T05:17:14` — **because I ran it today.** Prior to that: **2026-06-15**,
i.e. ~7 weeks stale, spanning v1.3.0 → v1.6.0 including the entire resume editor.

### `--live` is a stub

`scripts/run_evals.py:31-33`: even with `RUN_LIVE_EVALS=1` it prints *"Live eval runner
is not implemented yet; no Anthropic calls were made"* and returns 0. **There is zero
coverage of actual model output quality.**

---

# SECTION 2 — Q1 evidence: feedback to production

## The resume thread

### 2.1 One-page truncation — `ef113b7`, 2026-06-23, **v1.1.0**

**Original `_limit_words`** (`git show 4a10576:backend/services/resume_latex_template.py:79-83`):
```python
def _limit_words(value: str, max_words: int) -> str:
    words = _clean(value).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "."
```

It kept the first N tokens, stripped trailing punctuation off the **last surviving word**,
and appended a literal `"."` — bolting a fake terminator onto a non-terminal word.
Applied to every bullet (24 words), project description (10), and summary (52).

`tasks/lessons.md:321-345`: *"producing 'giving the team better.' across every bullet…
The output-token cap was NOT the cause (output was not end-clipped); a code-side word
trimmer was."*

**Current** (`resume_latex_template.py:80-106`): splits on sentence boundaries via
`re.split(r"(?<=[.!?])\s+", cleaned)`, drops whole trailing sentences, and keeps the
first sentence whole if it alone exceeds budget.

### 2.2 The validator that deleted user data — same commit `ef113b7`, **v1.1.0**

**Fields destroyed:** `skills` (over-pruned) and `education[].degree` (nulled).
`experience[].company` uses the same mechanism.

**The deleting code** (`git show ef113b7^:backend/evals/validators.py`):
```python
def _extract_cv_text(profile_text: str | None) -> str:
    marker = "## CV Text"
    if marker in profile_text:
        return profile_text.split(marker, 1)[1]
    return profile_text
```
```python
        output.skills = supported_skills          # destructive rebind
        ...
                education.degree = None            # destructive
```

**Root cause:** evidence was sliced to **only the pypdf-extracted CV blob**. pypdf
extraction is lossy (degree lines split across columns), and skills typed into the
Profile Review form live *before* that marker — invisible to the validator, therefore
deleted. Symptom: *"the generated resume dropped the candidate's degree and showed only
~5 skills."*

**The fix** — `_evidence_text` widens evidence to the whole profile:
> Evidence for factual checks = the whole candidate profile the resume was built from
> (YAML core_skills + Profile Review + CV Text), not just the CV blob. **User-curated
> structured data … is authoritative;** the pypdf-extracted CV text is only one source
> and is often lossy.

### 2.3 LaTeX ligature / ATS — `22e44a9`, 2026-06-24, **v1.1.1**

The PDF *image* was correct; the **text layer** dropped f-ligature glyphs on extraction:
`"fixed" → "xed"`, `"flagged" → "agged"`, `"MLflow" → "MLow"`. ATS parsers read the text
layer, so they saw garbage.

Fix in `assets/latex-format.tex`:
```latex
\input{glyphtounicode}
\pdfgentounicode=1
\DisableLigatures{encoding = *, family = * }
```

⚠️ **Sequel:** the accompanying `\usepackage{lmodern}` caused a **~11-day total prod
outage** — `lmodern` is a separate Debian package, so pdflatex failed in the deployed
image and every PDF download 503'd from 2026-06-24 until `4c9415f` (2026-07-05, v1.3.1).
Masked because the frontend silently fell back to DOCX.

### 2.4 OCR fallback — `f851f9b`, 2026-07-18, **v1.4.0**

`backend/services/cv_parser.py:14-40`:
```python
_OCR_TRIGGER_CHARS = 20
...
    text = await loop.run_in_executor(None, _extract_pdf_sync, pdf_bytes)
    if len(text.strip()) >= _OCR_TRIGGER_CHARS:
        return text
    try:
        ocr_text = await loop.run_in_executor(None, _extract_pdf_ocr_sync, pdf_bytes)
    except Exception:
        logger.warning("OCR fallback failed for scanned/image-only PDF", exc_info=True)
        return text
    return ocr_text if len(ocr_text.strip()) > len(text.strip()) else text
```

- **Trigger:** `_OCR_TRIGGER_CHARS = 20` (`:17`), condition at `:33`
- **Keep-if-longer:** `:40`, strictly greater — a tie keeps pypdf text
- **Exception handling:** `:37-39`, bare `except Exception`, WARNING with `exc_info`,
  returns original text. Fail-open: broken tesseract never breaks upload.

Both extractors run in `run_in_executor` (CPU-bound on an async path).

### 2.5 Download-vs-preview — `b1e45b2`, 2026-07-25, **v1.5.0**

**Exactly what the PDF renderer truncated (pre-fix):**

| Field | Normal | Compact |
|---|---|---|
| summary | 52 words | 36 |
| experience entries | `[:3]` | `[:2]` |
| bullets per entry | 3 | 2 |
| words per bullet | 24 | 18 |
| projects | `[:3]` | `[:2]` |
| bullets per project | 2 | 1 |
| words per project bullet | 22 | 16 |
| project description | 10 words | 6 |
| skills | `[:24]` | `[:16]` |
| education | `[:2]` | — |

Plus the header used `identity.name` and **ignored `content.headline` entirely**.

Preview and DOCX rendered full; only the PDF applied these caps. User symptom:
*"missing parts" + "changes not saved"* (`tasks/lessons.md:409-425`).

**The `faithful` flag**, threaded top-down:
```python
bullets  = cleaned if faithful else cleaned[:limit]
word_cap = None if faithful else max_words
selected = items if faithful else items[: 2 if compact else 3]
```
`_limit_words` gained `max_words: int | None` with a `None` short-circuit (`:91-92`).
Download route (`routes/history.py:224-226`):
> Faithful (WYSIWYG) render: the downloaded PDF must match the on-screen preview and the
> DOCX exactly — never silently truncate the user's curated content to force one page.

One-page discipline retained **only** for the auto-generated path.

### 2.6 The editor epic

- **59 commits** in `0d9a55b..934ec3f`; **51** carry the `resume-editor` scope; 6 merges
- **2026-07-22 15:23:35 → 2026-07-27 22:46:05** = **5 days, 7 hours, 23 minutes**
- Shipped across **v1.5.0** (Plans 1-5) and **v1.6.0** (Results-tab editor)

**Tables introduced** — all in `64ba014`, migration `alembic/versions/0014_resume_documents.py`:

| Table | ORM | Location |
|---|---|---|
| `resume_documents` | `ResumeDocument` | `models.py:385-402` |
| `cover_letter_documents` | `CoverLetterDocument` | `models.py:404-420` |
| `resume_document_revisions` | `ResumeDocumentRevision` | `models.py:422-433` |
| `resume_edit_rules` | `ResumeEditRule` | `models.py:435-442` |

`resume_document_revisions` has `UniqueConstraint("document_id","rev")`. `ResumeDocument.rev`
is the optimistic-concurrency counter.

**Concurrency design worth claiming** (`services/resume_document.py`): `apply_write` takes
a `base_rev` and issues `UPDATE … WHERE id = :doc AND rev = :base_rev SET rev = base_rev+1`
(`:151-152`) — compare-and-swap. The pre-write snapshot is inserted first with autoflush
suppressed (`:144-146`) so it doesn't hit the unique constraint before the CAS runs.
`undo` is expressed as a forward write (`:197-198` → `restore_revision` → `apply_write`
with `source="undo"`), so it inherits the same safety.

## Timeline

| Tag | Tagged | Commits since prev | What shipped |
|---|---|---:|---|
| v1.0.0 | 2026-06-22 04:04 | 179 | First tagged release: six-agent pipeline, discovery + Batch API, Celery campaigns, Postgres/Alembic, auth, Docker/ECS |
| v1.1.0 | 2026-06-24 10:11 | 7 | Resume completeness; education/skills authoritative; per-user PDF header; mobile sign-out |
| v1.1.1 | 2026-06-24 11:51 | 3 | ATS-parseable text layer (f-ligature). **Also silently broke all prod PDFs** |
| v1.2.0 | 2026-07-01 15:13 | 14 | Admin discovery search-criteria Phase 1; fixed discovery returning 0 jobs |
| v1.3.0 | 2026-07-04 00:33 | 13 | Semantic Stage-1 matching + backlog rescore (migration 0013) |
| v1.3.1 | 2026-07-06 10:45 | 4 | **Restored prod PDF** (lmodern) + build-time TeX guard + DOCX dedup + visible fallback |
| v1.4.0 | 2026-07-20 17:31 | 17 | Sentry error alerting + resume OCR fallback |
| v1.5.0 | 2026-07-27 13:18 | 51 | Resume editor Plans 1-5, profile dashboard, faithful WYSIWYG PDF |
| v1.6.0 | 2026-07-27 22:48 | 9 | Editable resume in the Results tab |

Untagged since v1.6.0: `cc0c73b`, `0d3e166` (AWS cost docs, 08-02), `2e3428a`, `312050d`,
`f2845a3` (eval gating, 08-04/05).

## ⚠️ Cycle time — **the data does not exist. Do not quote a median.**

Searched all 425 lines of `lessons.md`, all 78 revisions of `HANDOFF.md`, all 302 commit
messages, and the issue tracker.

- **There is no issue tracker.** `gh issue list --state all` returns **zero** issues.
- The 14 PRs are self-merged within seconds (PR #14: created `13:04:16Z`, merged
  `13:04:26Z` — 10 seconds). PR timestamps carry no latency signal.
- **No commit, HANDOFF revision, or lesson entry records a date on which a user reported
  a bug.**

Three *bounded* intervals can be evidenced, none a true report-to-ship measurement:

| Issue | Bound | Evidence quality |
|---|---|---|
| Resume garbled / missing sections | ≈1 day to v1.1.0; ≈1d 2h to the real root cause (v1.1.1) | Report date inferred from the session's HANDOFF header. **Weak.** |
| Prod PDF 503 outage | **Defect latency ≈11 days**; report→ship ≤2 days | Defect latency solidly evidenced. Report date is a lower bound. **Partial.** |
| Download ≠ preview | Same-day diagnose+fix; 2 days commit→tag | Report date = fix date. **Weak.** |

**The one real number: ~11 days defect latency** on the lmodern outage
(`22e44a9` 2026-06-24 → `4c9415f` 2026-07-05), knowable only because the fix commit
named the introducing version. The structural reason the rest is missing is §2's
feedback gap: there is no free-text channel anywhere in the product.

## The feedback capture gap

**Table** (`models.py:313-327`):

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `id` | String | no | PK |
| `analysis_id` | String | **no** | FK→analyses, indexed |
| `agent_name` | String | yes | — |
| `rating` | Integer | no | comment: `# e.g. -1/+1, or 1–5` |
| `note` | Text | yes | — |
| `trace_id` | String | yes | **not indexed, no FK** |
| `created_at` | DateTime(tz) | no | — |

**API** (`schemas.py:561-565`) accepts `analysis_id`, `agent_name`, `rating`, `note`.
⚠️ `rating` is a bare `int` with **no `Field(ge=…, le=…)`** — any integer validates.

**Route** `routes/feedback.py:28-50` populates `trace_id=get_trace_id()`.

**Frontend** — `api/client.ts:136-137` *supports* `agentName` and `note`. The only call
site does not use them, `pages/Results.tsx:194-198`:
```tsx
const sendFeedback = (rating: number) => {
  if (!data) return;
  setFeedbackRating(rating);
  api.submitFeedback(data.id, rating).catch(() => setFeedbackRating(null));
};
```
UI is two icon buttons (`Results.tsx:253-275`). No other call site exists anywhere.

| Column | Written by UI? |
|---|---|
| `analysis_id`, `rating`, `id`, `created_at` | yes (`rating` is only ever `1` or `-1`) |
| **`agent_name`** | **NEVER** — always `null` |
| **`note`** | **NEVER** — no text input anywhere in the product |
| **`trace_id`** | **NEVER meaningfully** — see below |

**`trace_id` is structurally always NULL.** `new_trace_id()` is called at exactly five
sites, all pipeline entry points (`orchestrator.py:328`, `:486`, `discovery.py:218`,
`:339`, `:512`). **There is no ASGI middleware binding a trace id per request** —
`main.py` registers only `SlowAPIMiddleware` (`:79`) and `CORSMiddleware` (`:81`).
`POST /api/feedback` is a separate request in its own asyncio task, so
`trace_id_var.get()` returns the `None` default.

**Nothing reads `feedback.trace_id`** — the only references are the response schema
(`schemas.py:575`) and a docstring (`evals/__init__.py:10`). The "future evals scorer"
was never written. The only working join is `feedback.analysis_id → analyses.id`, which
tells you *which analysis* but not *which agent*.

**Row counts — LOCAL DEV SQLITE, NOT PRODUCTION.** `data/jobfit.db`, mtime
**2026-06-08 11:35**:

| Table | Rows |
|---|---:|
| **feedback** | **0** |
| users | 1 |
| analyses | 99 |
| jobs | 1216 |
| llm_calls | 603 |

⚠️ Caveats, all material: production is **Postgres on AWS**, not this file. This snapshot
predates the SQLite→Postgres migration (`694d86b`, 2026-06-08) — every release from
v1.0.0 on is absent. Its schema is obsolete (has a dropped `github_cache` table, missing
migrations 0004-0014). **`feedback = 0` is NOT evidence that no user gave feedback in
production.** What *is* true regardless of database: `agent_name`, `note`, and `trace_id`
are NULL in every row that could exist.

## ⚠️ One-page constraint — "three times" is **NOT supported. The defensible number is TWO.**

Enumerated every commit touching the rendering surface. Seven resume rendering/truncation
fixes exist; classified by whether the one-page constraint *caused* them:

| # | SHA | Date | Tag | Caused by one-page? |
|---|---|---|---|---|
| 1 | `3bc189a` | 06-10 | v1.0.0 | **No — this is the constraint's ORIGIN** (the fix *added* the rule) |
| 2 | `ef113b7` | 06-23 | v1.1.0 | **YES — breakage #1** |
| 3 | `d88fe46` | 06-24 | v1.1.0 | No — identity/tenancy bug |
| 4 | `22e44a9` | 06-24 | v1.1.1 | No — font/encoding |
| 5 | `98877a3` | 07-04 | v1.3.1 | Partly — surfaced as page overflow, root cause was misrendered metadata |
| 6 | `4c9415f` | 07-05 | v1.3.1 | No — packaging bug caused by #4's fix |
| 7 | `b1e45b2` | 07-25 | v1.5.0 | **YES — breakage #2** |

- **Strictly** (constraint *caused* breakage): **2**
- **Generously** (constraint is the *subject*, counting its origin): **3** — probably what
  "three times" meant, but `3bc189a` is the constraint being introduced, not causing harm
- **Loosely** (symptom was "2 pages / missing parts"): **4** — and three of those four had
  root causes unrelated to page count

**The most interesting finding is that last row: the one-page framing repeatedly
mis-attributed unrelated bugs.** That is a better story than "it broke three times."

## MAX_TOKENS — global cap split per-agent

Single commit **`ef113b7`**, 2026-06-23, **v1.1.0** (same commit as 2.1 and 2.2 — it was
root cause #3 in that investigation).

**Before:** one module constant `MAX_TOKENS = 4096` in `base.py`, passed to every
`client.messages.create()`. All eight agents shared it despite output sizes differing ~10×.

**After** (`base.py:86-88`):
```python
class BaseAgent:
    model: str = SONNET
    # Output-token cap per call. Small structured agents are fine on the default;
    # whole-document emitters (resume JSON, full .tex) override this…
    max_output_tokens: int = MAX_TOKENS
```
with `base.py:165` changed to `max_tokens=self.max_output_tokens`.

| Location | Value |
|---|---:|
| `base.py:88` default | 4096 |
| `resume_tailorer.py:12` | **8192** |
| `resume_latex.py:50` | **8192** |
| `resume_editor.py:14` | **8192** (added later, `6932260`) |

⚠️ **The token cap was a latent risk, not the cause of the reported truncation.**
`lessons.md:341-345` is explicit. Do not mis-attribute. Regression test:
`tests/test_agents/test_max_output_tokens.py`.

---

# SECTION 3 — Q2 evidence: agentic development

## Workflow scaffolding

| Artifact | Path | Size | Contents |
|---|---|---|---|
| `CLAUDE.md` | repo root | 238 lines | Overview, stack, ~90-line annotated architecture map, Key Conventions, and a section literally headed **"Undocumented Conventions"** (`:129+`) holding six hard-won invariants (SSE ordering, `_inject()` vs `.replace()`, `with_tracking()` in-place mutation, runtime model override, JD cache semantics, `validation_warnings` as `exclude=True`). Then Workflow Orchestration, Task Management, Session Lifecycle, Definition of Done. |
| `HANDOFF.md` | repo root | 121 lines | Six sections matching `HANDOFF.template.md`. Update triggers live in `CLAUDE.md:190`: *"task complete · wave/milestone done · user signals stop · unresolved blocker · context window ~70% · 30+ min without an update"* |
| `tasks/todo.md` | | 102 lines | 45 `[x]`, 13 `[ ]`. Last modified 2026-07-04 |
| `tasks/lessons.md` | | 425 lines | 31 dated entries, 2026-05-26 → 2026-07-25. Consistent schema: **Pattern / Fix / Avoid / See:** with `path:LINE` refs |
| `tasks/agent_memory.md` | | 125 lines | Architecture Decisions / Known Gotchas / Solved Problems / Useful Patterns. ⚠️ **Untouched since 2026-06-12** |

**Plans and specs:** `docs/plans/` and `docs/specs/` do **not** exist. Everything is under
`docs/superpowers/` — **23 plan documents**, **10 design specs**.

### Stop hook

`.claude/settings.json` contains exactly one hook and nothing else. **No PreToolUse,
PostToolUse, SessionStart, or UserPromptSubmit hooks.** `.claude/hooks/stop.sh` blocks
when **`git status --porcelain` is non-empty AND `HANDOFF.md` mtime is older than 30
minutes**, via `exit 2` with stderr fed back to the agent.

Three escape valves: `stop_hook_active: true` (loop guard), non-repo/clean tree, fresh
HANDOFF. Both design choices trace directly to lessons — `lessons.md:267` *"Stop hooks
need the stop_hook_active guard or they loop"* (2026-05-31) and `lessons.md:279` *"A
dirty-tree Stop hook must tell the user to COMMIT, not just edit the gated file"*
(2026-06-01). **The hook is a codified bug post-mortem.**

`.claude/skills/` — **8 skill files** (5 in `<name>/SKILL.md` form): `fde-plan`,
`fde-review`, `debug-playbook`, `api-conventions`, `pr-checklist`, `conventions.md`,
`git-leak-cleanup.md`, `pre-push-checklist.md`. Plus `.claude/commands/promote-lesson.md`
(mines `lessons.md` for patterns worth codifying), `.claude/brief_template.md`, and
`RESOLVER.md` (keyword→skill routing, capped at "at most 2 files per response").

## Commands

```make
test:  pytest tests/ -v --cov=backend --cov-report=term-missing --cov-fail-under=70
fmt:   ruff format backend/ tests/
lint:  ruff check backend/ tests/ ; mypy backend/ ; python scripts/check_schema_drift.py
check: make fmt && make lint && make test
```

`make check` runs fmt → lint (three commands) → test, chained with `&&`.
Note `fmt` **rewrites** files; CI uses `ruff format --check` instead.

## Test infrastructure

**Testcontainers confirmed** — `tests/conftest.py:10`, real **postgres:16** over asyncpg,
`scope="session"` (`:32`). Docker is a hard prerequisite.

**Isolation: outer-transaction rollback with savepoint-joined sessions.** Not a fresh
container, not schema recreate, not truncation. `conftest.py:51-53`:
> Open one outer transaction per test and roll it back at the end. Sessions bound to this
> connection join it via SAVEPOINTs, so every commit a test (or the code under test) makes
> is undone on teardown.

Mechanism: `join_transaction_mode="create_savepoint"` (`:88`) + `await outer.rollback()` (`:77`).
**FK enforcement is ON by default** (prod parity); `@pytest.mark.relaxed_fks` issues
`SET session_replication_role = replica` (`:60-61`).

**Counts — both numbers are real and mean different things:**
- **648** `def test_` definitions across 103 test files
- **713** tests collected/passed (parametrisation expands them), 1 deselected

**Coverage: 83%** — I measured this directly today (`5616` statements, `961` missed).
Consistent with the 82.88% recorded at `HANDOFF.md:115`.

⚠️ **CI does not enforce the coverage floor.** `.github/workflows/ci.yml:42` comments that
`--cov-fail-under=70` comes from pyproject addopts; `pyproject.toml:8` shows addopts is
**only** `-m 'not integration'`. The 70% gate exists solely in `Makefile:7`, i.e. locally.

## ⚠️ Model configuration — CORRECTNESS CRITICAL

Constants `base.py:29-31`: `HAIKU = "claude-haiku-4-5-20251001"`,
`SONNET = "claude-sonnet-4-6"`, `MAX_TOKENS = 4096`. Class default `base.py:84`: `SONNET`.
`config.py:53-54`: `resume_model = "claude-opus-4-8"`, fallback `"claude-sonnet-4-6"` —
**neither overridden in `.env`, so the Opus default is live** (verified directly).

**Interactive analysis pipeline (`/api/analyse`):**

| Phase | Agent | Model |
|---|---|---|
| 1 | job_parser | **Haiku 4.5** (`job_parser.py:13`) |
| 1 | match_scorer | **Haiku 4.5** (`match_scorer.py:9`) |
| 1 | gap_analyst | **Sonnet 4.6** (inherits `base.py:84`) |
| 2 | resource_planner | **Sonnet 4.6** (inherits) |
| 2 | cover_letter | **Sonnet 4.6** (inherits) |
| 2 | resume_tailorer | **Sonnet 4.6** (inherits) |

**Off-pipeline:** cold_email = Sonnet · profile_extractor = **Haiku** (`profile_extractor.py:8`)
· `_ResumeLatexAgent` = Sonnet (`resume_latex.py:47`) · **resume_editor = Opus 4.8**
(`resume_editor.py:18`), one-shot Sonnet fallback on service failure only (`resume_chat.py:64-70`).

**Bulk discovery:** all three Phase-1 agents forced to Haiku; Stage-2 relevance is Haiku
(`stage2.py:9,44`; `batch_processor.py:40`).

### The runtime mutation — CLAUDE.md is imprecise

`CLAUDE.md:137` says the orchestrator mutates `agent.model = HAIKU`. There is exactly one
`agent.model =` assignment (`orchestrator.py:254`) and it assigns a **function parameter**,
not a hardcoded HAIKU. It lives in `_run_phase1`, which is the **discovery-only** entry
point — every production caller passes `model=HAIKU` (`discovery.py:298`, `:919`, `:1006`).
The interactive SSE path (`run_steps`, `orchestrator.py:472`) contains **no** model mutation.

Net effect: job_parser and match_scorer are already Haiku (no-op); **only `gap_analyst`
actually changes tier**, downgraded Sonnet→Haiku for bulk discovery.

### ⚠️ Cost bug — Opus is priced as Sonnet

`cost_calculator.py:3-18` defines `COST_PER_MILLION` for **only** Haiku and Sonnet.
`claude-opus-4-8` is absent, so `rates = COST_PER_MILLION.get(model, _FALLBACK)` (`:34`)
prices every Opus call at Sonnet rates. **Resume-editor spend is understated** in
`LLMCall`, `/api/metrics/costs/summary`, and the per-user spend caps. Do not quote
per-run cost figures as precise.

## Evidence of agentic authorship

**Quantified, and verifiable:**

| Metric | Value |
|---|---|
| Total commits on `main` | **302** |
| Carrying a `Co-Authored-By` trailer | **201 (66.6%)** |
| Carrying "Generated with Claude Code" | **0** |

Trailer breakdown: Opus 4.8 ×141, Sonnet 4.6 ×31, Fable 5 ×20, Opus 5 ×5, Sonnet 5 ×1,
Opus 4.7 ×1, plus 2 with inline prose. **Every trailer names a Claude model; there are no
human co-authors.** Six model generations, consistent with 2026-05-23 → 2026-08-05.

Two trailers are self-documenting about the workflow:
> *"Implemented by controller after the task subagent hit its session limit (tests were
> already written)."*
> *"Final review flagged the builder ignored the remote flag; add Remote to
> allowed_locations when set."*

**Plan-precedes-implementation, at commit-timestamp granularity:**

| Artifact | Commit | Timestamp |
|---|---|---|
| Design spec `2026-07-22-resume-editor-design.md` | `3661489` | 07-22 **15:23** |
| Plan `…-02-chat-agent.md` | `1174eb2` | 07-22 **22:16** |
| `backend/agents/resume_editor.py` created | `6932260` | 07-22 **22:36** |
| `backend/services/resume_chat.py` created | `12b3d31` | 07-22 **22:51** |

Second instance: plan `4da7ec4` 07-27 **20:23** → implementation `24a7cc3` 07-27 **20:53**.

`.superpowers/sdd/` holds machine-generated task briefs, task reports, and
`review-<sha>..<sha>.diff` artifacts — byproducts of a subagent execution loop.

### ⚠️ What is NOT demonstrable — do not overstate

- Trailers are **self-asserted text**. They show commits were authored *through* Claude
  Code with trailer conventions on. They do not distinguish "agent wrote it" from "agent
  assisted".
- **101 commits (33%) carry no trailer at all** — the earliest work (05-23 → 05-28) is
  largely untrailered, so the practice was adopted partway in.
- No signed commits, no CI attestation, no provenance signature.
- **No artifact proves any specific line was model-generated.** Author on every commit is
  `divyanshu144`.

**Defensible sentence:** *"201 of 302 commits (67%) carry a Claude co-author trailer
spanning six model generations; 23 implementation plans and 10 design specs are committed;
and for the resume editor the spec, plan and implementation commits are timestamped in
that order within hours of each other."*

## Prompt caching — built, measured, removed

**Removal commits:**
- **`598076d`** 2026-06-03 01:32 — *"Each pipeline agent's system prompt is unique per
  request … so cache_control wrote **21K+ cache-creation tokens with 0 reads** — pure
  1.25x write premium, no savings."*
- **`c0c7236`** 2026-06-03 07:44 — *"stage2's system is ~396 tok, ~10x below Haiku's 4096
  cache minimum, so its cache_control never created cache (dead code that looked
  load-bearing). **No cache_control remains anywhere in backend/.**"*

**`docs/DEVLOG.md:7`, the numbers to quote:**
- 21K+ creation tokens, **0 reads**
- stage2 system **~396 tokens** vs Haiku's **4,096** minimum (~10× short)
- Padding maths: a cached read of a 4,096-token prefix costs `0.1 × 4096 = 409.6`
  effective tokens — **more than today's full-price ~380** — so **break-even is negative
  at any call count**
- Runs fire **~85–110 stage2 calls** (max 107, run `2de843fa`); full profile ~2,545 tokens

⚠️ **The honest arc includes a self-correction.** `lessons.md:101` (2026-05-28) states the
Haiku threshold as **≥2048** and concludes the merged_profile agents *"already exceed 2916
tokens and cache correctly."* Six days later the DevLog states **4,096** and reports **0
reads** — the earlier estimate was wrong and was corrected by live measurement. The lesson
file was never revised. **Tell the correction, not the stale estimate.**

Residue: the cost model, DB columns and metrics endpoint built for caching survive
(`cost_calculator.py:4-15`, `routes/metrics.py:18-19,80-81`) and now always evaluate
against zero-valued columns. `README.md:32` still advertises "cache metrics".

## OCR system binaries

`Dockerfile:33-40`, stage **`backend-tex`** (`:28`):
```
    33	RUN apt-get update && apt-get install -y --no-install-recommends \
    34	    texlive-latex-recommended \
    35	    texlive-latex-extra \
    36	    texlive-fonts-recommended \
    37	    lmodern \
    38	    poppler-utils \
    39	    tesseract-ocr \
```

| Service | Base stage | Gets poppler/tesseract? |
|---|---|---|
| `api` | `backend-tex` | **YES** |
| `worker` | `backend-tex` (`Dockerfile:57`) | **YES** |
| `beat` | `backend-base` | **NO** |
| `frontend` | separate Dockerfile | **NO** |

⚠️ **The lesson text is wrong here.** `lessons.md` (2026-07-18) says the binaries went to
backend-tex *"api-only, not worker/beat"*. `Dockerfile:57` derives `worker` from
`backend-tex` too. **Don't repeat "api-only."**

**`lessons.md:374` (2026-07-18)** — the mocked-tests lesson:
> pdf2image needs the `pdftoppm` binary … and pytesseract needs `tesseract` … neither is
> importable at the Python level, so `pip install` and unit tests (which mock
> `convert_from_bytes` / `pytesseract.image_to_string`) **pass with zero signal on whether
> the binaries actually exist in the built image.** … **Flagged explicitly to the user as
> unverified rather than reported as done.**

**`86f0ab6` (2026-07-20)** closed it — docs-only, 12 insertions to `HANDOFF.md`:
> `docker build --target api` succeeds, and inside the image **`tesseract 5.5.0` /
> `pdftoppm 25.03.0`** resolve … The OCR-fallback change is fully verified.

**The earlier incident it references — `lessons.md:350` (2026-07-05):**
> *"verified by reading the package list" is not verification; silent fallbacks mask outages*
>
> v1.1.1's ligature fix added `\usepackage{lmodern}`. I "verified" deployability by reading
> the Dockerfile apt list and asserting "lmodern ∈ texlive-fonts-recommended" — which is
> false on Debian. Result: **EVERY prod PDF download 503'd for ~10 days.** Nobody noticed
> because the frontend silently fell back to the DOCX… **The claim even made it into a
> commit message as fact.**
>
> Fixes: lmodern added; a **build-time guard now renders + compiles the REAL template
> inside the image**; the frontend fallback now shows a visible notice.

**This arc is the strongest process evidence in the repo:** unverified claim → 10-day
silent outage → fix the *class* of bug with a build-time guard → same risk class recurs
2 weeks later → **recognised by name** and shipped flagged-unverified → verified with
named binary versions 2 days later → loop closed in HANDOFF.

---

# SECTION 4 — Claim vs reality audit

## 1. Are the agents autonomous? — **NO**

`orchestrator.py:52-56`:
```python
PHASE1: list[str] = ["job_parser", "match_scorer", "gap_analyst"]
PHASE2: list[str] = ["resource_planner", "cover_letter", "resume_tailorer"]
_PARALLEL_PHASE2 = ("cover_letter", "resume_tailorer")
```
Control flow is a `for` loop (`:396`, `:518`) plus one `asyncio.gather` (`:617`). Step
selection is set intersection against the same constant (`:512`), never model-chosen.
Grep for `tools=`, `tool_use`, `tool_choice`, `while True`, `max_iterations`, `agent_loop`,
`next_step` across `backend/agents/` and `orchestrator.py` → **zero matches**.

**⚠️ Also: `CLAUDE.md` overstates the concurrency.** It says Phase 2 runs
*"resource_planner + cover_letter + resume_tailorer concurrently."* Only two are parallel
— `resource_planner` falls into `sequential` at `:516` and runs in the loop at `:518`.
**Two concurrent, not three.**

The one genuine exception: `ResourcePlannerAgent` self-scores and regenerates low-confidence
gaps (`resource_planner.py:29-63`). Model-conditioned branching, but the branch structure
is hardcoded.

**VERDICT: overstated if spoken as "autonomous". `CLAUDE.md` never says "autonomous" —
the risk is what you say aloud.**

> **Safe phrasing:** "It's a six-stage LLM pipeline. Each stage has its own prompt template
> and its own Pydantic output schema, and the orchestrator runs them in a fixed dependency
> order — Phase 1 sequential because each stage consumes the previous one's structured
> output, then two of the Phase-2 generators concurrently. It's deliberately not a
> tool-calling agent loop; deterministic order is what lets me cache, retry and
> cost-attribute each step independently."

## 2. Circuit breaker — **IT IS A LOG LINE**

`base.py:40-65`, entire implementation:
```python
# Circuit-breaker signal (V1: the CRITICAL log IS the alert, no open/close state).
_BREAKER_THRESHOLD = 5
_consecutive_failures = 0

def _record_failure(agent_name: str) -> None:
    global _consecutive_failures
    _consecutive_failures += 1
    if _consecutive_failures == _BREAKER_THRESHOLD:  # exactly once per streak
        logger.critical(...)

def _record_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0
```

- No state machine — no closed/open/half-open
- **It never prevents a call.** `_consecutive_failures` appears only at lines 42, 54, 65 —
  written and reset, **never read before dispatch** (verified directly). At 500 consecutive
  failures the 501st still fires.
- Per-process module global — API, each Celery child, each worker have independent counters
- Resets on any single success, so a 50% failure rate never trips it
- Fires exactly once per streak (`==`, not `>=`)

**CONTRADICTION:** `CLAUDE.md:84` says "retry/**breaker**". The code's own comment at
`base.py:40` says *"no open/close state"*. **The code wins.**

> **Safe phrasing:** "There's a real retry layer — transient-only, 529s and timeouts,
> exponential backoff with jitter, plus a one-shot self-correction retry when the model
> returns JSON that fails schema validation. What I labelled a circuit breaker is honestly
> just a consecutive-failure counter that emits a CRITICAL log — no open/half-open state,
> and it doesn't block calls. A real distributed breaker in Redis is the obvious next step."

## 3. Celery — beat is real; **discovery is not on it**

**Beat: genuinely scheduled.** `celery_app.py:57-63`, `crontab(hour=2, minute=0)` →
`campaign.dispatch_nightly` → one enqueue per eligible user (`campaign_run.py:174-188`).
Surrounding config is thoughtful: `task_acks_late`, `task_reject_on_worker_lost`,
`worker_prefetch_multiplier=1`, time limits (`celery_app.py:50-54`), a DB partial-unique
index preventing two concurrent runs per user (`campaign_run.py:117-147`), stale-run
healing (`:101-114`).

**Discovery: in-process asyncio, NOT Celery.** `services/discovery.py:435-448`:
```python
task = asyncio.create_task(_run_discovery_task(run.id, source, user_id))
_background_tasks.add(task)
```
Zero `.delay(` / `apply_async` in `discovery.py`. Same for `run_all_discovery` (`:651`),
`run_batch_discovery` (`:672`), `run_rescore` (`:1083`). The **admin** campaign is also
in-process (`routes/campaign.py:39-60`), with a comment at `:23-25` saying *"**No scheduler
yet** — a single supervised run at a time… best-effort, reset on server restart."*

**Consequence: an API restart or ECS task replacement silently kills any in-flight
discovery run.** No retry, no ack, orphaned `DiscoveryRun` row.

> **Safe phrasing:** "Celery beat runs a real nightly cron at 02:00 UTC that fans out one
> campaign run per user, with acks-late and a DB-level uniqueness guard. Discovery is
> different and I'd call it out — it's an in-process asyncio task in the API, so it doesn't
> survive a restart. Moving it onto the same queue is the obvious fix."

## 4. Resource planner — **pure model recall, no fetching**

No `httpx`, `requests`, `aiohttp`, search API, or retrieval anywhere in
`backend/agents/`. Every path is `self._call(...)` → Anthropic (`resource_planner.py:23`,
`:34`, `:48`). The prompt has no URLs in its schema and no verification step.

The self-check asks the model to score *its own* output for **specificity**
(`resource_planner.py:75-84`) — whether a title *sounds* concrete, not whether it exists.
The regeneration prompt then pushes harder (`:96-97`): *"Regenerate ONLY these skills with
highly specific, named resources (exact course titles + platform, real book titles…)."*
**Pressuring for exact titles without retrieval increases confident fabrication.**

> **Safe phrasing:** "The resource planner is pure model knowledge — no web search, no RAG,
> no link checking. It has a two-pass design that scores its own output for specificity and
> regenerates the vague ones, but that's a specificity check, not a truth check. Course and
> book titles are leads to verify, not citations."

## 5. Semantic search / pgvector — **⚠️ OFF BY DEFAULT**

The embedding call is real (`memory.py:94-109`, OpenAI `embeddings.create`) and pgvector
is a real `<=>` cosine query with a native index path (`:283-293`).

**Scope: one profile's content only** — profile-review text, `profile_yaml`, CV text
(`memory.py:141-176`). Every retrieval filtered `WHERE profile_id = :profile_id` (`:241`,
`:288`, `:318`). Not the codebase, not the job corpus.

**Only 3 of 6 pipeline stages use it** (`orchestrator.py:152-183`): gap_analyst,
resource_planner, cover_letter. resume_tailorer uses a dedicated context builder; job_parser
and match_scorer use the static compact profile.

**Fallback is three-deep**, not two: pgvector → dense cosine in Python over
`embedding_json` (`:311-342`) → SHA256-hashed sparse bag-of-words into 512 buckets (`:64-72`).

**⚠️ THE PROBLEM:** `config.py:15` → `embedding_provider: str = "disabled"`. I verified
directly that the local `.env` contains **zero** `EMBEDDING_PROVIDER` lines, so it inherits
`"disabled"` and `embed_texts` returns `None` before any API call. **As configured on this
machine, semantic memory is OFF and the hashed-keyword fallback is running.**
AWS task definitions do set `EMBEDDING_PROVIDER=openai`, so it is live in the deployed env.

Same issue hits "semantic Stage-1" discovery: `semantic_stage1` returns `None` without
embeddings (`discovery.py:119-120`), falling through to a substring scan (`:151-155`):
`return any(role in text_lower for role in all_roles)`.

> **⚠️ VERIFY `EMBEDDING_PROVIDER=openai` IS SET BEFORE RECORDING, or you will be narrating
> a code path that is not executing.**

## 6. AWS deployment — **staging, and the docs already say so**

`infra/aws/README.md:248-257` lists the gap itself. Verified absences:

| Item | Status | Evidence |
|---|---|---|
| HTTPS / ACM | ❌ | `08_create_alb.sh:136`: *"WARNING: this creates HTTP only."* |
| WAF | ❌ | TODO at `infra/aws/README.md:254` |
| Autoscaling | ❌ | `10_create_ecs_services.sh` — only `--desired-count 1`. Single fixed task per service |
| Multi-AZ RDS | ❌ | `04_create_rds.sh` — no `--multi-az` |
| RDS backups | ⚠️ Partial | `--backup-retention-period 7` **is** set (`:61`); no restore drill, no deletion protection |
| Secrets rotation | ❌ | SSM used, nothing rotates |
| Alarms | ❌ | CloudWatch Logs only |
| S3 persistence | ❌ | uploaded resumes don't survive task replacement |

**Genuinely good, claim it:** OIDC federation with no long-lived AWS keys; tag-gated deploys
with a concurrency guard; **ECS deployment circuit breaker with auto-rollback on all four
services** (this *is* a real circuit breaker, unlike `base.py`); beat pinned to a strict
singleton to avoid double-scheduling.

> **Never say "it's in production" or "it's live on HTTPS."**

## 7. Gmail — **don't say a flat "drafts-only"**

**Two paths.** The automated campaign is drafts-only (`campaign_orchestrator.py:147-151`,
`:286` — never touches `send_message`). But a **manual admin send endpoint exists and does
send**: `routes/contacts.py:118-158` → `gmail_service.py:47-52`
`client.users().messages().send(...)`. It requires an explicit admin POST per contact and
refuses without a pre-existing draft (`contacts.py:139-140`).

The Celery multi-tenant campaign contains **zero** gmail/contact/cold-email references;
its docstring (`campaign_run.py:6-8`) states: *"**No contact discovery / cold email /
Gmail** — that stays in the separate admin campaign."*

**Credentials: one server-side refresh token** (`gmail_service.py:23-31`), no per-user
OAuth, no token column on `User`. Every draft and send comes from the **operator's own
mailbox** — which is exactly why both routes are `require_admin`-gated server-side.

> **Safe phrasing:** "Gmail is server-side with a single OAuth refresh token — one mailbox,
> mine, not per-user OAuth. That's why both endpoints are admin-gated server-side. The
> automated campaign path is deliberately drafts-only. There is a separate manual send
> endpoint, but it's admin-only and one explicit POST per contact."

## 8. Registration / invites — invite-only, but binding is optional

`auth.py:47-53` — invite required **except for the first user**; `auth.py:77` →
`is_admin=is_first_user`. **On an empty database, whoever POSTs `/api/auth/register` first
becomes admin.** Verified directly.

`schemas.py:470` → `email: str | None = None  # if set, only this email can use the invite`.
Enforcement `auth.py:65` short-circuits: `if invite.email and invite.email.lower() != …`.
**An invite created without an email is a bearer token — anyone holding the string can
register any address.**

Real mitigations: `secrets.token_urlsafe(32)` (`:212`), 72-hour expiry, single-use via
`used_at`, unique+indexed, admin-gated creation, register rate-limited 10/min per IP.
⚠️ The used-invite check is SELECT-then-UPDATE with no row lock (`:54-84`) — two concurrent
registrations on one token could both pass.

## 9. Other things more modest than they sound

- **Discovery is admin-only.** `require_admin` on **12 of 13** endpoints in `routes/discovery.py`,
  including `/feed`, `/saved`, and `PATCH /jobs/{id}/save`. Only `/shortlist` is open.
  `README.md:27-28` lists these under Core Features without noting it. It looks user-facing
  in a single-user demo because the operator is the admin.
- **Cost caps are enforced but only on the campaign path.** `check_user_caps` callers exist
  **only** in `campaign_run.py` and `campaign_user.py`. `routes/analyse.py` has **zero** —
  interactive analysis is rate-limited, not spend-capped. Default cap `$10.00/month`
  (`models.py:129`). Checks are per-run not per-call, so one run can overshoot.
- **Batch API is real** — `batch_processor.py:47`, with polling and per-item error isolation.
  The 50% discount claim is legitimate. Honest note at `:33`: *"cache_control is
  intentionally omitted — Batch API does not support prompt caching."*
- **Admin gating is server-side** — `require_admin` FastAPI dependency raising 403
  (`auth_service.py:53-56`), applied across discovery (12), contacts (5), campaign (5),
  metrics (3). Frontend checks are convenience on top. **Claim this confidently.**
- **Multi-tenancy is genuinely solid where it matters:** cache key is content-addressed and
  owner-scoped (`orchestrator.py:94-128`), campaigns hard-block on `get_owned_profile`
  returning `None` (`campaign_run.py:204-207`), analyses ownership-checked on every read.
- **Two stale docstrings contradict their own code:** `celery_app.py:8* claims the beat
  schedule is "a single no-op heartbeat" (it's the real nightly dispatch);
  `campaign_orchestrator.py:226` says *"Downstream steps are stubbed"* (lines 276-286 run a
  real tailor → contact-find → cold-email → draft chain). Here the code is *better* than
  its docstring.
- **Two self-labelled dead branches** at `orchestrator.py:543-544` and `:578-579`:
  *"Currently unreachable: resume_tailorer always dispatches via _PARALLEL_PHASE2 (kept for
  symmetry; untested)."* Honest in-code labelling, not a problem.

---

# SECTION 5 — Numbers you can quote

**All repo-sourced and safe to quote unless marked.**

| Metric | Value | Source |
|---|---|---|
| Backend Python | **12,046 lines** | `find backend -name "*.py"` |
| Test code | **14,718 lines** | `find tests -name "*.py"` |
| Frontend TS/TSX | **6,752 lines** | `find frontend/src` |
| Test functions | **648** `def test_` | across 103 files |
| Tests collected/passing | **713** (1 deselected) | `pytest` — measured today |
| Coverage | **83%** (5,616 stmts, 961 missed) | measured today; gate is 70% (`Makefile:7`) |
| Commits | **302** | 2026-05-23 → 2026-08-05 |
| Commits with Claude co-author trailer | **201 (66.6%)** | 6 model generations |
| Releases | **9 tags**, v1.0.0 → v1.6.0 | 2026-06-22 → 2026-07-27 |
| Agent classes | **9** in `backend/agents/` (+1 in services) | 6 in the pipeline |
| Prompt templates | **9** (313 lines total) | `backend/prompts/` |
| DB models | **23** | `backend/models.py` |
| Alembic migrations | **14** | `alembic/versions/` |
| Services | **37** | `backend/services/` |
| Route modules | **11** | `backend/routes/` |
| Discovery source clients | **7** | `*_client.py` |
| Frontend pages / components | **15 / 12** | `frontend/src/` |
| Eval cases | **5** | `jobfit_eval_cases.json` |
| Eval module | **549 lines** (526 at session start) | `backend/evals/` |
| Eval tests | **923 lines**, 8 files | `tests/test_evals/` |
| Implementation plans | **23** | `docs/superpowers/plans/` |
| Design specs | **10** | `docs/superpowers/specs/` |
| Lessons entries | **31** over 425 lines | `tasks/lessons.md` |
| Skills | **8** | `.claude/skills/` |
| Editor epic | **59 commits / 5d 7h** | `0d9a55b..934ec3f` |
| lmodern outage | **~11 days** | 2026-06-24 → 2026-07-05 |
| Prompt cache measurement | **21K+ writes, 0 reads** | `598076d`, DEVLOG |

**⚠️ Needs an environment caveat — do NOT quote as product metrics:**

| Metric | Value | Caveat |
|---|---|---|
| analyses | 99 | **Local dev SQLite, mtime 2026-06-08**, pre-dates the Postgres migration |
| jobs scored | 1,216 | same file; 1,089 filtered, 63 scored, **0 saved** |
| llm_calls | 603 | same file |
| feedback rows | 0 | same file — **not evidence about production** |

**Cannot be quoted at all:**
- **Cycle-time median** — no issue tracker, no report timestamps recorded anywhere. See §2.
- **Production feedback volume** — prod Postgres is not accessible; the RDS instance was
  deleted 2026-08-02 and survives only as snapshot `jobfit-postgres-final-2026-08-02`.
- **User count** — UNVERIFIED. No accessible production data.

---

# Quick reference: what to avoid saying

| Don't say | Say instead |
|---|---|
| "autonomous agents" | "a six-stage pipeline with typed contracts at each boundary" |
| "circuit breaker" (about `base.py`) | "a consecutive-failure counter that emits a CRITICAL log" |
| "in production" / "live on HTTPS" | "a staging deployment on ECS Fargate" |
| "three Phase-2 agents run concurrently" | "two of the three Phase-2 generators run concurrently" |
| "semantic memory over the codebase" | "semantic memory over the candidate's own profile" — **and check the env var** |
| "drafts-only" (flat) | "the automated path is drafts-only" |
| "no Opus anywhere" | "Opus 4.8 for the interactive resume editor" |
| "the eval suite has been green" | "the suite couldn't fail until I fixed the severity gating this week" |
| "invites are bound to a person" | "invites are single-use and expiring; email binding is optional" |
| "the one-page bug hit three times" | "it hit twice, and mis-attributed three other bugs" |
| any per-run cost figure as precise | Opus is priced as Sonnet — costs are understated |

**Recording hygiene:** `.env` holds live `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`HUNTER_API_KEY`, `REED_API_KEY`, `ADZUNA_APP_*`, `GMAIL_CLIENT_SECRET`,
`GMAIL_REFRESH_TOKEN`, `JWT_SECRET`. It is untracked and `git log --all -- .env` is empty,
so nothing leaked. **Do not open `.env`, `docker compose config`, or `/metrics` on camera.**
