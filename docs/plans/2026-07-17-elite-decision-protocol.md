# Elite Decision Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a three-stage, minimum-three-member Elite Decision Protocol that produces a stronger evidence-audited decision through conclave's existing auditable verdict.

**Architecture:** Add `elite` as a first-class mode implemented by `run_elite`: independent fan-out, concurrent council-wide evidence critiques, concurrent revisions, then existing synthesis and verdict application. Store phase artifacts in a backward-compatible `EliteResult`, and extend manifest receipts with phase provenance so the full protocol is auditable without exposing provider identities or secrets.

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2, Typer, pytest/pytest-asyncio, Ruff.

---

### Task 1: Define the result contract

**Files:**
- Modify: `src/conclave/models.py`
- Modify: `src/conclave/__init__.py`
- Test: `tests/test_elite_mode.py`

**Step 1: Write failing tests**

Test that `EliteResult` defaults to protocol version `elite_v1`, requires three responders,
starts incomplete, serializes phase artifacts, and is publicly importable. Test that existing
`CouncilResult` construction still works with `elite is None`.

**Step 2: Verify red**

Run: `pytest tests/test_elite_mode.py -q`
Expected: collection/import failure because `EliteResult` does not exist.

**Step 3: Implement minimally**

Add `ELITE_PROTOCOL_VERSION = "elite_v1"` and `ELITE_MIN_RESPONDERS = 3`. Add a Pydantic
`EliteResult` containing `protocol_version`, `required_responders`, `completed`,
`failure_reason`, `initial_answers`, `critiques`, and `revisions`. Add
`elite: EliteResult | None = None` to `CouncilResult` and export the public symbols.

**Step 4: Verify green**

Run: `pytest tests/test_elite_mode.py -q`
Expected: result-contract tests pass.

**Step 5: Commit**

`git add src/conclave/models.py src/conclave/__init__.py tests/test_elite_mode.py && git commit -m "feat(elite): define protocol result contract"`

### Task 2: Add evidence-audit and revision prompts

**Files:**
- Modify: `src/conclave/prompts.py`
- Test: `tests/test_elite_mode.py`

**Step 1: Write failing tests**

Test prompt builders with three model answers. Assert stable Model A/B/C aliases, stable
answer IDs, no provider/model names, explicit supported/conflicting/externally-unverified
categories, no-invented-citations instruction, original answer inclusion in revision, and
deterministic output for the same input.

**Step 2: Verify red**

Run: `pytest tests/test_elite_mode.py -q`
Expected: failures because elite prompt builders are missing.

**Step 3: Implement minimally**

Add `ELITE_CRITIC_SYSTEM`, `ELITE_REVISION_SYSTEM`, and narrow builder functions that accept
the original prompt and `ModelAnswer` sequences. Reuse the established anonymization style;
represent panel entries using aliases plus answer IDs, never provider names. Keep the elite
protocol version separate from the synthesis prompt version.

**Step 4: Verify green**

Run: `pytest tests/test_elite_mode.py -q`
Expected: prompt tests pass.

**Step 5: Commit**

`git add src/conclave/prompts.py tests/test_elite_mode.py && git commit -m "feat(elite): add evidence audit and revision prompts"`

### Task 3: Implement the three-stage mode

**Files:**
- Modify: `src/conclave/modes.py`
- Test: `tests/test_elite_mode.py`

**Step 1: Write failing tests**

Add async tests using the repository's call seams for: independent initial fan-out; critique
calls only after three initials; revision calls only after three critiques; successful 4-to-3
partial failure; initial, critique, and revision gate failures; correct stage artifacts; and
no calls after a failed gate.

**Step 2: Verify red**

Run: `pytest tests/test_elite_mode.py -q`
Expected: failures because `run_elite` is missing.

**Step 3: Implement minimally**

Implement `run_elite(council, prompt)` using the existing concurrent `fan_out` and provider
call seams. Centralize the strict success-count gate. Preserve provider-side errors in phase
artifacts, stop immediately below three successes, never raise, and set `result.answers` to
successful revisions only after the revision gate passes.

**Step 4: Verify green**

Run: `pytest tests/test_elite_mode.py -q`
Expected: mode tests pass.

**Step 5: Commit**

`git add src/conclave/modes.py tests/test_elite_mode.py && git commit -m "feat(elite): implement evidence audited decision flow"`

### Task 4: Wire library synthesis and verdict behavior

**Files:**
- Modify: `src/conclave/council.py`
- Test: `tests/test_elite_mode.py`
- Test: `tests/test_cache.py`

**Step 1: Write failing tests**

Test `elite` and `elite_sync`, final synthesis receiving revised answers, existing
`_apply_verdict` producing the canonical verdict only on completed runs, incomplete runs
having no synthesis/verdict, and cache keys/results remaining isolated from other modes.

**Step 2: Verify red**

Run: `pytest tests/test_elite_mode.py tests/test_cache.py -q`
Expected: failures because Council does not expose or dispatch elite.

**Step 3: Implement minimally**

Add async/sync public wrappers routed through `_cached_run`. On a completed `run_elite`
result, invoke existing `_synthesize` followed by `_apply_verdict`; skip both when incomplete.
Use the existing mode field in cache identity and preserve normal-mode behavior.

**Step 4: Verify green**

Run: `pytest tests/test_elite_mode.py tests/test_cache.py -q`
Expected: library and cache tests pass.

**Step 5: Commit**

`git add src/conclave/council.py tests/test_elite_mode.py tests/test_cache.py && git commit -m "feat(elite): expose elite council API"`

### Task 5: Make every elite phase auditable

**Files:**
- Modify: `src/conclave/manifest.py`
- Modify: `src/conclave/providers.py`
- Modify: `src/conclave/council.py`
- Test: `tests/test_manifest_all_modes.py`
- Test: `tests/test_secret_safety_matrix.py`

**Step 1: Write failing tests**

Test that successful elite manifests have initial/critique/revision receipts, unique
`providers_called`, aggregate usage for all recorded member phases, cache-hit manifests,
and `verified_no_secrets`. Inject secret-like provider errors and assert serialization stays
redacted. Test incomplete runs retain receipts for completed/attempted phases.

**Step 2: Verify red**

Run: `pytest tests/test_manifest_all_modes.py tests/test_secret_safety_matrix.py -q`
Expected: failures because receipts lack phase provenance and elite calls are absent.

**Step 3: Implement minimally**

Add optional `phase` to `ProviderExecutionReceipt` and `receipt_from_answer`. Build an elite
manifest from flattened phase artifacts, keeping `providers_called` unique and phase receipts
repeatable. Aggregate usage and latency using existing manifest conventions, record skipped
members without secrets, then rerun the existing secret-material scan before stamping safety.

**Step 4: Verify green**

Run: `pytest tests/test_manifest_all_modes.py tests/test_secret_safety_matrix.py -q`
Expected: manifest and secret-safety tests pass.

**Step 5: Commit**

`git add src/conclave/manifest.py src/conclave/providers.py src/conclave/council.py tests/test_manifest_all_modes.py tests/test_secret_safety_matrix.py && git commit -m "feat(elite): audit every protocol phase"`

### Task 6: Add the CLI surface

**Files:**
- Modify: `src/conclave/cli.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_cli_verdict.py`

**Step 1: Write failing tests**

Test `--mode elite` dispatch, human completion summary, full JSON serialization, incomplete
exit code 1 with reason, and explicit rejection of `--stream`. Confirm all existing modes'
CLI tests remain unchanged.

**Step 2: Verify red**

Run: `pytest tests/test_cli.py tests/test_cli_verdict.py -q`
Expected: elite is rejected as an invalid mode.

**Step 3: Implement minimally**

Add elite to mode validation/help and dispatch to `Council.elite_sync`. Render a concise
protocol summary plus the existing verdict panel. JSON emits the complete `CouncilResult`.
Return exit 1 for incomplete elite runs and reject stream before provider calls.

**Step 4: Verify green**

Run: `pytest tests/test_cli.py tests/test_cli_verdict.py -q`
Expected: CLI tests pass.

**Step 5: Commit**

`git add src/conclave/cli.py tests/test_cli.py tests/test_cli_verdict.py && git commit -m "feat(elite): add quality first CLI mode"`

### Task 7: Document behavior and verify the release candidate

**Files:**
- Modify: `README.md`
- Modify: `docs/PRODUCT_DESIGN_DOCUMENT.md`
- Modify: `SYSTEM_CONTEXT_DIAGRAM.md`
- Modify: `DOCUMENTATION_INDEX.md`
- Modify: `CHANGELOG.md`

**Step 1: Update existing docs**

Document the quality-first intent, three-member invariant, exact call stages, partial-failure
semantics, latency/cost trade-off, CLI/library examples, manifest phase receipts, and lack of
streaming. Keep the three core docs under 500 lines where feasible and do not claim a release.

**Step 2: Run focused verification**

Run: `pytest tests/test_elite_mode.py tests/test_manifest_all_modes.py tests/test_secret_safety_matrix.py tests/test_cli.py tests/test_cli_verdict.py tests/test_cache.py -q`
Expected: all focused tests pass.

**Step 3: Run full verification**

Run: `pytest -q`
Expected: zero failures.

Run: `ruff check .`
Expected: zero lint errors.

Run: `ruff format --check .`
Expected: all files already formatted.

**Step 4: Review safety and scope**

Run: `git diff --check && git status --short && git diff --stat main...HEAD`
Expected: no whitespace errors; only planned code, tests, and existing documentation changed.
Confirm no `.env`, credentials, generated caches, release tags, or publishing changes exist.

**Step 5: Commit**

`git add README.md docs/PRODUCT_DESIGN_DOCUMENT.md SYSTEM_CONTEXT_DIAGRAM.md DOCUMENTATION_INDEX.md CHANGELOG.md && git commit -m "docs: specify elite decision protocol"`
