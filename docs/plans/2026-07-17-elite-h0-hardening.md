# Elite Horizon 0 Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close every Horizon 0 correctness gap in DSE-690 so Elite is execution-traceable, configuration-consistent, cache-safe, and incapable of reporting a decision-ready success when required protocol phases fail.

**Architecture:** Keep Elite inside the existing `Council` orchestration surface. Add stable Conclave-owned answer identity at the provider boundary, explicit readiness on `EliteResult`, a shared per-call receipt path that covers every phase, and a canonical versioned cache identity. Preserve backward-compatible `completed` semantics as protocol completion while making readiness a separate machine-readable contract.

**Tech Stack:** Python 3.11+, Pydantic, asyncio, httpx, pytest, pytest-asyncio, Ruff, Typer.

---

## Working rules

- Implement each task red-green-refactor: add the smallest failing test, run it and confirm the expected failure, implement the minimum fix, then rerun focused tests.
- Use a fresh serialized subagent for each bounded implementation task; do not allow concurrent edits in this worktree.
- After each subagent commit, review the diff and independently rerun its focused verification before continuing.
- Use this sandbox-safe pytest form:

  ```bash
  PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p no:cacheprovider -p pytest_asyncio.plugin <tests>
  ```

- Do not merge PR #51 or release a package. Horizon 0 ends at verified code, docs, Linear evidence, commit, and push.

## Task 1: Persistent answer identity and accurate claim-audit language

**Files:**

- Modify: `src/conclave/models.py`
- Modify: `src/conclave/providers.py`
- Modify: `src/conclave/prompts.py`
- Modify: `src/conclave/modes.py`
- Modify: `src/conclave/verdict.py`
- Modify: `src/conclave/verdict_synthesis.py`
- Test: `tests/test_providers.py`
- Test: `tests/test_elite_mode.py`
- Test: `tests/test_output_contract_plumbing.py`

1. Add failing tests proving every successful `call_model` answer receives a non-secret Conclave-owned stable ID and that IDs survive critique, revision, synthesis input, serialization, and reconstructed/cache-replayed results.
2. Add failing prompt/contract tests that use `claim audit`, `answer provenance`, or `within-run provenance`, and reject language implying answer IDs are external evidence.
3. Generate the ID deterministically from non-secret immutable response facts, with phase-specific derived IDs where a phase transforms an answer. Do not use provider names as the normal fallback.
4. Rename internal prompt text and docstrings from evidence audit to claim audit without breaking serialized field compatibility (`evidence_answer_ids` remains a compatibility field until a separately versioned schema migration).
5. Run focused tests and commit:

   ```bash
   git add src/conclave/models.py src/conclave/providers.py src/conclave/prompts.py src/conclave/modes.py src/conclave/verdict.py src/conclave/verdict_synthesis.py tests/test_providers.py tests/test_elite_mode.py tests/test_output_contract_plumbing.py
   git commit -m "fix(elite): preserve answer identity and claim provenance"
   ```

## Task 2: Thread resolved configuration through every model call

**Files:**

- Modify: `src/conclave/council.py`
- Modify: `src/conclave/verdict_synthesis.py`
- Test: `tests/test_council.py`
- Test: `tests/test_elite_mode.py`
- Test: `tests/test_output_contract_plumbing.py`

1. Add failing spies proving member fan-out, synthesis, verdict extraction, and verdict repair all receive the same resolved `ConclaveConfig` instance.
2. Extend verdict extraction with an optional config parameter and pass it to both initial and repair `call_model` invocations.
3. Pass `self.config` through `Council.fan_out`, `Council.synthesize_blocks`, and `_extract_verdict`.
4. Preserve public call compatibility by keeping new arguments keyword-only and optional where functions are public.
5. Run focused tests and commit `fix(config): thread resolved config through council calls`.

## Task 3: Make cache identity protocol- and version-aware

**Files:**

- Modify: `src/conclave/cache.py`
- Modify: `src/conclave/council.py`
- Modify: `src/conclave/prompts.py`
- Modify: `src/conclave/verdict.py`
- Test: `tests/test_cache.py`
- Test: `tests/test_secret_safety_matrix.py`

1. Add failing tests showing cache keys differ when protocol version, prompt version, verdict schema version, model roster, generation settings, verdict extraction flag, relevant resolved config, or evidence-bundle digest changes.
2. Define explicit protocol/prompt/schema version constants next to their owners and include them in one canonical secret-free cache identity document.
3. Add an optional `source_bundle_digest` cache-key input now so Horizon 2 source grounding cannot silently reuse ungrounded entries.
4. Hash only normalized, non-secret configuration identity; prove API keys and raw secret values cannot enter keys or payloads.
5. Preserve cache replay compatibility by treating old entries as misses rather than attempting unsafe migration.
6. Run focused tests and commit `fix(cache): version elite protocol identity`.

## Task 4: Separate protocol completion from decision readiness

**Files:**

- Modify: `src/conclave/models.py`
- Modify: `src/conclave/modes.py`
- Modify: `src/conclave/council.py`
- Modify: `src/conclave/cli.py`
- Test: `tests/test_elite_mode.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_cache.py`

1. Add failing tests for `ready`, `not_ready`, and `indeterminate`, including machine-readable reasons.
2. Keep `completed` as the factual protocol-completion flag. Add `decision_readiness` and `readiness_reasons` to `EliteResult` with conservative defaults.
3. Define readiness deterministically: required successful synthesis and, when configured, a valid verdict are necessary for `ready`; partial/failed required phases become `not_ready`; insufficient or ambiguous evidence becomes `indeterminate`.
4. Change CLI success reporting and exit behavior so protocol completion alone cannot imply decision readiness. JSON must expose both dimensions.
5. Verify cache serialization/replay preserves the new fields and old cached shapes default conservatively.
6. Run focused tests and commit `fix(elite): separate completion from readiness`.

## Task 5: Capture complete call receipts and accounting

**Files:**

- Modify: `src/conclave/models.py`
- Modify: `src/conclave/manifest.py`
- Modify: `src/conclave/providers.py`
- Modify: `src/conclave/council.py`
- Modify: `src/conclave/verdict_synthesis.py`
- Modify: `src/conclave/modes.py`
- Test: `tests/test_manifest_all_modes.py`
- Test: `tests/test_elite_mode.py`
- Test: `tests/test_output_contract_plumbing.py`

1. Add failing tests asserting receipts cover member, audit, revision, synthesis, verdict extraction, and repair attempts, including failed calls.
2. Introduce a secret-free call receipt carrying phase, provider/model identity, attempt/outcome, latency, available usage/cost, error category, prompt version, and schema version.
3. Return or collect receipts through explicit values/callbacks; do not depend on global mutable state or logs.
4. Aggregate receipt latency/usage/cost into the manifest while preserving `None` when a provider does not report a trustworthy value.
5. Ensure prompt content, API keys, raw provider bodies, and exception chains never enter receipts.
6. Run focused tests plus the secret matrix and commit `feat(manifest): account for every elite model call`.

## Task 6: Align product documentation with verified behavior

**Files:**

- Modify: `README.md`
- Modify: `docs/PRODUCT_DESIGN_DOCUMENT.md`
- Modify: `docs/DOCUMENTATION_INDEX.md`
- Modify: `docs/plans/2026-07-17-decision-quality-roadmap.md`
- Modify: other tracked docs returned by the claim scan

1. Scan tracked prose for `fully auditable`, `evidence audit`, `consensus means truth`, and claims that answer IDs prove external facts.
2. Document the exact distinction among execution traceability, answer provenance, source grounding, protocol completion, and decision readiness.
3. Keep Horizon 1-4 claims explicitly prospective and preserve the roadmap's falsification/kill gates.
4. Run documentation/whitespace checks and commit `docs: align elite claims with verified guarantees`.

## Task 7: Completion audit, Linear evidence, and remote synchronization

1. Audit each DSE-690 Horizon 0 checkbox against code, tests, and documentation. Fix any gap before checking it off.
2. Run the complete verification gate:

   ```bash
   PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p no:cacheprovider -p pytest_asyncio.plugin
   ruff check .
   ruff format --check .
   git diff --check origin/main...HEAD
   git grep -nE 'sk_live|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9_]+' -- . ':!uv.lock'
   ```

3. Update DSE-690 with checked acceptance items and a concise evidence comment containing commit SHA, test counts, PR #51, and any deliberately deferred Horizon 1-4 work.
4. Push `feat/elite-decision-protocol`, verify the remote SHA, and confirm PR #51 remains draft.
5. Stop. Do not merge or publish.
