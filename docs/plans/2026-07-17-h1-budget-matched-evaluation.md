# H1 Budget-Matched Evaluation Implementation Plan

> **Linear:** DSE-708. Execute with test-driven development on stacked branch `feat/h1-budget-matched-eval`; make no live provider calls in this plan.

**Goal:** Deliver an offline, reproducible substrate that can fairly compare Elite with five fixed budget-matched alternatives.

**Architecture:** Add an experimental `conclave.evals` package around versioned artifacts. Keep provider I/O at the existing transport seam, thread output-token caps through adapters, and keep study execution separate from the public council API.

## Phase 1 — Contracts and deterministic planning

1. Add failing tests in `tests/evals/test_models.py` and `tests/evals/test_protocols.py` for schema versions, the exact six-condition registry, ±5% planned-budget enforcement, stable hashes, deterministic seeded order, and complete task x condition x replicate matrices.
2. Run `pytest tests/evals/test_models.py tests/evals/test_protocols.py -q` and confirm RED.
3. Implement `src/conclave/evals/__init__.py`, `models.py`, `dataset.py`, and `protocols.py` minimally.
4. Re-run focused tests, Ruff, and commit.

## Phase 2 — Output caps and strict replay

1. Add failing adapter/provider tests proving `max_output_tokens` reaches OpenAI-compatible, Anthropic, and Gemini request bodies without changing default requests.
2. Add failing `tests/evals/test_replay.py` cases for record/replay identity, repeated-call occurrence indexes, zero-network replay, secret exclusion, and fail-closed missing/extra/version-mismatched artifacts.
3. Thread optional `max_output_tokens` through `ProviderAdapter`, concrete adapters, `call_model`, and streaming paths; implement `src/conclave/evals/replay.py` at `transport.post_json`.
4. Run focused tests, existing adapter/provider/transport tests, Ruff, and commit.

## Phase 3 — Runner and blinding

1. Add failing `tests/evals/test_runner.py` and `test_blinding.py` for all six conditions, per-cell caps, immutable failures in denominators, receipts/totals, seeded opaque IDs, and a separate blind map.
2. Implement `src/conclave/evals/runner.py`, `protocols.py`, and `blinding.py`. Protocol executors may use recorded fixtures only in this increment.
3. Re-run focused tests and commit.

## Phase 4 — Atomic scoring and reports

1. Add failing `tests/evals/test_scoring.py` for raw grader preservation, adjudication, critical-error-free rates, Wilson intervals, seeded paired bootstrap intervals, and Cohen's kappa.
2. Implement `src/conclave/evals/scoring.py` and machine-readable plus Markdown report writers.
3. Add a small synthetic fixture set under `tests/fixtures/evals/`; label every result exploratory/synthetic.
4. Re-run focused tests and commit.

## Phase 5 — CLI and documentation

1. Add failing `tests/evals/test_cli.py` for `conclave eval plan`, `run --replay`, `blind`, and `report`; reject live execution unless explicitly enabled and configured.
2. Implement `src/conclave/eval_cli.py` and mount its Typer app in `src/conclave/cli.py`.
3. Update `README.md`, `DOCUMENTATION_INDEX.md`, and `SYSTEM_CONTEXT_DIAGRAM.md` with the experimental boundary and reproducible commands.
4. Run CLI tests and commit.

## Final verification and handoff

Run:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
gitleaks git --redact
```

Then push the stacked branch, open a PR linked to DSE-708 and PR #51, request independent review, and attach test/scan evidence to Linear. Do not merge, publish, or run paid models without the required approval.
