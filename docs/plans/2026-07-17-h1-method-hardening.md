# H1 Method Hardening Implementation Plan

> **Linear:** DSE-708. Use strict TDD on `feat/h1-pilot-pack`; make zero live provider calls.

## Phase 1 — Freeze contract and blocked plan

1. Add failing tests in `tests/evals/test_study_design.py` for complete provenance, two rosters, task-family mapping, cost/spend and policy freezes, tamper detection, exploratory/confirmatory separation, and deterministic task x roster blocked permutations.
2. Extend `src/conclave/evals/models.py` and `protocols.py`; keep old manifests readable only as synthetic exploratory artifacts.
3. Re-run focused tests and commit.

## Phase 2 — Atomic grading and blinding

1. Add failing tests for typed atomic errors, severe-error accounting, reviewer seconds, confidence/abstain, grader order/guess, complete double grading, raw agreement/prevalence, undefined constant kappa, and family/roster reliability.
2. Extend `scoring.py`, `blinding.py`, and models. Queue only successful outputs, normalize views, scan leakage, and hash the separate blind map.
3. Re-run focused tests and commit.

## Phase 3 — Confirmatory estimands and refusal gates

1. Add tests for task-level roster averaging, task-clustered paired bootstrap, one-sided severe-error and effort/latency noninferiority boundaries, symmetric task exclusions, deviations, complete cost/latency receipts and summaries, and report refusal on freeze drift or exploratory evidence.
2. Extend `scoring.py`, `reporting.py`, and CLI artifact validation. Archive every gate input and seed.
3. Re-run focused tests and commit.

## Phase 4 — Synthetic QA pack

1. Add `studies/elite_qa_v1/public_tasks.json`, committed fixture `grader_keys.json`,
   `qa_protocol.md`, `confirmatory_preregistration.md`, and `README.md`. Classify the pack as
   open-book harness QA, not a paid pilot.
2. Add `tests/evals/test_qa_pack.py`: exactly 24 unique tasks, 12/12 families, 4 per
   subfamily, balanced tier/readiness, matching nonempty fixture keys, packet citation IDs, no
   grader material in public records, no duplicates, and documented leakage/holdout rules.
3. Generate and validate an offline manifest only; do not create model outputs.
4. Re-run focused tests and commit.

## Phase 5 — Verify and hand off

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider -q
python -m ruff check .
python -m ruff format --check .
git diff --check
gitleaks git --redact
```

Push, open a draft PR stacked on #52, obtain independent methods/code review, and attach
evidence to DSE-708. Paid pilot execution remains blocked until PR #51 is merged/pinned,
Ernest approves a hard spend ceiling, and separately access-controlled grader keys are frozen
by hash before execution.
