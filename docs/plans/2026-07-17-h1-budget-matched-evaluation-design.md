# H1 Budget-Matched Evaluation Design

**Linear:** DSE-708
**Status:** Approved direction; implementation begins offline
**Base:** Elite H0 commit `23e8fce437172c040084d31dcea4ed5737ebae2a`

## Decision

Build a narrow, experimental `conclave.evals` package that can plan, replay, blind, score, and report a fixed six-condition study. It is an evidence instrument for Elite, not a general evaluation platform and not a new public council mode.

## Study contract

The six frozen conditions are:

1. `single_frontier`
2. `self_refine`
3. `independent_synthesis`
4. `critique_only`
5. `revision_only`
6. `elite_full`

Every task-condition-replicate cell is declared before execution. Missing, timed-out, malformed, abstained, and incomplete cells remain in the denominator. Conditions receive identical public task material and reference packets. Grader-only keys live in a separate file that the runner never loads.

## Architecture

- `models.py`: versioned Pydantic task, condition, run, score, and study-manifest contracts.
- `dataset.py`: load and hash public tasks; separately load grader keys only for scoring.
- `protocols.py`: immutable six-condition registry and token-budget allocation.
- `replay.py`: record/replay at `conclave.transport.post_json`; match sanitized request identity plus occurrence index; reject missing, extra, incompatible, or secret-bearing artifacts.
- `runner.py`: create a seeded task x condition x replicate matrix, enforce budgets, preserve failures, and emit atomic artifacts.
- `blinding.py`: deterministic seeded opaque output IDs and a separate restricted blind map.
- `scoring.py`: atomic judgments, adjudication without overwriting raw scores, Wilson intervals, paired bootstrap differences, and Cohen's kappa.
- `eval_cli.py`: `plan`, `run`, `blind`, and `report` commands. The main CLI exposes these under `conclave eval` only after the offline substrate is stable.

## Budget matching

The study manifest declares one output-token ceiling per task-condition cell. Provider adapters receive `max_output_tokens` end to end. Planned ceilings must be within 5% across conditions; actual tokens, latency, failures, and unused budget are reported. Token ceilings are a reproducible provider-spend proxy; a later live-study manifest may add frozen price metadata without placing mutable pricing in library code.

## Replay and security

Replay performs zero network calls. Request identities include provider-safe URL components, model, normalized non-secret body, and occurrence index; authorization headers, API keys, raw endpoint credentials, and exception chains are never serialized. Replay fails closed on schema/version/hash mismatch or unmatched calls.

## Analysis

The primary endpoint is failure-inclusive critical-error-free decision rate. Reports retain task-level paired outcomes and show distributions and uncertainty: Wilson intervals for rates, seeded paired bootstrap intervals for condition differences, and kappa/adjudication rates for graders. Pilot results are exploratory; held-out results require a frozen preregistration.

## Delivery boundaries

This increment excludes provider spend, a hosted dashboard, retrieval, embeddings, routing, mutable pricing tables, LLM-as-primary-grader, and public product claims. H0 must be merged and pinned before a confirmatory live run. Any paid pilot requires an explicit spend ceiling.
