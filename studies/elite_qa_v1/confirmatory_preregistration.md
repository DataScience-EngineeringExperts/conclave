# Confirmatory Study Preregistration Gate

> **Status:** This is **not a completed preregistration** and does not authorize a
> confirmatory run. Every bracketed field must be resolved, independently reviewed, signed,
> hashed, and frozen before holdout unsealing or provider execution.

## Required declarations

- Study ID and immutable artifact location: `[required]`
- Owners for methods, corpus, execution, grading, and adjudication: `[required]`
- Primary hypothesis and direction: `[required]`
- Primary endpoint: failure-inclusive critical-error-free decision rate: `[confirm]`
- Primary comparison and multiplicity treatment: `[required]`
- Minimum practically important effect: `[required]`
- Power, alpha, sample-size method, and assumptions justified independently of this open-book
  QA pack: `[required]`
- Two prespecified macro-families and minimum family-specific sample sizes: `[required]`
- Conditions, prompts, roster, provider/model versions, budgets, replicates, and seed: `[required]`
- Inclusion, failure, retry, exclusion, stopping, and missing-data rules: `[required]`
- Secondary dimensions, calibration analysis, grader-time, cost, and latency guardrails: `[required]`
- Go, redesign, and kill thresholds: `[required]`

## Independent holdout construction

The confirmatory corpus must be authored as a separate project by people who cannot see QA or
paid-pilot outputs while authoring keys. It must use **new scenario archetypes**, entities, numbers,
packet prose, constraint interactions, and minority-view traps. Simple **parameter swaps** or
paraphrases of QA tasks are prohibited. QA task or answer text may not enter examples,
few-shot context, prompt tuning, grader training, or rubric demonstrations.

Before acceptance, run exact-ID and exact-text checks, an **eight-token** contiguous-overlap
screen, MinHash or equivalent near-duplicate detection, embedding similarity review, and a
manual semantic-leakage review. Search repository code, tests, documentation, replay fixtures,
model-facing prompts, and prior reports. Replace a leaked or semantically duplicated holdout
task; do not repair it by editing only its answer key.

Public tasks and grader keys remain separate. The execution identity cannot read the keys.
Corpus authors do not grade outputs they authored unless declared and sensitivity-tested.
Grader training uses separate calibration examples that are in neither this QA pack nor the
holdout.

Committed repository keys are fixtures only. Any paid pilot or confirmatory run requires a
separately access-controlled key artifact, unavailable to models and runner identities, whose
cryptographic hash is frozen in the study manifest before execution.

## Freeze and unsealing gate

Before **unsealing** any confirmatory task to a model, freeze and publish internally the
cryptographic hashes for public tasks, grader keys, prompt, conditions, roster, model/provider
versions, generation settings, token and dollar ceilings, replicate count, seed, blind-map
procedure, rubric, critical errors, grader training set, exclusions, retry policy, analysis
code, primary comparison, minimum effect, sample size, stopping rules, and decision gates.

An independent reviewer must attest that:

1. this QA pack, any paid pilot, and the holdout are semantically distinct;
2. no model, protocol author, or runner has received the access-controlled study keys;
3. budgets and evidence access are condition-matched;
4. the full failure-inclusive analysis can run from frozen artifacts;
5. provider spend and automatic-stop authority are approved; and
6. H0 and the H1 harness revision are merged, pinned, and green.

If any item fails, the confirmatory study remains sealed. After unsealing, no prompt, task,
key, rubric, roster, budget, outcome rule, or analysis change is permitted. Deviations are
logged and the primary result remains based on the preregistered analysis.

## Reporting boundary

Report task-paired effects and uncertainty, family distributions, severe-error and calibration
guardrails, execution failures, tokens, dollar cost, latency, reviewer minutes, raw grader
agreement, and adjudication. A go decision requires the preregistered improvement and no
material regression on severe errors, readiness calibration, reviewer effort, cost, or
latency. Results that miss the gate trigger redesign or kill; they must not be reframed as
proof through post-hoc slices.
