# Elite Pilot v1 Protocol

## Purpose and decision boundary

This is a frozen, synthetic, exploratory pilot. It tests whether the H1 harness, task design,
atomic rubric, blinding, and adjudication process are usable. It cannot demonstrate that a
condition is better, justify a product-quality claim, or change a real decision. The primary
pilot outcome is method readiness for a separately authored confirmatory study.

## Frozen corpus and prompt

The study uses all 24 tasks in `public_tasks.json`: 12 per macro-family, four per subfamily,
eight per difficulty tier, and eight per keyed readiness class. Every condition receives the
same task, packet, prompt, and declared output-token ceiling. The prompt is:

> Using only the reference packet, choose the best course of action. Return: (1)
> Recommendation, (2) Readiness: ready|not_ready|indeterminate, (3) Hard-constraint check,
> (4) Conflicts/minority view, (5) Next actions and owners, (6) Unknowns. Cite packet IDs. Do
> not invent facts.

The six frozen comparison conditions are `single_frontier`, `self_refine`,
`independent_synthesis`, `critique_only`, `revision_only`, and `elite_full`. The manifest
predeclares the complete task-by-condition-by-replicate matrix and a seeded condition order.
Planned output-token ceilings must satisfy the harness's budget-match tolerance. The public
task hash, prompt, conditions, model and provider identities, generation settings, budgets,
replicate count, seed, software revision, and replay-artifact hashes are recorded before run.

## Execution

Only `public_tasks.json` may enter execution. `grader_keys.json` is grader-only and must be
kept outside the runner's readable inputs. Every planned cell produces exactly one record.
Timeouts, malformed output, abstentions, incomplete output, provider errors, and missing
records are non-successes; **failures remain in the denominator**. No cell may be rerun or
excluded because its answer is inconvenient. Any permitted infrastructure retry must be
predeclared and retain every attempt in the audit artifact.

This repository pack does not authorize paid calls. A live pilot requires a separately
approved spend ceiling, an automatic stop below that ceiling, pinned model identifiers, and a
fresh manifest. Offline replay is the default until those gates are satisfied.

## Atomic grading

Outputs are randomized behind opaque identifiers and presented without condition, provider,
model, timing, token, or execution-order labels. Views use a predeclared length policy; graders
may not see grader keys until model execution is complete. Two graders independently score
each successful output before discussion. They assign 0, 1, or 2 on exactly six dimensions:

1. `constraint_recall`
2. `conflict_minority_recognition`
3. `unsupported_claim_avoidance`
4. `recommendation_correctness`
5. `completeness_actionability`
6. `readiness_calibration`

For each dimension, 2 means all keyed elements are accurate and supported; 1 means the
direction is correct with one noncritical omission; 0 means a wrong, unsupported, or material
omission. Unsupported-claim avoidance is 2 only when no material fact is invented and
inferences are labeled. Completeness/actionability is 2 only when the response supplies the
decision, constraint check, conflict, concrete action, owner, trigger, and unknowns.

The primary metric is the failure-inclusive **critical-error-free decision rate**. Any keyed
critical error forces `critical_error_free=false` regardless of dimension totals. Secondary
outputs are the six separate dimension distributions, severe and unsupported error counts,
readiness confusion matrix, conflict/minority recall, tokens, latency, failures, and grader
minutes. No composite quality score is reported.

Grader disagreements retain both raw judgments and receive a separate adjudication citing
exactly the source judgments. Report Cohen's kappa, disagreement count, adjudication rate,
Wilson intervals for rates, and seeded task-paired bootstrap intervals for Elite-minus-each-
baseline differences. Report macro-family and subfamily results descriptively; this pilot is
not powered for confirmatory subgroup claims.

## Frozen exclusions and changes

Exclude no task or output after execution. Pre-execution rejection is allowed only for a
schema-invalid or hash-mismatched entire pack, which aborts the study rather than deleting
cells. Changes to a scenario, packet, key, prompt, condition, dimension, critical-error rule,
budget rule, or analysis create a new pack version. Pilot observations may improve a future
rubric and estimate variance; they may not alter, reveal, or train on confirmatory tasks.

