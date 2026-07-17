# H1 Method Hardening Design

**Linear:** DSE-708  
**Dependency:** PR #52 at `4365c28`  
**Gate:** Required before any paid study call

## Decision

Keep the existing H1 substrate as the execution primitive, then add a frozen study-method
contract and a balanced 24-task open-book synthetic QA pack. The committed pack validates
mechanics only; it cannot estimate unbiased effects or produce a product GO or KILL. A paid
pilot requires separately access-controlled grader keys and a hash frozen before execution.

## Frozen study contract

Every manifest must bind the exact base commit, study phase, task-family map, two roster specifications, provider/model revisions, condition prompt and protocol versions, generation settings, public-task and private-key hashes, rubric and grader-instruction hashes, evaluator and analysis-code versions, randomization and bootstrap settings, timeout/retry policy, exclusion/deviation policy, price snapshot, approved spend ceiling, and preregistration identifier/hash.

Confirmatory reports fail closed if any frozen field is missing or its observed receipt differs. Exploratory manifests are explicit and cannot be promoted after results exist.

## Experimental unit and randomization

The unit of analysis is the task. Roster and replicate are repeated measurements averaged within task. Create every task x roster x condition x replicate cell before execution. Derive an independent condition permutation from the master seed plus task and roster IDs; interleave blocks so time or provider order cannot align with one condition. All failed, timed-out, malformed, over-budget, abstained, incomplete, ungraded, and unresolved cells remain failures.

## Grading contract

Two independent graders score every successful output. Atomic records include rubric item, error category and severity, severe-error flag, holistic dimensions, reviewer seconds, confidence or abstention, rubric version/hash, grader batch/order, and condition/provider guess. Adjudication sees only genuine disagreements, cites every source record, and never overwrites raw judgments.

Non-success records bypass the human queue and are automatic failures. Successful grader views strip provider, model, condition, mode, and formatting fingerprints; normalize presentation deterministically; run a leakage scan; and retain a separate hashed/access-controlled blind map.

## Analysis and gates

Study reporting uses failure-inclusive Wilson rates, task-clustered paired bootstrap differences with roster/replicate averaging, raw agreement, prevalence, kappa, adjudication rate, severe errors, reviewer effort, token/cost distributions, latency distributions, deviations, and leakage guesses.

Reliability is undefined—not perfect—when expected agreement is one. A future paid pilot is
method-ready only with at least 95% double grading, raw agreement at least 80%, kappa at least
0.60 overall and no family below 0.50, adjudication at most 20%, and no material leakage.
Otherwise redesign. The committed open-book QA pack cannot satisfy or measure this gate.

Confirmation freezes one strongest baseline. GO requires a point gain of at least 10 percentage points with the paired 95% interval above zero, severe-error noninferiority within +2 points, readiness noninferiority within +5 points, reviewer-effort and latency gates, positive direction in every family and roster, and failure-inclusive validity. Simpler-baseline equivalence, harm, weak reliability, or post-hoc changes trigger redesign or kill per DSE-708.

## Synthetic QA pack

Use 24 synthetic, current-fact-free tasks: 12 operational-execution and 12
organizational-stewardship, balanced across six subfamilies, three difficulty tiers, and
`ready`/`not_ready`/`indeterminate`. Public packets and committed fixture keys are separate
files, but not separate security domains. Paid-study keys must be provisioned outside the
repository under access control and frozen by hash. Confirmatory tasks are independently
authored, access-restricted, and screened for sentence, numeric, structural, and semantic
overlap.

## Non-goals

No provider spend, held-out task publication, hosted dashboard, mutable pricing service, LLM primary grader, routing product, release, or quality claim in this increment.
