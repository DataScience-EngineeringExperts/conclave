# H1 Live Evaluation Runner Design

**Linear:** DSE-708
**Evidence class:** Paid exploratory pilot only
**Approved ceiling:** USD 10.00
**Boundary:** Verify correctness with minimal paid calls; do not study efficiency or make a
decision-quality claim.

## Decision

Add a sequential, checkpointed execution path under `conclave.evals`. The existing
`run_study` matrix remains the cell-level primitive, while an eval-only live provider client
owns every paid call. Before each call, the client computes a pessimistic reservation from a
manifest-bound price snapshot, writes that reservation to an atomic checkpoint, and refuses
the call if committed cost plus the reservation would exceed USD 10.00. It permits exactly
one in-flight provider request.

This is the smallest approach that makes the cap enforceable. Calling `Council` directly is
not suitable because `fan_out` intentionally launches concurrent calls and does not expose a
pre-call cost gate. A subprocess wrapper would isolate calls but add process, credential, and
checkpoint complexity without improving the accounting invariant. The live runner therefore
reuses the existing adapter/provider highway through `call_model`, but owns condition
orchestration, call order, reservations, and persistence.

## Components

- `pricing.py` defines and loads an external, immutable price book. It validates exact
  provider/model/revision coverage, canonical hashing, USD currency, positive pessimistic
  input/output rates, a positive operator-attested `max_output_bytes_per_token`, and the
  `FrozenStudyDesign.price_snapshot` binding. No current prices are compiled into library code.
- `live_protocols.py` defines the six versioned condition call graphs and deterministic
  per-stage output-token allocation. Calls are awaited serially and all stage caps sum to the
  cell's frozen `max_output_tokens`.
- `live.py` owns the guarded provider client, bounded call receipts, atomic checkpoint state,
  interruption recovery, dry-run estimation, and final `StudyRun` assembly.
- `eval_cli.py` adds a separate `conclave eval live` surface. Dry-run is the default. Paid
  execution requires both `--execute` and an `--approve-spend-usd` value that exactly matches
  the frozen manifest ceiling, plus `--checkpoint-seal-key-file` naming an owner-only POSIX
  regular file containing at least 32 operator-generated random bytes.

The new path accepts only a complete `paid_exploratory_pilot` manifest with a frozen design.
It rejects legacy synthetic manifests and confirmatory manifests.

## Frozen price snapshot and reservation

The operator supplies a JSON price book containing a snapshot ID, capture timestamp,
currency, and one pessimistic rate entry for every exact provider/model/revision in every
roster. The canonical entry hash must equal `FrozenStudyDesign.price_snapshot.prices_hash`;
the snapshot ID, capture time, and currency must also match. Duplicate or missing entries,
unknown models, non-USD currency, and nonpositive rates fail before key resolution or network
access.

Execution reservations use the exact UTF-8 byte length of resolved messages plus provider
framing and the call's output-token ceiling; treating every input byte as a possible token is
pessimistic. Dry-run cannot know future upstream text, so it removes its internal sentinels and
reserves each upstream token ceiling multiplied by the maximum
`max_output_bytes_per_token` attestation in the frozen price book. That attestation is part of
the canonical price hash, so drift invalidates the manifest binding. The price book uses
maximum input/output rates, not discounted or cached rates. `Decimal` arithmetic rounds upward
to USD microcents. Tests pin the formula and multibyte maximum-expansion case.

Before network I/O:

1. Acquire the runner's single-call lock.
2. Calculate the reservation from the exact frozen call context.
3. Refuse if `committed_usd + reservation_usd > 10.00`.
4. Atomically persist a pending call with the reservation.
5. Invoke `call_model` with that stage's `max_output_tokens`.

Afterward, complete usage is priced from provider token receipts. The runner commits the
actual calculated cost when it is within the reservation; missing usage commits the full
reservation. A usage count or calculated cost above the reservation is a fail-closed
`reservation_breach`: the full reservation is charged, the cell is non-successful, and no
new call is scheduled. The final artifacts distinguish usage-priced cost from pessimistically
charged cost.

## Six conditions

Every condition receives the same `PublicTask.prompt` and `reference_packets`. Roster order
is already frozen; the first member is the single-frontier and synthesis member, while all
members participate in multi-model stages. Paid manifests must freeze at least three members
per roster so `elite_full` exercises its implemented three-responder contract.

1. `single_frontier`: the first roster member produces the final answer.
2. `self_refine`: the first member drafts, then revises its own answer.
3. `independent_synthesis`: all members answer independently, then the first member produces
   the final synthesis.
4. `critique_only`: all members answer, all surviving members audit the anonymized answer
   set, then the first member synthesizes without a revision stage.
5. `revision_only`: all members answer, each sees anonymized peer answers and revises without
   a separate claim-audit stage, then the first member synthesizes.
6. `elite_full`: all members answer, audit claims, and revise using the existing versioned
   Elite prompt builders; the first member then produces the same synthesis and structured
   verdict stages required by the current Elite protocol.

A frozen allocation table divides each cell ceiling across its maximum call graph, including
Elite's optional verdict repair. Version `live_stage_minimum_caps_v1` reserves 256 output tokens
for `initial`, `draft`, and `critique`; 384 for `self_revision` and `revision`; 512 for
`synthesis`; and 768 for each of `verdict` and `verdict_repair`. These conservative floors give
short reasoning stages useful space, give integration stages more room, and fit the current
multi-position structured verdict on either attempt. A three-member Elite cell therefore
requires at least 4,736 output tokens before any call. Extra tokens are distributed
deterministically and exactly; integer remainder goes to the normal graded-output stage while
the optional repair retains its full floor. Failed responder gates stop later stages and produce
a failure-inclusive cell record. The graded output is the condition's canonical final decision
artifact, never an internal critique or unvalidated extractor text.

## Data flow

```text
manifest + public tasks + external price book
                 |
                 v
        validate hashes and live gates
                 |
                 v
    dry-run same frozen call graph (no keys/network)
                 |
          --execute + exact $10 approval
                 |
                 v
 next planned cell -> stage reservation -> atomic pending checkpoint
                 |                         |
                 v                         v
          one call_model await       crash-safe evidence
                 |
                 v
       bounded receipt + checkpoint -> next stage/cell
                 |
                 v
      complete StudyRun + replay-safe call receipts
```

The runner never loads grader keys. Final `StudyRun` records remain compatible with existing
blinding and scoring. A companion live receipt artifact records call IDs, planned-run IDs,
stages, provider/model identities, caps, usage, latency, reservation, charged cost, cost
basis, outcome, and bounded error category.

## Checkpoint and resume

Checkpoint format `conclave_live_checkpoint_v2` is authenticated with HMAC-SHA256 under the
external seal key. Old unkeyed checkpoints, a wrong key, or a publicly recomputed SHA-256
digest fail closed. The key is required by the paid `run_live_study` API but is never serialized.
The authenticated payload binds the manifest hash, price-book hash, public-task hash, USD
10.00 ceiling, records, receipts, committed cost, and active/pending state. It never stores
headers, credential values, endpoint query secrets, raw exceptions, grader material, or the
seal key.

Every transition is written to a temporary file in the destination directory, flushed,
`fsync`ed, and installed with `os.replace`. Before replacement, serialized content must be
unchanged by the existing secret redactor and must not contain any active provider-key value.

Resume is deliberately cell-granular. If a process stopped with an active cell, the runner
does not repeat any call from that cell. It charges an unresolved pending reservation at its
full amount, records the cell as `incomplete` with
`interrupted_cell_not_retried`, clears the active state, and continues with the next
unrecorded planned cell. This may waste a small amount of budget, but prevents duplicate
billing and post-crash selection bias.

When a reservation cannot fit, the runner makes no call and materializes the current and all
remaining cells as `incomplete` with `budget_exhausted`. The matrix stays complete and every
non-executed cell remains in the denominator.

## Dry-run and replay

Dry-run traverses the exact same condition call graph without loading configuration, reading
provider or checkpoint-seal keys, or invoking transport. It reports planned cells, maximum provider calls, worst-case
reserved cost by roster and condition, largest single reservation, total upper bound, the
USD 10.00 ceiling, and whether the complete plan fits. It is an authorization aid, not a bill
forecast.

Committed test fixtures contain fictional tasks, fake keys supplied only through test
environment variables, a test-only price book, and a sanitized transport replay. The replay
integration executes all six conditions through real adapters with zero network calls,
asserts exact consumption, and produces the same call receipts and `StudyRun` on repeat.

## Failure modes

| Failure | Behavior |
|---|---|
| Snapshot, manifest, task, or checkpoint hash drift | Abort before keys or network. |
| Missing/duplicate/unknown price entry | Abort before keys or network. |
| Missing/short/insecure seal-key file or unsupported POSIX checks | Abort before live execution. |
| Old, forged, or wrong-key checkpoint | Reject before state or cost is trusted. |
| Next reservation crosses USD 10.00 | No call; mark remaining cells `budget_exhausted`. |
| Provider timeout/error/malformed output | Bounded failure receipt; retain cell in denominator. |
| Missing usage | Charge full reservation; mark cost basis pessimistic. |
| Usage exceeds reserved bound | Charge reservation, stop scheduling, record breach. |
| Crash with pending call | Charge full reservation; do not repeat interrupted cell. |
| Checkpoint cannot be persisted | Do not make the next provider call. |
| Secret-like content in checkpoint payload | Reject write and stop. |
| Replay mismatch or unconsumed record | Fail closed; no fallback to network. |

## Acceptance gates

- Dry-run and replay make zero network calls and load no live keys.
- The six condition graphs and stage allocations are versioned, deterministic, and complete.
- Tests prove no more than one provider request is active and every request has a persisted
  reservation first.
- Total charged cost never exceeds the manifest-bound USD 10.00 ceiling in success, timeout,
  missing-usage, budget-exhaustion, and crash/resume tests.
- Resume never repeats an interrupted paid cell.
- Tests reject public resealing, wrong keys, old formats, and group/other-readable key files.
- Maximum-expansion multibyte execution reservations do not exceed the dry-run estimate.
- Every planned cell appears once in the final `StudyRun`, including unscheduled failures.
- Live/checkpoint/replay artifacts contain no credential values or raw exception chains.
- Focused tests, full pytest, Ruff lint/format, `git diff --check`, and Gitleaks pass.

This increment ends after a minimal paid correctness smoke test. It does not tune prompts,
compare efficiency, estimate variance, grade model quality, run the 24-task pack at scale, or
support a GO/REDESIGN/KILL conclusion.
