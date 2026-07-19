# H1 Live Evaluation Runner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a resumable paid-exploratory runner that executes the six frozen H1 conditions with one provider call in flight and a manifest-bound USD 10.00 hard ceiling.

**Architecture:** Keep the current failure-inclusive `run_study` contract and add an eval-only sequential call gateway. Bind every pre-call reservation to an external frozen price book, checkpoint before network I/O, and use explicit six-condition executors instead of concurrent `Council.fan_out`.

**Tech Stack:** Python 3.11, Pydantic v2, Typer, asyncio, Decimal, existing Conclave adapters/`call_model`, pytest/pytest-asyncio, Ruff, Gitleaks.

---

Use @test-driven-development for every production change and
@verification-before-completion before the handoff. Do not call a live provider until Tasks
1-8 pass in replay and the operator has reviewed the dry-run estimate. This plan is paid
exploratory infrastructure only; do not add an efficiency study, quality comparison, or
confirmatory decision gate.

### Task 1: Frozen price-book contracts and manifest binding

**Files:**
- Create: `src/conclave/evals/pricing.py`
- Create: `tests/evals/test_pricing.py`
- Create: `tests/fixtures/evals/live_smoke/price_book.json`

**Step 1: Write the failing contract tests**

Add tests named:

- `test_price_book_hash_is_canonical_and_binds_exact_frozen_snapshot`
- `test_price_book_rejects_duplicate_missing_unknown_or_revision_drift`
- `test_price_book_requires_usd_positive_pessimistic_rates`
- `test_call_reservation_rounds_up_and_covers_input_output_and_framing`

The fixture must use fictional provider/model IDs and rates. Pin a reservation example with
`Decimal`, including prompt-token upper bound, provider framing allowance, inserted upstream
token ceilings, and the stage output cap. Assert no float arithmetic reaches the result.

**Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_pricing.py -q
```

Expected: FAIL during collection because `conclave.evals.pricing` does not exist.

**Step 3: Implement the minimal immutable price book**

In `pricing.py`, add frozen Pydantic contracts for `ModelPrice`, `PriceBook`, and
`CallReservation`; canonical JSON hashing; JSON loading; exact roster coverage validation;
and `reserve_call_cost(...)`. Use ceiling rates per million tokens and quantize upward to
`Decimal("0.000001")`. Validate snapshot ID, capture time, currency, and canonical entry hash
against `FrozenStudyDesign.price_snapshot`.

Do not add real provider prices to package code or documentation.

**Step 4: Run focused tests and Ruff**

Run the Step 2 command, then:

```bash
python -m ruff check src/conclave/evals/pricing.py tests/evals/test_pricing.py
python -m ruff format --check src/conclave/evals/pricing.py tests/evals/test_pricing.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/conclave/evals/pricing.py tests/evals/test_pricing.py tests/fixtures/evals/live_smoke/price_book.json
git commit -m "feat(evals): freeze live price snapshots"
```

### Task 2: Deterministic six-condition call graphs

**Files:**
- Create: `src/conclave/evals/live_protocols.py`
- Create: `tests/evals/test_live_protocols.py`
- Modify: `src/conclave/evals/protocols.py`

**Step 1: Write failing call-graph tests**

Add tests named:

- `test_live_registry_covers_exactly_six_versioned_conditions`
- `test_stage_caps_are_positive_deterministic_and_sum_to_cell_ceiling`
- `test_single_and_self_refine_use_only_frozen_lead_member`
- `test_multi_model_conditions_call_members_in_frozen_order`
- `test_critique_revision_and_elite_prompts_are_anonymized`
- `test_elite_uses_current_versioned_prompt_builders_and_three_responder_gate`
- `test_too_small_cell_budget_fails_before_any_call`

Use a fake sequential client that records `(stage, provider, model, messages, cap)` and
returns deterministic answers. Assert the final output is a decision artifact, not an
internal critique. For a three-member roster, assert exact call order for all six conditions
and assert the recorded maximum concurrency is one.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_protocols.py -q
```

Expected: FAIL because the live protocol registry and executors are absent.

**Step 3: Add the minimal versioned protocol implementation**

Implement a frozen `StageCall` contract, one allocation table, public-task prompt assembly,
and six executor functions. Reuse existing Elite prompt constants/builders for
`elite_full`; do not copy their text. Use the first frozen roster member as lead/synthesizer,
all members in the multi-model stages, and sequential `await` calls only. Reject paid live
rosters with fewer than three members.

Keep `protocols.py` as the canonical six-ID registry and expose only the live registry hook
needed by the runner.

**Step 4: Verify GREEN and regress the offline planner**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_protocols.py tests/evals/test_protocols.py tests/test_elite_mode.py -q
python -m ruff check src/conclave/evals/live_protocols.py src/conclave/evals/protocols.py tests/evals/test_live_protocols.py
python -m ruff format --check src/conclave/evals/live_protocols.py src/conclave/evals/protocols.py tests/evals/test_live_protocols.py
```

Expected: PASS with existing offline and Elite behavior unchanged.

**Step 5: Commit**

```bash
git add src/conclave/evals/live_protocols.py src/conclave/evals/protocols.py tests/evals/test_live_protocols.py
git commit -m "feat(evals): define live condition protocols"
```

### Task 3: Guarded provider client and bounded receipts

**Files:**
- Create: `src/conclave/evals/live.py`
- Create: `tests/evals/test_live_gateway.py`
- Modify: `src/conclave/evals/__init__.py`

**Step 1: Write failing gateway tests**

Add tests named:

- `test_gateway_persists_reservation_before_calling_provider`
- `test_gateway_allows_only_one_in_flight_call`
- `test_gateway_rejects_call_that_would_cross_hard_cap`
- `test_gateway_prices_complete_usage_and_charges_reservation_when_missing`
- `test_gateway_stops_on_reservation_breach`
- `test_gateway_receipt_contains_bounded_error_category_not_raw_exception`

Inject a fake `call_model` coroutine and checkpoint callback. The fake must assert that a
pending reservation already exists when it starts. Launch two gateway calls with
`asyncio.gather` only to prove the internal lock holds maximum active calls at one. Use caps
small enough to exercise the boundary exactly.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_gateway.py -q
```

Expected: FAIL because `LiveProviderClient` and receipt contracts are absent.

**Step 3: Implement the minimal gateway**

Add frozen `ProviderCallReceipt`, `PendingCall`, and cost-basis contracts plus
`LiveProviderClient`. Resolve price before keys, acquire one `asyncio.Lock`, persist pending
state before `await call_model`, always pass the stage `max_output_tokens`, map errors to the
same bounded categories as provider receipts, and reconcile `ModelAnswer.usage` against the
reservation. Never retain messages, headers, URLs, keys, or raw exception text in receipts.

**Step 4: Verify GREEN and provider regressions**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_gateway.py tests/test_providers.py tests/test_adapters.py -q
python -m ruff check src/conclave/evals/live.py tests/evals/test_live_gateway.py
python -m ruff format --check src/conclave/evals/live.py tests/evals/test_live_gateway.py
```

Expected: PASS; default provider requests remain unchanged.

**Step 5: Commit**

```bash
git add src/conclave/evals/live.py src/conclave/evals/__init__.py tests/evals/test_live_gateway.py
git commit -m "feat(evals): guard paid provider calls"
```

### Task 4: Atomic checkpoints and no-repeat resume

**Files:**
- Modify: `src/conclave/evals/live.py`
- Create: `tests/evals/test_live_checkpoint.py`

**Step 1: Write failing checkpoint tests**

Add tests named:

- `test_checkpoint_write_is_flush_fsync_replace_and_secret_scanned`
- `test_checkpoint_rejects_manifest_price_task_or_ceiling_drift`
- `test_resume_charges_pending_reservation_and_never_repeats_interrupted_cell`
- `test_resume_preserves_completed_records_and_call_receipts`
- `test_corrupt_or_tampered_checkpoint_fails_closed`

Simulate an interruption after pending-call persistence with a `BaseException` so normal
provider-error conversion cannot swallow it. On resume, assert the fake provider never sees
the interrupted planned-run ID, the record is `incomplete`, the deviation code is
`interrupted_cell_not_retried`, and the reservation is included in committed cost.

Set a fake provider key in the test environment and assert a checkpoint payload containing
that exact value is rejected before `os.replace`.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_checkpoint.py -q
```

Expected: FAIL because checkpoint load/write/recovery is absent.

**Step 3: Implement atomic state transitions**

Add `LiveCheckpoint`, `ActiveCell`, canonical checkpoint hashing, atomic JSON write, strict
load validation, active-key-value detection, and cell-granular recovery. Persist in the
destination directory, flush, `os.fsync`, then `os.replace`. If persistence fails, propagate
before any next provider call.

**Step 4: Verify GREEN**

Run the Step 2 command plus:

```bash
python -m ruff check src/conclave/evals/live.py tests/evals/test_live_checkpoint.py
python -m ruff format --check src/conclave/evals/live.py tests/evals/test_live_checkpoint.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/conclave/evals/live.py tests/evals/test_live_checkpoint.py
git commit -m "feat(evals): checkpoint live study execution"
```

### Task 5: Complete live study execution and USD 10.00 stop

**Files:**
- Modify: `src/conclave/evals/live.py`
- Modify: `src/conclave/evals/runner.py`
- Create: `tests/evals/test_live_runner.py`

**Step 1: Write failing end-to-end runner tests**

Add tests named:

- `test_live_runner_requires_paid_exploratory_frozen_manifest`
- `test_live_runner_executes_manifest_order_and_builds_complete_study_run`
- `test_budget_exhaustion_makes_no_call_and_marks_all_remaining_cells_incomplete`
- `test_timeout_missing_usage_and_provider_error_never_exceed_ten_dollars`
- `test_final_records_cover_every_planned_run_exactly_once`
- `test_runner_never_loads_grader_keys`

Use the existing `build_study_manifest` with two three-member fictional rosters and an exact
`approved_spend_ceiling_usd=10.0`. Assert `validate_run_records` accepts the final output and
`total_cost_usd <= 10.0` in every parameterized failure path.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_runner.py -q
```

Expected: FAIL because live matrix orchestration is absent.

**Step 3: Implement live matrix orchestration**

Validate the manifest, tasks, snapshot, checkpoint, evidence class, roster size, and exact
USD 10.00 ceiling before configuration or keys. Execute planned cells in frozen order through
the six-condition registry. Aggregate stage receipts into one `ProtocolExecution`; when the
next reservation does not fit, create `budget_exhausted` incomplete records for every
remaining cell without invoking the gateway. Finalize through existing
`validate_run_records` and the same `StudyRun` accounting fields.

**Step 4: Verify GREEN and offline-runner compatibility**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_runner.py tests/evals/test_runner.py -q
python -m ruff check src/conclave/evals/live.py src/conclave/evals/runner.py tests/evals/test_live_runner.py
python -m ruff format --check src/conclave/evals/live.py src/conclave/evals/runner.py tests/evals/test_live_runner.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/conclave/evals/live.py src/conclave/evals/runner.py tests/evals/test_live_runner.py
git commit -m "feat(evals): run capped live study cells"
```

### Task 6: Network-free dry-run estimate

**Files:**
- Modify: `src/conclave/evals/live.py`
- Create: `tests/evals/test_live_estimate.py`

**Step 1: Write failing estimator tests**

Add tests named:

- `test_dry_run_walks_same_calls_without_loading_keys_or_transport`
- `test_dry_run_reports_calls_costs_largest_reservation_and_headroom`
- `test_dry_run_breaks_down_upper_bound_by_roster_and_condition`
- `test_dry_run_rejects_plan_whose_worst_case_exceeds_frozen_ceiling`

Monkeypatch configuration loading, key lookup, and `transport.post_json` to raise if touched.
Compare estimator call counts and stage caps to the protocol tests, not a duplicate table.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_estimate.py -q
```

Expected: FAIL because `estimate_live_study` is absent.

**Step 3: Implement the estimator**

Traverse the same registry and reservation function used by execution. Return a frozen
`LiveStudyEstimate` with planned cells, maximum call count, per-roster/condition upper bounds,
largest reservation, total upper bound, ceiling, headroom, and `fits_ceiling`. Do not import
or call provider configuration from this path.

**Step 4: Verify GREEN**

Run the Step 2 command plus Ruff checks for the changed files. Expected: PASS.

**Step 5: Commit**

```bash
git add src/conclave/evals/live.py tests/evals/test_live_estimate.py
git commit -m "feat(evals): estimate capped live studies"
```

### Task 7: Fail-closed live CLI

**Files:**
- Modify: `src/conclave/eval_cli.py`
- Modify: `tests/evals/test_eval_cli.py`

**Step 1: Write failing CLI tests**

Add tests named:

- `test_eval_live_defaults_to_dry_run_and_never_calls_provider`
- `test_eval_live_requires_execute_and_exact_frozen_spend_approval`
- `test_eval_live_rejects_confirmatory_legacy_or_snapshot_drift`
- `test_eval_live_writes_checkpoint_receipts_and_study_run_atomically`
- `test_eval_live_resume_uses_existing_checkpoint`

The command shape is:

```bash
conclave eval live MANIFEST TASKS PRICE_BOOK OUTPUT CHECKPOINT RECEIPTS \
  --approve-spend-usd 10.00 [--execute]
```

Without `--execute`, print the JSON estimate and exit zero only when it fits. With
`--execute`, require `--approve-spend-usd` to equal both `10.00` and the frozen design value.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_eval_cli.py -q
```

Expected: FAIL because `eval live` is not registered.

**Step 3: Implement the CLI adapter**

Keep CLI behavior thin: load/validate artifacts, call estimator or runner, atomically write
the final run and separate receipt artifact, and render bounded errors. Update the eval app
help so only `live --execute` may reach providers; retain the existing offline `run`
semantics unchanged.

**Step 4: Verify GREEN and CLI regressions**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_eval_cli.py tests/test_cli.py -q
python -m ruff check src/conclave/eval_cli.py tests/evals/test_eval_cli.py
python -m ruff format --check src/conclave/eval_cli.py tests/evals/test_eval_cli.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/conclave/eval_cli.py tests/evals/test_eval_cli.py
git commit -m "feat(cli): gate live evaluation execution"
```

### Task 8: Sanitized transport replay for all conditions

**Files:**
- Create: `tests/evals/test_live_replay.py`
- Create: `tests/fixtures/evals/live_smoke/public_tasks.json`
- Create: `tests/fixtures/evals/live_smoke/manifest.json`
- Create: `tests/fixtures/evals/live_smoke/replay.json`
- Modify: `tests/evals/test_replay.py`

**Step 1: Add the failing replay integration test**

Add `test_live_smoke_replay_executes_all_conditions_with_zero_network_calls`. Load the
committed artifacts, install `ReplayingPostJson` at the existing transport seam, set only a
fake test key, execute the live runner, call `assert_consumed`, and assert:

- all six conditions produced exactly one final cell record;
- call order and counts match the frozen live protocol registry;
- the result and call receipts are byte-stable on a second replay;
- the replay, checkpoint, receipts, and run JSON contain no fake key value;
- an extra, missing, or changed request fails closed and never falls back to network.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider tests/evals/test_live_replay.py tests/evals/test_replay.py -q
```

Expected: FAIL until the sanitized fixtures match the exact request identities.

**Step 3: Record fixtures with fake credentials and review them**

Generate the replay only through `RecordingPostJson` backed by a deterministic fake
delegate. Do not call a provider. Inspect the committed JSON and assert the fake credential
is absent before staging it. Keep all task/model/rate data explicitly fictional.

**Step 4: Verify GREEN**

Run the Step 2 command, Ruff on the tests, and:

```bash
rg -n 'sk-|AIza|Bearer|authorization|x-api-key' tests/fixtures/evals/live_smoke
```

Expected: tests PASS; the scan finds no credential value or header field.

**Step 5: Commit**

```bash
git add tests/evals/test_live_replay.py tests/evals/test_replay.py tests/fixtures/evals/live_smoke
git commit -m "test(evals): replay capped live conditions"
```

### Task 9: Documentation, full verification, and paid-smoke handoff

**Files:**
- Modify: `README.md`
- Modify: `SYSTEM_CONTEXT_DIAGRAM.md`
- Modify: `DOCUMENTATION_INDEX.md`
- Modify: `docs/PRODUCT_DESIGN_DOCUMENT.md`
- Modify: `CHANGELOG.md`

**Step 1: Write the documentation assertions**

Extend the existing eval CLI/QA tests to require these exact boundaries in docs:

- live execution is paid exploratory only;
- dry-run is default and `--execute` plus exact USD 10.00 approval is required;
- one call is in flight, reservations precede calls, and resume never repeats an interrupted
  cell;
- the 24-task fixture remains offline/open-book and is not the paid smoke corpus;
- the smoke checks correctness only, not efficiency or decision quality.

**Step 2: Verify RED**

Run the new documentation assertion test. Expected: FAIL on the current offline-only text.

**Step 3: Update the five existing docs**

Describe the new opt-in live edge honestly, retain the offline `eval run` path, add dry-run
and execute examples using placeholder artifact paths, and keep the PDD and three core docs
under 500 lines. Do not publish real prices, endpoints, task keys, or pilot results.

**Step 4: Run the complete local gate**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin -p no:cacheprovider -q
python -m ruff check .
python -m ruff format --check .
git diff --check
gitleaks git --redact
```

Expected: all tests and static/security checks PASS. If Gitleaks is unavailable, stop and
report the missing verification; do not claim the security gate passed.

**Step 5: Run the network-free operator rehearsal**

Run the CLI dry-run and sanitized replay fixtures. Confirm no provider call occurs, the
estimate fits USD 10.00, the checkpoint/resume path passes, and output labels say paid
exploratory/not decision eligible.

**Step 6: Self-audit the plan boundary before any paid call**

Confirm the branch contains no efficiency analysis, no quality claim, no confirmatory gate
change, no grader-key access, no mutable price service, and no concurrency above one. Fix any
drift before proceeding.

**Step 7: Commit**

```bash
git add README.md SYSTEM_CONTEXT_DIAGRAM.md DOCUMENTATION_INDEX.md docs/PRODUCT_DESIGN_DOCUMENT.md CHANGELOG.md tests
git commit -m "docs(evals): document capped live pilot"
```

**Step 8: Paid smoke execution gate**

Only after Tasks 1-9 are reviewed, run the dry-run against the separately prepared paid-smoke
manifest and price book. Execute the smallest predeclared matrix that exercises all six
conditions under the shared USD 10.00 ceiling. Stop after correctness artifacts are verified;
do not expand calls to improve precision or compare efficiency.

**Step 9: Final handoff evidence**

Provide commit SHA, exact test/lint/format/diff/Gitleaks results, dry-run estimate, charged
cost at or below USD 10.00, checkpoint/resume evidence, sanitized artifact paths, and any
blocked provider/model cells. Do not present exploratory outputs as proof that Elite is
better.
