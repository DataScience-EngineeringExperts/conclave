from __future__ import annotations

import asyncio
import json
import multiprocessing
from collections import Counter
from decimal import Decimal

import pytest

import conclave.evals.dataset as dataset_module
import conclave.evals.live as live_module
import conclave.evals.runner as runner_module
from conclave.evals.live import (
    CheckpointValidationError,
    build_checkpoint_bindings,
)
from conclave.evals.live_protocols import LIVE_PROTOCOL_REGISTRY, stage_call_sequence
from conclave.evals.models import (
    AnalysisGateConfig,
    BootstrapConfig,
    ExclusionDeviationPolicy,
    FrozenStudyDesign,
    PriceSnapshot,
    ProviderModelSpec,
    PublicTask,
    RandomizationConfig,
    RosterSpec,
    TimeoutRetryPolicy,
)
from conclave.evals.pricing import ModelPrice, PriceBook, hash_price_entries
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.runner import RunValidationError, validate_run_records
from conclave.models import ModelAnswer, TokenUsage

DIGEST = "sha256:" + "a" * 64
HARD_CAP = Decimal("10.00")
LIVE_FIXTURE_CELL_CEILING = 6144
SEAL_KEY = bytes(range(32))


def create_live_checkpoint(*args, **kwargs):
    kwargs.setdefault("seal_key", SEAL_KEY)
    return live_module.create_live_checkpoint(*args, **kwargs)


def write_live_checkpoint(*args, **kwargs):
    kwargs.setdefault("seal_key", SEAL_KEY)
    return live_module.write_live_checkpoint(*args, **kwargs)


def load_live_checkpoint(*args, **kwargs):
    kwargs.setdefault("seal_key", SEAL_KEY)
    return live_module.load_live_checkpoint(*args, **kwargs)


async def run_live_study(*args, **kwargs):
    kwargs.setdefault("checkpoint_seal_key", SEAL_KEY)
    return await runner_module.run_live_study(*args, **kwargs)


def _valid_verdict_extraction_text() -> str:
    return json.dumps(
        {
            "verdict_applies": True,
            "verdict_type": "decision",
            "headline": "Choose the fictional safe option.",
            "recommendation": "Use the fictional safe option with safeguards.",
            "positions": [
                {
                    "label": "safe-option",
                    "summary": "The fictional safe option is preferred.",
                    "providers": ["Model A", "Model B", "Model C"],
                    "evidence_answer_ids": ["fixture-a", "fixture-b", "fixture-c"],
                }
            ],
            "provider_votes": [
                {"provider": "Model A", "position_label": "safe-option"},
                {"provider": "Model B", "position_label": "safe-option"},
                {"provider": "Model C", "position_label": "safe-option"},
            ],
            "conflicts": [],
            "minority_reports": [],
            "caveats": [],
        }
    )


def _rosters(*, members_per_roster: int = 3) -> tuple[RosterSpec, ...]:
    rosters = []
    member_number = 0
    for roster_name in ("a", "b"):
        members = []
        for _ in range(members_per_roster):
            member_number += 1
            members.append(
                ProviderModelSpec(
                    provider_id=f"fictional-provider-{member_number}",
                    model_id=f"fictional/model-{member_number}",
                    model_revision=f"fixture-r{member_number}",
                )
            )
        rosters.append(RosterSpec(roster_id=f"fictional-roster-{roster_name}", members=members))
    return tuple(rosters)


def _live_inputs(
    *,
    rate: str = "1",
    members_per_roster: int = 3,
    evidence_classification: str = "paid_exploratory_pilot",
    approved_spend_ceiling_usd: float = 10.0,
    version_drift: bool = False,
    cell_ceiling: int = LIVE_FIXTURE_CELL_CEILING,
):
    tasks = [
        PublicTask(
            task_id="fictional-task",
            prompt="Choose the safest fictional option.",
            reference_packets=("All facts in this packet are fictional.",),
        )
    ]
    rosters = _rosters(members_per_roster=members_per_roster)
    entries = tuple(
        ModelPrice(
            provider_id=member.provider_id,
            model_id=member.model_id,
            model_revision=member.model_revision,
            input_ceiling_usd_per_million_tokens=Decimal(rate),
            output_ceiling_usd_per_million_tokens=Decimal(rate),
            max_output_bytes_per_token=4,
        )
        for roster in rosters
        for member in roster.members
    )
    price_book = PriceBook(
        snapshot_id="fictional-live-runner-prices",
        captured_at="2026-07-18T12:00:00Z",
        currency="USD",
        entries=entries,
    )
    prompt_versions = {
        condition: spec.prompt_version for condition, spec in LIVE_PROTOCOL_REGISTRY.items()
    }
    if version_drift:
        prompt_versions["single_frontier"] = "drifted-prompt-version"
    design = FrozenStudyDesign(
        evidence_classification=evidence_classification,
        base_commit="1" * 40,
        task_family_map={"fictional-task": "fixture-family"},
        rosters=rosters,
        condition_prompt_versions=prompt_versions,
        condition_protocol_versions={
            condition: spec.protocol_version for condition, spec in LIVE_PROTOCOL_REGISTRY.items()
        },
        generation_settings_hash=DIGEST,
        evaluator_version="fictional-evaluator-v1",
        analysis_code_hash=DIGEST,
        rubric_hash=DIGEST,
        grader_instructions_hash=DIGEST,
        grader_keys_hash=DIGEST,
        exclusion_deviation_policy=ExclusionDeviationPolicy(),
        timeout_retry_policy=TimeoutRetryPolicy(timeout_seconds=1, retry_attempts=0),
        randomization=RandomizationConfig(master_seed=20260718),
        bootstrap=BootstrapConfig(seed=20260718, samples=10),
        analysis_gates=AnalysisGateConfig(
            primary_baseline="single_frontier",
            absolute_p95_latency_seconds=60,
            minimum_confirmatory_tasks=2,
        ),
        price_snapshot=PriceSnapshot(
            snapshot_id=price_book.snapshot_id,
            captured_at=price_book.captured_at,
            currency=price_book.currency,
            prices_hash=hash_price_entries(price_book.entries),
        ),
        approved_spend_ceiling_usd=approved_spend_ceiling_usd,
        preregistration_id=(
            "fixture:confirmatory" if evidence_classification == "confirmatory" else None
        ),
        preregistration_hash=(DIGEST if evidence_classification == "confirmatory" else None),
    )
    manifest = build_study_manifest(
        study_id="fictional-paid-pilot",
        tasks=tasks,
        replicates=1,
        seed=20260718,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, cell_ceiling),
        frozen_design=design,
    )
    return tasks, manifest, price_book


async def _successful_provider(name, model_id, messages, **kwargs):
    del messages
    return ModelAnswer(
        name=name,
        model_id=model_id,
        answer=(
            _valid_verdict_extraction_text()
            if kwargs.get("output_contract") is not None
            else f"fictional decision from {name}"
        ),
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _run_live_study_process(
    manifest,
    tasks,
    price_book,
    checkpoint_path,
    start_event,
    lease_held_event,
    release_event,
    results,
) -> None:
    calls = 0

    async def provider(name, model_id, messages, **kwargs):
        nonlocal calls
        del messages, kwargs
        calls += 1
        if calls == 1:
            lease_held_event.set()
            if not release_event.wait(timeout=10):
                raise RuntimeError("fixture lease release timed out")
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer=f"fixture answer from {name}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    if not start_event.wait(timeout=10):
        results.put(("start_timeout", calls))
        return
    try:
        asyncio.run(
            run_live_study(
                manifest=manifest,
                tasks=tasks,
                price_book=price_book,
                checkpoint_path=checkpoint_path,
                call_model_func=provider,
            )
        )
    except CheckpointValidationError:
        results.put(("lease_rejected", calls))
    except BaseException as exc:  # noqa: BLE001 - child reports bounded test status
        results.put((f"unexpected:{type(exc).__name__}", calls))
    else:
        results.put(("completed", calls))


@pytest.mark.asyncio
async def test_live_runner_requires_paid_exploratory_frozen_manifest(tmp_path) -> None:
    provider_calls = 0

    async def forbidden_provider(*args, **kwargs):
        nonlocal provider_calls
        del args, kwargs
        provider_calls += 1
        raise AssertionError("validation must finish before provider configuration or keys")

    invalid_cases = []
    legacy_tasks = [PublicTask(task_id="legacy-task", prompt="Legacy")]
    legacy_manifest = build_study_manifest(
        study_id="legacy",
        tasks=legacy_tasks,
        replicates=1,
        seed=1,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 64),
    )
    _, _, valid_book = _live_inputs()
    invalid_cases.append((legacy_tasks, legacy_manifest, valid_book))
    invalid_cases.append(_live_inputs(evidence_classification="confirmatory"))
    invalid_cases.append(_live_inputs(approved_spend_ceiling_usd=9.99))
    invalid_cases.append(_live_inputs(members_per_roster=2))
    invalid_cases.append(_live_inputs(version_drift=True))
    invalid_cases.append(_live_inputs(cell_ceiling=1024))

    for index, (tasks, manifest, price_book) in enumerate(invalid_cases):
        with pytest.raises(RunValidationError):
            await run_live_study(
                manifest=manifest,
                tasks=tasks,
                price_book=price_book,
                checkpoint_path=tmp_path / f"invalid-{index}.json",
                call_model_func=forbidden_provider,
            )

    tasks, manifest, price_book = _live_inputs()
    drifted_book = price_book.model_copy(update={"snapshot_id": "drifted-snapshot"})
    with pytest.raises(RunValidationError, match="snapshot"):
        await run_live_study(
            manifest=manifest,
            tasks=tasks,
            price_book=drifted_book,
            checkpoint_path=tmp_path / "snapshot-drift.json",
            call_model_func=forbidden_provider,
        )

    changed_tasks = [tasks[0].model_copy(update={"prompt": "Changed after freezing."})]
    with pytest.raises(RunValidationError, match="task"):
        await run_live_study(
            manifest=manifest,
            tasks=changed_tasks,
            price_book=price_book,
            checkpoint_path=tmp_path / "task-drift.json",
            call_model_func=forbidden_provider,
        )

    checkpoint_path = tmp_path / "checkpoint-drift.json"
    bindings = build_checkpoint_bindings(manifest, price_book, hard_cap_usd=HARD_CAP)
    drifted_bindings = bindings.model_copy(update={"manifest_hash": DIGEST})
    write_live_checkpoint(
        checkpoint_path,
        create_live_checkpoint(bindings=drifted_bindings),
    )
    with pytest.raises(CheckpointValidationError, match="manifest_hash"):
        await run_live_study(
            manifest=manifest,
            tasks=tasks,
            price_book=price_book,
            checkpoint_path=checkpoint_path,
            call_model_func=forbidden_provider,
        )

    assert provider_calls == 0


@pytest.mark.asyncio
async def test_live_runner_executes_manifest_order_and_builds_complete_study_run(
    tmp_path,
) -> None:
    tasks, manifest, price_book = _live_inputs()
    checkpoint_path = tmp_path / "live-checkpoint.json"
    calls: list[tuple[str, str, str, int]] = []

    async def recording_provider(name, model_id, messages, **kwargs):
        del messages
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        active = payload["active_cell"]
        assert active["pending_call"] is not None
        calls.append((active["planned_run_id"], name, model_id, kwargs["max_output_tokens"]))
        return await _successful_provider(name, model_id, (), **kwargs)

    study_run = await run_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        checkpoint_path=checkpoint_path,
        call_model_func=recording_provider,
    )

    expected_ids = tuple(run.planned_run_id for run in manifest.planned_runs)
    first_seen_ids = tuple(dict.fromkeys(planned_run_id for planned_run_id, *_ in calls))
    call_counts = Counter(planned_run_id for planned_run_id, *_ in calls)

    assert first_seen_ids == expected_ids
    assert call_counts == {
        run.planned_run_id: len(stage_call_sequence(run.condition_id, roster_size=3))
        - (1 if run.condition_id == "elite_full" else 0)
        for run in manifest.planned_runs
    }
    assert tuple(record.planned_run_id for record in study_run.records) == expected_ids
    assert study_run.outcome_counts == {"success": len(manifest.planned_runs)}
    assert validate_run_records(manifest, study_run.records) == study_run.records
    assert study_run.total_cost_usd <= 10.0
    assert (
        load_live_checkpoint(
            checkpoint_path,
            expected_bindings=build_checkpoint_bindings(
                manifest, price_book, hard_cap_usd=HARD_CAP
            ),
        ).active_cell
        is None
    )


def test_live_runner_allows_only_one_process_per_checkpoint(tmp_path) -> None:
    context = multiprocessing.get_context("fork")
    tasks, manifest, price_book = _live_inputs()
    checkpoint_path = tmp_path / "leased-checkpoint.json"
    start_event = context.Event()
    lease_held_event = context.Event()
    release_event = context.Event()
    results = context.Queue()
    args = (
        manifest,
        tasks,
        price_book,
        checkpoint_path,
        start_event,
        lease_held_event,
        release_event,
        results,
    )
    processes = [context.Process(target=_run_live_study_process, args=args) for _ in range(2)]
    for process in processes:
        process.start()

    try:
        start_event.set()
        assert lease_held_event.wait(timeout=10)
        rejected = results.get(timeout=10)
        assert rejected == ("lease_rejected", 0)
        release_event.set()
        completed = results.get(timeout=20)
        assert completed[0] == "completed"
        assert completed[1] > 0
    finally:
        release_event.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)


def test_live_runner_rejects_symlinked_lifecycle_lock(tmp_path) -> None:
    checkpoint_path = tmp_path / "symlinked-checkpoint.json"
    checkpoint_path.write_text("{}", encoding="utf-8")
    lock_path = checkpoint_path.with_name(f"{checkpoint_path.name}.lock")
    lock_path.symlink_to(checkpoint_path)

    with pytest.raises(CheckpointValidationError, match="lease"):
        with runner_module._checkpoint_lifecycle_lease(checkpoint_path):
            pass


@pytest.mark.asyncio
async def test_live_runner_fails_closed_when_filesystem_leases_are_unsupported(
    tmp_path, monkeypatch
) -> None:
    tasks, manifest, price_book = _live_inputs()
    provider_calls = 0

    async def forbidden_provider(*args, **kwargs):
        nonlocal provider_calls
        del args, kwargs
        provider_calls += 1
        raise AssertionError("unsupported lease platform must fail before provider access")

    monkeypatch.setattr(runner_module, "_fcntl", None, raising=False)
    with pytest.raises(CheckpointValidationError, match="unsupported"):
        await run_live_study(
            manifest=manifest,
            tasks=tasks,
            price_book=price_book,
            checkpoint_path=tmp_path / "unsupported-lease.json",
            call_model_func=forbidden_provider,
        )
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_budget_exhaustion_makes_no_call_and_marks_all_remaining_cells_incomplete(
    tmp_path,
) -> None:
    tasks, manifest, price_book = _live_inputs(rate="100000000")
    provider_calls = 0

    async def forbidden_provider(*args, **kwargs):
        nonlocal provider_calls
        del args, kwargs
        provider_calls += 1
        raise AssertionError("a rejected reservation must not call the provider")

    study_run = await run_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        checkpoint_path=tmp_path / "budget-checkpoint.json",
        call_model_func=forbidden_provider,
    )

    assert provider_calls == 0
    assert len(study_run.records) == len(manifest.planned_runs)
    assert all(record.outcome == "incomplete" for record in study_run.records)
    assert all(record.error_category == "budget_exhausted" for record in study_run.records)
    assert all(record.deviation_codes == ("budget_exhausted",) for record in study_run.records)
    assert study_run.total_cost_usd == 0.0
    assert validate_run_records(manifest, study_run.records) == study_run.records


@pytest.mark.parametrize(
    "failure_mode",
    ("timeout", "missing_usage", "provider_error", "reservation_breach"),
)
@pytest.mark.asyncio
async def test_timeout_missing_usage_and_provider_error_never_exceed_ten_dollars(
    tmp_path,
    failure_mode,
) -> None:
    tasks, manifest, price_book = _live_inputs(rate="10")

    async def provider(name, model_id, messages, **kwargs):
        del messages
        if failure_mode == "timeout":
            raise TimeoutError("fictional provider timed out")
        if failure_mode == "missing_usage":
            return ModelAnswer(
                name=name,
                model_id=model_id,
                answer=(
                    _valid_verdict_extraction_text()
                    if kwargs.get("output_contract") is not None
                    else "unmetered fixture answer"
                ),
            )
        if failure_mode == "provider_error":
            return ModelAnswer(name=name, model_id=model_id, error="fictional provider error")
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="over reservation fixture answer",
            usage=TokenUsage(
                prompt_tokens=1,
                completion_tokens=kwargs["max_output_tokens"] + 1,
                total_tokens=kwargs["max_output_tokens"] + 2,
            ),
        )

    checkpoint_path = tmp_path / f"{failure_mode}.json"
    study_run = await run_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        checkpoint_path=checkpoint_path,
        call_model_func=provider,
    )
    checkpoint = load_live_checkpoint(
        checkpoint_path,
        expected_bindings=build_checkpoint_bindings(manifest, price_book, hard_cap_usd=HARD_CAP),
    )

    assert study_run.total_cost_usd <= 10.0
    assert checkpoint.committed_cost_usd <= HARD_CAP
    assert sum((receipt.charged_cost_usd for receipt in checkpoint.receipts), Decimal()) <= HARD_CAP
    assert validate_run_records(manifest, study_run.records) == study_run.records
    assert "fictional provider timed out" not in checkpoint.model_dump_json()
    if failure_mode == "timeout":
        assert any(record.outcome == "timed_out" for record in study_run.records)
        assert all(record.error_category == "timeout" for record in study_run.records)
    elif failure_mode == "missing_usage":
        assert all(record.outcome == "success" for record in study_run.records)
    elif failure_mode == "provider_error":
        assert any(record.outcome == "failed" for record in study_run.records)
        assert all(record.error_category == "provider_error" for record in study_run.records)
    else:
        assert study_run.records[0].error_category == "reservation_breach"
        assert all(record.outcome == "incomplete" for record in study_run.records[1:])


@pytest.mark.asyncio
async def test_final_records_cover_every_planned_run_exactly_once(tmp_path) -> None:
    tasks, manifest, price_book = _live_inputs(rate="10")
    calls = 0

    async def provider(name, model_id, messages, **kwargs):
        nonlocal calls
        del messages, kwargs
        calls += 1
        if calls == 4:
            return ModelAnswer(name=name, model_id=model_id, error="bounded fixture failure")
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer=f"fixture answer {calls}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    study_run = await run_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        checkpoint_path=tmp_path / "coverage.json",
        call_model_func=provider,
    )

    expected = Counter(run.planned_run_id for run in manifest.planned_runs)
    actual = Counter(record.planned_run_id for record in study_run.records)
    assert actual == expected
    assert all(count == 1 for count in actual.values())
    assert validate_run_records(manifest, study_run.records) == study_run.records


@pytest.mark.asyncio
async def test_runner_never_loads_grader_keys(tmp_path, monkeypatch) -> None:
    tasks, manifest, price_book = _live_inputs()

    def forbidden_grader_key_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the live execution boundary must never load grader keys")

    monkeypatch.setattr(dataset_module, "load_grader_keys", forbidden_grader_key_load)

    study_run = await run_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        checkpoint_path=tmp_path / "no-grader-keys.json",
        call_model_func=_successful_provider,
    )

    assert len(study_run.records) == len(manifest.planned_runs)
    assert validate_run_records(manifest, study_run.records) == study_run.records
