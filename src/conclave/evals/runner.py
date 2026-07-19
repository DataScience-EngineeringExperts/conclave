"""Deterministic, executor-injected runner for offline evaluation studies."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path

from conclave.providers import call_model

from .dataset import hash_public_tasks
from .live import (
    BudgetExceededError,
    CallModel,
    CheckpointValidationError,
    GatewayStoppedError,
    LiveCheckpoint,
    LiveProviderClient,
    ProviderCallReceipt,
    ReservationBreachError,
    build_checkpoint_bindings,
    create_live_checkpoint,
    finish_active_cell,
    load_live_checkpoint,
    recover_interrupted_checkpoint,
    start_active_cell,
    update_live_checkpoint,
    write_live_checkpoint,
)
from .live_protocols import (
    LIVE_PROTOCOL_REGISTRY,
    allocate_stage_caps,
    execute_live_condition,
)
from .models import (
    EVAL_CONDITION_IDS,
    ConditionId,
    PlannedRun,
    ProtocolExecution,
    PublicTask,
    RunRecord,
    StudyManifest,
    StudyRun,
)
from .pricing import PriceBook, validate_price_book

LIVE_HARD_CAP_USD = Decimal("10.00")

ProtocolExecutor = Callable[
    ...,
    Awaitable[ProtocolExecution],
]


class RunValidationError(ValueError):
    """Execution inputs or outputs do not exactly match the frozen manifest."""


def validate_run_records(
    manifest: StudyManifest, records: Sequence[RunRecord]
) -> tuple[RunRecord, ...]:
    """Require complete coverage and enforce each cell's execution invariants."""

    planned_ids = [planned.planned_run_id for planned in manifest.planned_runs]
    actual_ids = [record.planned_run_id for record in records]
    if Counter(actual_ids) != Counter(planned_ids):
        raise RunValidationError("run records must cover every planned_run_id exactly once")
    planned_by_id = {planned.planned_run_id: planned for planned in manifest.planned_runs}
    for record in records:
        planned = planned_by_id[record.planned_run_id]
        if (
            record.completion_tokens is not None
            and record.completion_tokens > planned.max_output_tokens
            and not (
                record.outcome == "malformed" and record.error_category == "output_budget_exceeded"
            )
        ):
            raise RunValidationError(
                f"run {record.planned_run_id} exceeds its planned output budget"
            )
        if record.outcome == "success" and record.output is None:
            raise RunValidationError(f"run {record.planned_run_id} is missing its success output")
    return tuple(records)


def _validate_inputs(
    manifest: StudyManifest,
    tasks: Sequence[PublicTask],
    executors: Mapping[ConditionId, ProtocolExecutor],
) -> dict[str, PublicTask]:
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks) or set(task_by_id) != set(manifest.task_ids):
        raise RunValidationError("public tasks must exactly match the manifest task_ids")
    if hash_public_tasks(tasks) != manifest.public_tasks_hash:
        raise RunValidationError("public task content does not match the manifest hash")
    if set(executors) != set(EVAL_CONDITION_IDS):
        raise RunValidationError("executors must contain exactly the six frozen conditions")
    return task_by_id


async def _execute_cell(
    *,
    planned_run: PlannedRun,
    task: PublicTask,
    executor: ProtocolExecutor,
    timeout_seconds: float | None,
) -> RunRecord:
    started = time.perf_counter()
    try:
        invocation = executor(
            task=task,
            planned_run=planned_run,
            max_output_tokens=planned_run.max_output_tokens,
        )
        execution = (
            await asyncio.wait_for(invocation, timeout=timeout_seconds)
            if timeout_seconds is not None
            else await invocation
        )
        if not isinstance(execution, ProtocolExecution):
            return RunRecord(
                planned_run_id=planned_run.planned_run_id,
                outcome="malformed",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_category="invalid_executor_result",
            )
        if execution.completion_tokens is not None and (
            execution.completion_tokens > planned_run.max_output_tokens
        ):
            return RunRecord(
                planned_run_id=planned_run.planned_run_id,
                outcome="malformed",
                completion_tokens=execution.completion_tokens,
                latency_ms=execution.latency_ms,
                error_category="output_budget_exceeded",
                cost_usd=execution.cost_usd,
                cost_receipt_complete=execution.cost_receipt_complete,
                deviation_codes=execution.deviation_codes,
            )
        if execution.outcome == "success" and execution.output is None:
            return RunRecord(
                planned_run_id=planned_run.planned_run_id,
                outcome="malformed",
                completion_tokens=execution.completion_tokens,
                latency_ms=execution.latency_ms,
                error_category="missing_success_output",
                cost_usd=execution.cost_usd,
                cost_receipt_complete=execution.cost_receipt_complete,
                deviation_codes=execution.deviation_codes,
            )
        return RunRecord(
            planned_run_id=planned_run.planned_run_id,
            outcome=execution.outcome,
            output=execution.output,
            completion_tokens=execution.completion_tokens,
            latency_ms=execution.latency_ms,
            error_category=execution.error_category,
            cost_usd=execution.cost_usd,
            cost_receipt_complete=execution.cost_receipt_complete,
            deviation_codes=execution.deviation_codes,
        )
    except TimeoutError:
        return RunRecord(
            planned_run_id=planned_run.planned_run_id,
            outcome="timed_out",
            latency_ms=(time.perf_counter() - started) * 1000,
            error_category="timeout",
        )
    except Exception:
        return RunRecord(
            planned_run_id=planned_run.planned_run_id,
            outcome="failed",
            latency_ms=(time.perf_counter() - started) * 1000,
            error_category="executor_error",
        )


async def run_study(
    *,
    manifest: StudyManifest,
    tasks: Sequence[PublicTask],
    executors: Mapping[ConditionId, ProtocolExecutor],
    timeout_seconds: float | None = None,
) -> StudyRun:
    """Execute every predeclared cell sequentially with no provider dependency."""

    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    task_by_id = _validate_inputs(manifest, tasks, executors)
    records = tuple(
        [
            await _execute_cell(
                planned_run=planned_run,
                task=task_by_id[planned_run.task_id],
                executor=executors[planned_run.condition_id],
                timeout_seconds=timeout_seconds,
            )
            for planned_run in manifest.planned_runs
        ]
    )
    records = validate_run_records(manifest, records)
    outcome_counts = dict(sorted(Counter(record.outcome for record in records).items()))
    return StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(manifest.planned_runs),
        outcome_counts=outcome_counts,
        total_completion_tokens=sum(record.completion_tokens or 0 for record in records),
        total_latency_ms=sum(record.latency_ms or 0.0 for record in records),
        total_cost_usd=sum(record.cost_usd for record in records),
        total_deviation_count=sum(len(record.deviation_codes) for record in records),
    )


def _validate_live_inputs(
    manifest: StudyManifest,
    tasks: Sequence[PublicTask],
    price_book: PriceBook,
) -> tuple[dict[str, PublicTask], dict[str, tuple]]:
    design = manifest.frozen_design
    if manifest.evidence_classification != "paid_exploratory_pilot" or design is None:
        raise RunValidationError("live execution requires a paid exploratory frozen manifest")
    if design.evidence_classification != "paid_exploratory_pilot":
        raise RunValidationError("live execution accepts paid exploratory evidence only")
    if Decimal(str(design.approved_spend_ceiling_usd)) != LIVE_HARD_CAP_USD:
        raise RunValidationError("live execution requires an exact USD 10.00 frozen ceiling")

    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks) or set(task_by_id) != set(manifest.task_ids):
        raise RunValidationError("public tasks must exactly match the manifest task_ids")
    if hash_public_tasks(tasks) != manifest.public_tasks_hash:
        raise RunValidationError("public task content does not match the manifest hash")

    roster_by_id = {roster.roster_id: roster.members for roster in design.rosters}
    if len(roster_by_id) != len(design.rosters):
        raise RunValidationError("frozen live roster IDs must be unique")
    if any(len(members) < 3 for members in roster_by_id.values()):
        raise RunValidationError("paid live rosters require at least three members")

    expected_prompt_versions = {
        condition: spec.prompt_version for condition, spec in LIVE_PROTOCOL_REGISTRY.items()
    }
    expected_protocol_versions = {
        condition: spec.protocol_version for condition, spec in LIVE_PROTOCOL_REGISTRY.items()
    }
    if design.condition_prompt_versions != expected_prompt_versions:
        raise RunValidationError("frozen live prompt versions do not match the runner")
    if design.condition_protocol_versions != expected_protocol_versions:
        raise RunValidationError("frozen live protocol versions do not match the runner")

    for planned_run in manifest.planned_runs:
        members = roster_by_id.get(planned_run.roster_id)
        if members is None:
            raise RunValidationError("planned run references an unknown frozen roster")
        try:
            allocate_stage_caps(
                planned_run.condition_id,
                roster_size=len(members),
                cell_ceiling=planned_run.max_output_tokens,
            )
        except (TypeError, ValueError) as exc:
            raise RunValidationError("planned run has an invalid live stage allocation") from exc

    try:
        validate_price_book(price_book, frozen_design=design)
    except ValueError as exc:
        raise RunValidationError(str(exc)) from exc
    return task_by_id, roster_by_id


def _load_or_create_live_checkpoint(
    checkpoint_path: Path,
    *,
    manifest: StudyManifest,
    price_book: PriceBook,
) -> LiveCheckpoint:
    bindings = build_checkpoint_bindings(
        manifest,
        price_book,
        hard_cap_usd=LIVE_HARD_CAP_USD,
    )
    if checkpoint_path.exists():
        checkpoint = load_live_checkpoint(checkpoint_path, expected_bindings=bindings)
    else:
        checkpoint = create_live_checkpoint(bindings=bindings)
        write_live_checkpoint(checkpoint_path, checkpoint)

    planned_ids = {planned_run.planned_run_id for planned_run in manifest.planned_runs}
    record_ids = {record.planned_run_id for record in checkpoint.records}
    active_id = checkpoint.active_cell.planned_run_id if checkpoint.active_cell else None
    if not record_ids.issubset(planned_ids) or (
        active_id is not None and active_id not in planned_ids
    ):
        raise CheckpointValidationError("checkpoint contains an unplanned live cell")
    if checkpoint.active_cell is not None:
        checkpoint = recover_interrupted_checkpoint(checkpoint)
        write_live_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def _cell_receipts(
    checkpoint: LiveCheckpoint, *, receipt_start_index: int
) -> tuple[ProviderCallReceipt, ...]:
    return checkpoint.receipts[receipt_start_index:]


def _aggregate_live_execution(
    receipts: Sequence[ProviderCallReceipt],
    *,
    latency_ms: float,
    output: str | None = None,
    error_category: str | None = None,
    outcome: str | None = None,
    deviation_codes: tuple[str, ...] = (),
) -> ProtocolExecution:
    failed = tuple(receipt for receipt in receipts if receipt.outcome == "failed")
    resolved_error = error_category or (failed[-1].error_category if failed else None)
    if outcome is None:
        outcome = "timed_out" if resolved_error == "timeout" else "failed"
    if output is not None:
        outcome = "success"
    completion_tokens = (
        sum(receipt.usage.completion_tokens for receipt in receipts if receipt.usage is not None)
        if receipts and not failed and all(receipt.usage is not None for receipt in receipts)
        else None
    )
    receipt_deviations = tuple(
        dict.fromkeys(
            f"provider_{receipt.error_category}"
            for receipt in failed
            if receipt.error_category is not None
        )
    )
    cost = sum((receipt.charged_cost_usd for receipt in receipts), Decimal("0"))
    return ProtocolExecution(
        outcome=outcome,
        output=output,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        error_category=None if output is not None else resolved_error,
        cost_usd=float(cost),
        cost_receipt_complete=True,
        deviation_codes=(*receipt_deviations, *deviation_codes),
    )


def _record_from_execution(planned_run_id: str, execution: ProtocolExecution) -> RunRecord:
    return RunRecord(
        planned_run_id=planned_run_id,
        outcome=execution.outcome,
        output=execution.output,
        completion_tokens=execution.completion_tokens,
        latency_ms=execution.latency_ms,
        error_category=execution.error_category,
        cost_usd=execution.cost_usd,
        cost_receipt_complete=execution.cost_receipt_complete,
        deviation_codes=execution.deviation_codes,
    )


def _materialize_remaining(
    checkpoint: LiveCheckpoint,
    planned_runs: Sequence[PlannedRun],
    *,
    error_category: str,
) -> LiveCheckpoint:
    recorded = {record.planned_run_id for record in checkpoint.records}
    additions = tuple(
        RunRecord(
            planned_run_id=planned_run.planned_run_id,
            outcome="incomplete",
            error_category=error_category,
            cost_receipt_complete=True,
            deviation_codes=(error_category,),
        )
        for planned_run in planned_runs
        if planned_run.planned_run_id not in recorded
    )
    return create_live_checkpoint(
        bindings=checkpoint.bindings,
        records=(*checkpoint.records, *additions),
        receipts=checkpoint.receipts,
        committed_cost_usd=checkpoint.committed_cost_usd,
    )


def _build_live_study_run(manifest: StudyManifest, checkpoint: LiveCheckpoint) -> StudyRun:
    by_id = {record.planned_run_id: record for record in checkpoint.records}
    ordered = tuple(by_id[planned.planned_run_id] for planned in manifest.planned_runs)
    records = validate_run_records(manifest, ordered)
    return StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(manifest.planned_runs),
        outcome_counts=dict(sorted(Counter(record.outcome for record in records).items())),
        total_completion_tokens=sum(record.completion_tokens or 0 for record in records),
        total_latency_ms=sum(record.latency_ms or 0.0 for record in records),
        total_cost_usd=float(checkpoint.committed_cost_usd),
        total_deviation_count=sum(len(record.deviation_codes) for record in records),
    )


async def run_live_study(
    *,
    manifest: StudyManifest,
    tasks: Sequence[PublicTask],
    price_book: PriceBook,
    checkpoint_path: str | Path,
    call_model_func: CallModel = call_model,
) -> StudyRun:
    """Execute a frozen paid-exploratory matrix under the exact USD 10.00 cap."""

    task_by_id, roster_by_id = _validate_live_inputs(manifest, tasks, price_book)
    path = Path(checkpoint_path)
    checkpoint = _load_or_create_live_checkpoint(
        path,
        manifest=manifest,
        price_book=price_book,
    )

    def persist(
        pending_call,
        receipts: tuple[ProviderCallReceipt, ...],
    ) -> None:
        nonlocal checkpoint
        checkpoint = update_live_checkpoint(
            checkpoint,
            pending_call=pending_call,
            receipts=receipts,
        )
        write_live_checkpoint(path, checkpoint)

    client = LiveProviderClient(
        price_book=price_book,
        hard_cap_usd=LIVE_HARD_CAP_USD,
        checkpoint=persist,
        call_model_func=call_model_func,
        timeout=manifest.frozen_design.timeout_retry_policy.timeout_seconds,
        resume_from=checkpoint,
    )

    planned_runs = manifest.planned_runs
    for index, planned_run in enumerate(planned_runs):
        if not checkpoint.should_execute(planned_run.planned_run_id):
            continue
        checkpoint = start_active_cell(
            checkpoint,
            planned_run_id=planned_run.planned_run_id,
        )
        write_live_checkpoint(path, checkpoint)
        receipt_start_index = checkpoint.active_cell.receipt_start_index
        started = time.perf_counter()
        try:
            output = await execute_live_condition(
                planned_run.condition_id,
                task=task_by_id[planned_run.task_id],
                roster=roster_by_id[planned_run.roster_id],
                cell_ceiling=planned_run.max_output_tokens,
                client=client,
            )
            execution = _aggregate_live_execution(
                _cell_receipts(checkpoint, receipt_start_index=receipt_start_index),
                latency_ms=(time.perf_counter() - started) * 1000,
                output=output,
            )
        except BudgetExceededError:
            execution = _aggregate_live_execution(
                _cell_receipts(checkpoint, receipt_start_index=receipt_start_index),
                latency_ms=(time.perf_counter() - started) * 1000,
                error_category="budget_exhausted",
                outcome="incomplete",
                deviation_codes=("budget_exhausted",),
            )
            checkpoint = finish_active_cell(
                checkpoint,
                record=_record_from_execution(planned_run.planned_run_id, execution),
            )
            checkpoint = _materialize_remaining(
                checkpoint,
                planned_runs[index + 1 :],
                error_category="budget_exhausted",
            )
            write_live_checkpoint(path, checkpoint)
            break
        except (ReservationBreachError, GatewayStoppedError) as exc:
            category = (
                "reservation_breach"
                if isinstance(exc, ReservationBreachError)
                else "gateway_stopped"
            )
            execution = _aggregate_live_execution(
                _cell_receipts(checkpoint, receipt_start_index=receipt_start_index),
                latency_ms=(time.perf_counter() - started) * 1000,
                error_category=category,
                deviation_codes=(category,),
            )
            checkpoint = finish_active_cell(
                checkpoint,
                record=_record_from_execution(planned_run.planned_run_id, execution),
            )
            checkpoint = _materialize_remaining(
                checkpoint,
                planned_runs[index + 1 :],
                error_category="gateway_stopped",
            )
            write_live_checkpoint(path, checkpoint)
            break
        except Exception:  # noqa: BLE001 -- bounded receipts determine the public failure
            receipts = _cell_receipts(checkpoint, receipt_start_index=receipt_start_index)
            execution = _aggregate_live_execution(
                receipts,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_category=(
                    None
                    if any(receipt.outcome == "failed" for receipt in receipts)
                    else "protocol_error"
                ),
            )

        checkpoint = finish_active_cell(
            checkpoint,
            record=_record_from_execution(planned_run.planned_run_id, execution),
        )
        write_live_checkpoint(path, checkpoint)

    return _build_live_study_run(manifest, checkpoint)
