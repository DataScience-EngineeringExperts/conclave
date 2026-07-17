"""Deterministic, executor-injected runner for offline evaluation studies."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence

from .dataset import hash_public_tasks
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
