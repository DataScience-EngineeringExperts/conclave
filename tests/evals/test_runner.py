from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from conclave.evals.models import ProtocolExecution, PublicTask, RunRecord
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.runner import RunValidationError, run_study, validate_run_records


def _study():
    tasks = [
        PublicTask(task_id="task-a", prompt="Choose A."),
        PublicTask(task_id="task-b", prompt="Choose B."),
    ]
    manifest = build_study_manifest(
        study_id="offline-study",
        tasks=tasks,
        replicates=2,
        seed=17,
        output_token_budgets={condition_id: 400 for condition_id in CONDITION_IDS},
    )
    return tasks, manifest


@pytest.mark.asyncio
async def test_runner_executes_complete_matrix_with_cell_budget_and_immutable_records() -> None:
    tasks, manifest = _study()
    calls: list[tuple[str, str, int]] = []

    async def executor(*, task, planned_run, max_output_tokens):
        calls.append((task.task_id, planned_run.condition_id, max_output_tokens))
        return ProtocolExecution(
            outcome="success",
            output=f"{task.task_id}:{planned_run.condition_id}",
            completion_tokens=11,
            latency_ms=2.5,
            cost_usd=0.01,
            cost_receipt_complete=True,
            deviation_codes=("retry_used",),
        )

    study_run = await run_study(
        manifest=manifest,
        tasks=tasks,
        executors={condition_id: executor for condition_id in CONDITION_IDS},
    )

    assert len(calls) == len(manifest.planned_runs) == 24
    assert {condition_id for _, condition_id, _ in calls} == set(CONDITION_IDS)
    assert {budget for _, _, budget in calls} == {400}
    assert tuple(record.planned_run_id for record in study_run.records) == tuple(
        run.planned_run_id for run in manifest.planned_runs
    )
    assert study_run.total_planned_runs == 24
    assert study_run.outcome_counts == {"success": 24}
    assert study_run.total_completion_tokens == 264
    assert study_run.total_latency_ms == 60.0
    assert study_run.total_cost_usd == pytest.approx(0.24)
    assert study_run.total_deviation_count == 24
    assert study_run.records[0].deviation_codes == ("retry_used",)
    assert study_run.records[0].cost_receipt_complete is True
    with pytest.raises(ValidationError):
        study_run.records[0].output = "changed"


@pytest.mark.asyncio
async def test_runner_preserves_every_non_success_outcome_in_denominator() -> None:
    tasks = [PublicTask(task_id="task-a", prompt="Choose A.")]
    manifest = build_study_manifest(
        study_id="failure-study",
        tasks=tasks,
        replicates=1,
        seed=2,
        output_token_budgets={condition_id: 100 for condition_id in CONDITION_IDS},
    )
    outcomes = dict(
        zip(
            CONDITION_IDS,
            ("failed", "timed_out", "abstained", "malformed", "incomplete", "success"),
            strict=True,
        )
    )

    async def executor(*, task, planned_run, max_output_tokens):
        del task, max_output_tokens
        outcome = outcomes[planned_run.condition_id]
        return ProtocolExecution(
            outcome=outcome,
            output="answer" if outcome == "success" else None,
            completion_tokens=3 if outcome in {"success", "incomplete"} else None,
            latency_ms=4.0,
            error_category=None if outcome in {"success", "abstained"} else outcome,
        )

    study_run = await run_study(
        manifest=manifest,
        tasks=tasks,
        executors={condition_id: executor for condition_id in CONDITION_IDS},
    )

    assert study_run.total_planned_runs == 6
    assert sum(study_run.outcome_counts.values()) == 6
    assert study_run.outcome_counts == {
        "abstained": 1,
        "failed": 1,
        "incomplete": 1,
        "malformed": 1,
        "success": 1,
        "timed_out": 1,
    }
    assert study_run.total_completion_tokens == 6
    assert study_run.total_latency_ms == 24.0


@pytest.mark.asyncio
async def test_runner_fails_closed_when_executor_exceeds_budget_or_returns_empty_success() -> None:
    tasks = [PublicTask(task_id="task-a", prompt="Choose A.")]
    manifest = build_study_manifest(
        study_id="budget-study",
        tasks=tasks,
        replicates=1,
        seed=4,
        output_token_budgets={condition_id: 100 for condition_id in CONDITION_IDS},
    )

    async def executor(*, task, planned_run, max_output_tokens):
        del task, max_output_tokens
        if planned_run.condition_id == "single_frontier":
            return ProtocolExecution(outcome="success", output="too long", completion_tokens=101)
        if planned_run.condition_id == "self_refine":
            return ProtocolExecution(outcome="success", output=None, completion_tokens=4)
        return ProtocolExecution(outcome="success", output="answer", completion_tokens=4)

    study_run = await run_study(
        manifest=manifest,
        tasks=tasks,
        executors={condition_id: executor for condition_id in CONDITION_IDS},
    )
    by_condition = {
        planned.condition_id: record
        for planned, record in zip(manifest.planned_runs, study_run.records, strict=True)
    }

    assert by_condition["single_frontier"].outcome == "malformed"
    assert by_condition["single_frontier"].error_category == "output_budget_exceeded"
    assert by_condition["single_frontier"].completion_tokens == 101
    assert by_condition["self_refine"].outcome == "malformed"
    assert by_condition["self_refine"].error_category == "missing_success_output"
    assert study_run.total_planned_runs == 6
    assert sum(study_run.outcome_counts.values()) == 6
    assert study_run.total_completion_tokens == 121


@pytest.mark.asyncio
async def test_runner_converts_executor_errors_and_timeouts_to_records() -> None:
    tasks = [PublicTask(task_id="task-a", prompt="Choose A.")]
    manifest = build_study_manifest(
        study_id="exception-study",
        tasks=tasks,
        replicates=1,
        seed=3,
        output_token_budgets={condition_id: 100 for condition_id in CONDITION_IDS},
    )

    async def executor(*, task, planned_run, max_output_tokens):
        del task, max_output_tokens
        if planned_run.condition_id == "single_frontier":
            raise RuntimeError("provider detail must not escape")
        if planned_run.condition_id == "self_refine":
            await asyncio.sleep(0.05)
        return ProtocolExecution(outcome="success", output="answer")

    study_run = await run_study(
        manifest=manifest,
        tasks=tasks,
        executors={condition_id: executor for condition_id in CONDITION_IDS},
        timeout_seconds=0.001,
    )

    by_condition = {
        planned.condition_id: record
        for planned, record in zip(manifest.planned_runs, study_run.records, strict=True)
    }
    assert by_condition["single_frontier"].outcome == "failed"
    assert by_condition["single_frontier"].error_category == "executor_error"
    assert by_condition["self_refine"].outcome == "timed_out"
    assert by_condition["self_refine"].error_category == "timeout"
    assert "provider detail" not in study_run.model_dump_json()


def test_record_validation_fails_closed_on_missing_duplicate_or_unplanned_output() -> None:
    _, manifest = _study()
    valid = tuple(
        RunRecord(planned_run_id=run.planned_run_id, outcome="success", output="answer")
        for run in manifest.planned_runs
    )

    assert validate_run_records(manifest, valid) == valid
    with pytest.raises(RunValidationError, match="exactly once"):
        validate_run_records(manifest, valid[:-1])
    with pytest.raises(RunValidationError, match="exactly once"):
        validate_run_records(manifest, (*valid, valid[0]))
    unplanned = RunRecord(planned_run_id="run_" + "f" * 24, outcome="success", output="x")
    with pytest.raises(RunValidationError, match="exactly once"):
        validate_run_records(manifest, (*valid[:-1], unplanned))
