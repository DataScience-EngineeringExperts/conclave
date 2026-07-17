from __future__ import annotations

import pytest
from pydantic import ValidationError

from conclave.evals.models import PublicTask
from conclave.evals.protocols import (
    CONDITION_IDS,
    build_study_manifest,
    condition_order,
)

EXPECTED_CONDITIONS = (
    "single_frontier",
    "self_refine",
    "independent_synthesis",
    "critique_only",
    "revision_only",
    "elite_full",
)


def test_registry_contains_exactly_the_six_frozen_conditions() -> None:
    assert CONDITION_IDS == EXPECTED_CONDITIONS


def test_condition_order_is_seeded_deterministic_and_complete() -> None:
    first = condition_order(seed=20260717)
    second = condition_order(seed=20260717)
    other = condition_order(seed=20260718)

    assert first == second
    assert set(first) == set(EXPECTED_CONDITIONS)
    assert len(first) == len(EXPECTED_CONDITIONS)
    assert first != other


def test_budget_plan_accepts_exact_boundary_and_rejects_over_five_percent() -> None:
    tasks = [PublicTask(task_id="task-1", prompt="Decide")]
    accepted = {
        condition_id: (1000 if condition_id != "elite_full" else 1050)
        for condition_id in EXPECTED_CONDITIONS
    }
    manifest = build_study_manifest(
        study_id="study-1",
        tasks=tasks,
        replicates=1,
        seed=17,
        output_token_budgets=accepted,
    )
    assert {run.max_output_tokens for run in manifest.planned_runs} == {1000, 1050}

    rejected = dict(accepted)
    rejected["elite_full"] = 1051
    with pytest.raises(ValueError, match="within 5%"):
        build_study_manifest(
            study_id="study-1",
            tasks=tasks,
            replicates=1,
            seed=17,
            output_token_budgets=rejected,
        )


def test_plan_declares_complete_matrix_with_stable_immutable_ids() -> None:
    tasks = [
        PublicTask(task_id="task-b", prompt="B"),
        PublicTask(task_id="task-a", prompt="A"),
    ]
    budgets = dict.fromkeys(EXPECTED_CONDITIONS, 1200)

    first = build_study_manifest(
        study_id="elite-pilot",
        tasks=tasks,
        replicates=2,
        seed=99,
        output_token_budgets=budgets,
    )
    second = build_study_manifest(
        study_id="elite-pilot",
        tasks=list(reversed(tasks)),
        replicates=2,
        seed=99,
        output_token_budgets=budgets,
    )

    assert len(first.planned_runs) == 2 * 6 * 2
    assert len({run.planned_run_id for run in first.planned_runs}) == 24
    assert {run.task_id for run in first.planned_runs} == {"task-a", "task-b"}
    assert {run.condition_id for run in first.planned_runs} == set(EXPECTED_CONDITIONS)
    assert {run.replicate for run in first.planned_runs} == {1, 2}
    assert first == second
    assert all(run.planned_run_id.startswith("run_") for run in first.planned_runs)
    with pytest.raises(ValidationError):
        first.planned_runs[0].max_output_tokens = 1


def test_plan_requires_every_condition_budget_and_valid_replicates() -> None:
    tasks = [PublicTask(task_id="task-1", prompt="Decide")]
    with pytest.raises(ValueError, match="exactly"):
        build_study_manifest(
            study_id="study-1",
            tasks=tasks,
            replicates=1,
            seed=17,
            output_token_budgets={"elite_full": 1000},
        )
    with pytest.raises(ValueError, match="replicates"):
        build_study_manifest(
            study_id="study-1",
            tasks=tasks,
            replicates=0,
            seed=17,
            output_token_budgets=dict.fromkeys(EXPECTED_CONDITIONS, 1000),
        )
