"""Frozen study conditions and deterministic budget-matched planning."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence

from .dataset import hash_public_tasks
from .models import (
    EVAL_CONDITION_IDS,
    EVAL_SCHEMA_VERSION,
    ConditionId,
    ConditionSpec,
    PlannedRun,
    PublicTask,
    StudyManifest,
)

CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(condition_id="single_frontier", description="One frontier-model answer."),
    ConditionSpec(
        condition_id="self_refine", description="One answer followed by self-refinement."
    ),
    ConditionSpec(
        condition_id="independent_synthesis",
        description="Independent answers combined by a synthesizer.",
    ),
    ConditionSpec(condition_id="critique_only", description="Independent answers plus critique."),
    ConditionSpec(condition_id="revision_only", description="Independent answers plus revision."),
    ConditionSpec(condition_id="elite_full", description="The complete Elite decision protocol."),
)
CONDITION_IDS: tuple[ConditionId, ...] = EVAL_CONDITION_IDS


def condition_order(seed: int) -> tuple[ConditionId, ...]:
    """Return all frozen conditions in a deterministic seeded order."""

    ordered = list(CONDITION_IDS)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def _validate_budgets(output_token_budgets: Mapping[str, int]) -> None:
    if set(output_token_budgets) != set(CONDITION_IDS):
        raise ValueError("output_token_budgets must contain exactly the six frozen conditions")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in output_token_budgets.values()
    ):
        raise ValueError("output token budgets must be positive integers")
    minimum = min(output_token_budgets.values())
    maximum = max(output_token_budgets.values())
    if maximum > minimum * 1.05:
        raise ValueError("planned output-token ceilings must remain within 5% across conditions")


def _planned_run_id(
    *,
    study_id: str,
    task_id: str,
    condition_id: ConditionId,
    replicate: int,
    max_output_tokens: int,
) -> str:
    identity = json.dumps(
        {
            "schema_version": EVAL_SCHEMA_VERSION,
            "study_id": study_id,
            "task_id": task_id,
            "condition_id": condition_id,
            "replicate": replicate,
            "max_output_tokens": max_output_tokens,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"run_{hashlib.sha256(identity).hexdigest()[:24]}"


def build_study_manifest(
    *,
    study_id: str,
    tasks: Sequence[PublicTask],
    replicates: int,
    seed: int,
    output_token_budgets: Mapping[str, int],
) -> StudyManifest:
    """Predeclare every task-condition-replicate cell with stable identities."""

    if replicates < 1:
        raise ValueError("replicates must be at least one")
    if not tasks:
        raise ValueError("at least one public task is required")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("public task_id values must be unique")
    _validate_budgets(output_token_budgets)

    sorted_tasks = sorted(tasks, key=lambda task: task.task_id)
    ordered_conditions = condition_order(seed)
    planned_runs = tuple(
        PlannedRun(
            planned_run_id=_planned_run_id(
                study_id=study_id,
                task_id=task.task_id,
                condition_id=condition_id,
                replicate=replicate,
                max_output_tokens=output_token_budgets[condition_id],
            ),
            study_id=study_id,
            task_id=task.task_id,
            condition_id=condition_id,
            replicate=replicate,
            max_output_tokens=output_token_budgets[condition_id],
        )
        for task in sorted_tasks
        for replicate in range(1, replicates + 1)
        for condition_id in ordered_conditions
    )
    return StudyManifest(
        study_id=study_id,
        seed=seed,
        replicates=replicates,
        task_ids=tuple(task.task_id for task in sorted_tasks),
        public_tasks_hash=hash_public_tasks(sorted_tasks),
        planned_runs=planned_runs,
    )
