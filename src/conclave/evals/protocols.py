"""Frozen study conditions and deterministic budget-matched planning."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .dataset import hash_public_tasks
from .models import (
    EVAL_CONDITION_IDS,
    ConditionId,
    ConditionSpec,
    FrozenStudyDesign,
    PlannedRun,
    PublicTask,
    StudyManifest,
    derive_planned_run_id,
    hash_frozen_study_design,
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

if TYPE_CHECKING:
    from .live_protocols import LiveProtocolSpec


def live_protocol_registry() -> Mapping[ConditionId, LiveProtocolSpec]:
    """Return the versioned live executors' metadata without an import cycle."""

    from .live_protocols import LIVE_PROTOCOL_REGISTRY

    return LIVE_PROTOCOL_REGISTRY


def condition_order(seed: int) -> tuple[ConditionId, ...]:
    """Return all frozen conditions in a deterministic seeded order."""

    ordered = list(CONDITION_IDS)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def blocked_condition_order(
    *, master_seed: int, task_id: str, roster_id: str
) -> tuple[ConditionId, ...]:
    """Derive one deterministic, independent permutation per task-roster block."""

    identity = json.dumps(
        {
            "method": "sha256_task_roster_block_v1",
            "master_seed": master_seed,
            "task_id": task_id,
            "roster_id": roster_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(identity).digest(), "big")
    ordered = list(CONDITION_IDS)
    random.Random(derived_seed).shuffle(ordered)
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


def build_study_manifest(
    *,
    study_id: str,
    tasks: Sequence[PublicTask],
    replicates: int,
    seed: int,
    output_token_budgets: Mapping[str, int],
    frozen_design: FrozenStudyDesign | None = None,
) -> StudyManifest:
    """Predeclare every legacy or frozen task-roster-condition-replicate cell."""

    if replicates < 1:
        raise ValueError("replicates must be at least one")
    if not tasks:
        raise ValueError("at least one public task is required")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("public task_id values must be unique")
    _validate_budgets(output_token_budgets)
    if frozen_design is not None:
        if set(frozen_design.task_family_map) != set(task_ids):
            raise ValueError("task_family_map must exactly cover public task IDs")
        if frozen_design.randomization.master_seed != seed:
            raise ValueError("seed must match the frozen randomization master seed")

    sorted_tasks = sorted(tasks, key=lambda task: task.task_id)
    roster_ids = (
        tuple(sorted(roster.roster_id for roster in frozen_design.rosters))
        if frozen_design is not None
        else ("legacy_default",)
    )
    design_hash = hash_frozen_study_design(frozen_design) if frozen_design is not None else None
    planned_runs = tuple(
        PlannedRun(
            planned_run_id=derive_planned_run_id(
                study_id=study_id,
                task_id=task.task_id,
                condition_id=condition_id,
                replicate=replicate,
                max_output_tokens=output_token_budgets[condition_id],
                roster_id=roster_id,
                frozen_design_hash=design_hash,
            ),
            study_id=study_id,
            task_id=task.task_id,
            roster_id=roster_id,
            condition_id=condition_id,
            replicate=replicate,
            max_output_tokens=output_token_budgets[condition_id],
        )
        for task in sorted_tasks
        for replicate in range(1, replicates + 1)
        for roster_id in roster_ids
        for condition_id in (
            blocked_condition_order(master_seed=seed, task_id=task.task_id, roster_id=roster_id)
            if frozen_design is not None
            else condition_order(seed)
        )
    )
    return StudyManifest(
        study_id=study_id,
        evidence_classification=(
            frozen_design.evidence_classification
            if frozen_design is not None
            else "synthetic_exploratory"
        ),
        seed=seed,
        replicates=replicates,
        task_ids=tuple(task.task_id for task in sorted_tasks),
        public_tasks_hash=hash_public_tasks(sorted_tasks),
        frozen_design=frozen_design,
        frozen_design_hash=design_hash,
        planned_runs=planned_runs,
    )
