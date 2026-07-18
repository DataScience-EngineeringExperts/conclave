from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conclave.evals.dataset import (
    hash_grader_keys,
    hash_public_tasks,
    load_grader_keys,
    load_public_tasks,
)
from conclave.evals.models import (
    EVAL_SCHEMA_VERSION,
    GraderKey,
    PublicTask,
    ScoreRecord,
    StudyManifest,
)


def test_eval_contracts_are_versioned_and_immutable() -> None:
    task = PublicTask(task_id="task-1", prompt="Choose the safer migration plan.")
    score = ScoreRecord(
        planned_run_id="run_abc",
        grader_id="grader-1",
        critical_error_free=True,
        dimensions={"correctness": 4},
    )

    assert task.schema_version == EVAL_SCHEMA_VERSION
    assert score.schema_version == EVAL_SCHEMA_VERSION
    with pytest.raises(ValidationError):
        task.prompt = "changed"


def test_public_tasks_and_grader_keys_load_from_separate_files(tmp_path) -> None:
    public_path = tmp_path / "tasks.json"
    key_path = tmp_path / "grader-keys.json"
    public_path.write_text(
        json.dumps(
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "tasks": [
                    {
                        "task_id": "task-1",
                        "prompt": "Choose a plan.",
                        "reference_packets": ["Public constraint A"],
                    }
                ],
            }
        )
    )
    key_path.write_text(
        json.dumps(
            {
                "schema_version": EVAL_SCHEMA_VERSION,
                "grader_keys": [
                    {
                        "task_id": "task-1",
                        "required_facts": ["Private expected fact"],
                        "critical_errors": ["Unsafe recommendation"],
                    }
                ],
            }
        )
    )

    tasks = load_public_tasks(public_path)
    keys = load_grader_keys(key_path)

    assert tasks == [
        PublicTask(
            task_id="task-1",
            prompt="Choose a plan.",
            reference_packets=("Public constraint A",),
        )
    ]
    assert keys == [
        GraderKey(
            task_id="task-1",
            required_facts=("Private expected fact",),
            critical_errors=("Unsafe recommendation",),
        )
    ]
    assert "required_facts" not in tasks[0].model_dump()


def test_dataset_hashes_are_stable_across_input_order_and_formatting() -> None:
    first = [
        PublicTask(task_id="b", prompt="Second"),
        PublicTask(task_id="a", prompt="First", reference_packets=("R1",)),
    ]
    second = list(reversed(first))
    keys = [GraderKey(task_id="a", required_facts=("fact",))]

    assert hash_public_tasks(first) == hash_public_tasks(second)
    assert hash_public_tasks(first).startswith("sha256:")
    assert hash_grader_keys(keys) == hash_grader_keys(list(keys))
    assert hash_public_tasks(first) != hash_grader_keys(keys)


def test_study_manifest_rejects_an_unversioned_or_incomplete_run_matrix() -> None:
    with pytest.raises(ValidationError):
        StudyManifest(
            study_id="study-1",
            seed=7,
            replicates=1,
            task_ids=("task-1",),
            public_tasks_hash="sha256:" + "a" * 64,
            planned_runs=(),
        )
