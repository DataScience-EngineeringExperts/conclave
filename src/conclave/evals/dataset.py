"""Separate loading and canonical hashing for public tasks and grader keys."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .models import GraderKey, GraderKeyDataset, PublicTask, PublicTaskDataset


def _canonical_hash(namespace: str, records: Iterable[PublicTask | GraderKey]) -> str:
    ordered = sorted(records, key=lambda record: record.task_id)
    canonical = json.dumps(
        {
            "namespace": namespace,
            "records": [record.model_dump(mode="json") for record in ordered],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def hash_public_tasks(tasks: Iterable[PublicTask]) -> str:
    """Return the stable, order-independent digest of public task records."""

    return _canonical_hash("public_tasks", tasks)


def hash_grader_keys(keys: Iterable[GraderKey]) -> str:
    """Return the stable, order-independent digest of grader-only records."""

    return _canonical_hash("grader_keys", keys)


def _load_json(path: str | Path) -> object:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_public_tasks(path: str | Path) -> list[PublicTask]:
    """Load only a public-task envelope; grader fields are rejected as extras."""

    dataset = PublicTaskDataset.model_validate(_load_json(path))
    task_ids = [task.task_id for task in dataset.tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("public task_id values must be unique")
    return list(dataset.tasks)


def load_grader_keys(path: str | Path) -> list[GraderKey]:
    """Load grader keys independently from the public execution dataset."""

    dataset = GraderKeyDataset.model_validate(_load_json(path))
    task_ids = [key.task_id for key in dataset.grader_keys]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("grader key task_id values must be unique")
    return list(dataset.grader_keys)
