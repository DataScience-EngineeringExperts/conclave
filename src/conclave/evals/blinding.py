"""Seeded opaque output blinding with a physically separate identity map."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .models import EVAL_SCHEMA_VERSION, RunOutcome, RunRecord, SchemaVersion


class BlindModel(BaseModel):
    """Immutable and drift-resistant blinding artifact contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION


class BlindedOutput(BlindModel):
    """Grader-visible output without execution identity or labels."""

    opaque_output_id: str = Field(pattern=r"^output_[0-9a-f]{24}$")
    outcome: RunOutcome
    output: str | None = None


class BlindedOutputSet(BlindModel):
    """Artifact safe to provide to graders."""

    outputs: tuple[BlindedOutput, ...]


class BlindMapEntry(BlindModel):
    """Restricted mapping from an opaque output to its execution identity."""

    opaque_output_id: str = Field(pattern=r"^output_[0-9a-f]{24}$")
    planned_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")


class BlindMap(BlindModel):
    """Separate restricted identity artifact, never included in grader output."""

    entries: tuple[BlindMapEntry, ...]


def _opaque_id(*, planned_run_id: str, seed: int) -> str:
    identity = f"{seed}:{planned_run_id}".encode()
    return f"output_{hashlib.sha256(identity).hexdigest()[:24]}"


def blind_run_records(
    records: Sequence[RunRecord], *, seed: int
) -> tuple[BlindedOutputSet, BlindMap]:
    """Blind and deterministically shuffle records, returning the map separately."""

    if len({record.planned_run_id for record in records}) != len(records):
        raise ValueError("run records must have unique planned_run_id values")
    blinded = [
        (
            BlindedOutput(
                opaque_output_id=_opaque_id(planned_run_id=record.planned_run_id, seed=seed),
                outcome=record.outcome,
                output=record.output,
            ),
            BlindMapEntry(
                opaque_output_id=_opaque_id(planned_run_id=record.planned_run_id, seed=seed),
                planned_run_id=record.planned_run_id,
            ),
        )
        for record in records
    ]
    random.Random(seed).shuffle(blinded)
    return (
        BlindedOutputSet(outputs=tuple(output for output, _ in blinded)),
        BlindMap(entries=tuple(entry for _, entry in blinded)),
    )
