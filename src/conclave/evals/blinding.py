"""Seeded opaque output blinding with a physically separate identity map."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import EVAL_SCHEMA_VERSION, RunOutcome, RunRecord, SchemaVersion, Sha256Digest


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
    blind_map_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_hash(self) -> BlindMap:
        if self.blind_map_hash is not None and self.blind_map_hash != hash_blind_map(self.entries):
            raise ValueError("blind map hash does not match its entries")
        return self


class GraderQueueOutput(BlindModel):
    """One normalized successful output safe for paid human grading."""

    opaque_output_id: str = Field(pattern=r"^output_[0-9a-f]{24}$")
    presentation: str = Field(min_length=1)


class GraderQueue(BlindModel):
    """Successful-only grader queue with no execution labels or outcomes."""

    outputs: tuple[GraderQueueOutput, ...]


def hash_blind_map(entries: Sequence[BlindMapEntry]) -> str:
    """Hash a restricted identity map in its frozen queue order."""

    canonical = json.dumps(
        [entry.model_dump(mode="json") for entry in entries],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


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


def build_grader_queue(
    records: Sequence[RunRecord], *, seed: int, forbidden_labels: Sequence[str]
) -> tuple[GraderQueue, BlindMap]:
    """Normalize, leak-check, and blind successful outputs for paid grading."""

    if len({record.planned_run_id for record in records}) != len(records):
        raise ValueError("run records must have unique planned_run_id values")
    blinded = []
    labels = tuple(label.casefold() for label in forbidden_labels if label)
    for record in records:
        if record.outcome != "success":
            continue
        if record.output is None:
            raise ValueError("successful grader-queue records require output")
        presentation = " ".join(record.output.split())
        if not presentation:
            raise ValueError("successful grader-queue records require nonempty output")
        folded = presentation.casefold()
        if any(label in folded for label in labels):
            raise ValueError("grader presentation contains a forbidden label")
        opaque_id = _opaque_id(planned_run_id=record.planned_run_id, seed=seed)
        blinded.append(
            (
                GraderQueueOutput(opaque_output_id=opaque_id, presentation=presentation),
                BlindMapEntry(opaque_output_id=opaque_id, planned_run_id=record.planned_run_id),
            )
        )
    random.Random(seed).shuffle(blinded)
    entries = tuple(entry for _, entry in blinded)
    return (
        GraderQueue(outputs=tuple(output for output, _ in blinded)),
        BlindMap(entries=entries, blind_map_hash=hash_blind_map(entries)),
    )
