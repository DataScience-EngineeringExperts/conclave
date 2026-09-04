"""Strict buffered transport record/replay for offline evaluation runs.

DSE-1517 Task 1: the sanitizing transport this module used to own directly has
been promoted to :mod:`conclave.tape` as a product module (``conclave ask
--record`` / ``conclave replay`` reuse it). Every name below now delegates to
``conclave.tape`` -- this module keeps its own public names, its own
``ReplayArtifact`` schema (bound to an eval study's ``base_manifest_hash``
rather than a run identity), and byte-identical behavior for every existing
caller and test.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ..tape import (
    PostJson,
    ReplayCompatibilityError,
    ReplayError,
    ReplayMismatchError,
    Sha256Digest,
    _validate_tape_records,
)
from ..tape import RecordingTransport as _RecordingTransport
from ..tape import ReplayingTransport as _ReplayingTransport
from ..tape import TapeRecord as ReplayRecord

REPLAY_SCHEMA_VERSION = "conclave_replay_v1"

__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "PostJson",
    "RecordingPostJson",
    "ReplayArtifact",
    "ReplayCompatibilityError",
    "ReplayError",
    "ReplayingPostJson",
    "ReplayMismatchError",
    "ReplayRecord",
    "Sha256Digest",
]


class ReplayArtifact(BaseModel):
    """Versioned recording bound to the exact base study manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["conclave_replay_v1"] = REPLAY_SCHEMA_VERSION
    base_manifest_hash: Sha256Digest
    records: tuple[ReplayRecord, ...]

    @model_validator(mode="after")
    def validate_integrity(self) -> ReplayArtifact:
        _validate_tape_records(self.records)
        return self


class RecordingPostJson(_RecordingTransport):
    """Callable drop-in for :func:`transport.post_json` that records safe artifacts."""

    def __init__(self, delegate: PostJson, *, base_manifest_hash: str) -> None:
        super().__init__(delegate, run_identity_hash=base_manifest_hash)
        self._base_manifest_hash = base_manifest_hash

    def artifact(self) -> ReplayArtifact:
        return ReplayArtifact(
            base_manifest_hash=self._base_manifest_hash,
            records=tuple(self._records),
        )


class ReplayingPostJson(_ReplayingTransport):
    """Callable zero-network replay that requires an exact complete call set."""

    def __init__(self, artifact: ReplayArtifact, *, base_manifest_hash: str) -> None:
        if artifact.schema_version != REPLAY_SCHEMA_VERSION:
            raise ReplayCompatibilityError(
                f"replay schema version mismatch: {artifact.schema_version!r}"
            )
        if artifact.base_manifest_hash != base_manifest_hash:
            raise ReplayCompatibilityError("replay base manifest hash mismatch")
        try:
            artifact = ReplayArtifact.model_validate(artifact.model_dump(mode="python"))
        except ValueError as exc:
            raise ReplayCompatibilityError("replay artifact integrity validation failed") from exc
        self._records = {
            (record.request_hash, record.occurrence_index): record for record in artifact.records
        }
        if len(self._records) != len(artifact.records):
            raise ReplayCompatibilityError("replay contains duplicate request occurrences")
        self._counts: Counter[str] = Counter()
        self._consumed: set[tuple[str, int]] = set()
