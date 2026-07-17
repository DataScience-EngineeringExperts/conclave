"""Strict buffered transport record/replay for offline evaluation runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from ..adapters.base import redact
from .models import Sha256Digest

REPLAY_SCHEMA_VERSION = "conclave_replay_v1"
_SENSITIVE_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "key",
    "password",
    "secret",
    "signature",
    "token",
    "x-api-key",
    "x-goog-api-key",
}

PostJson = Callable[[str, dict[str, str], dict, float], Awaitable[tuple[int, object]]]


class ReplayError(RuntimeError):
    """Base class for fail-closed replay errors."""


class ReplayCompatibilityError(ReplayError):
    """The artifact cannot be used with the requested study manifest."""


class ReplayMismatchError(ReplayError):
    """Recorded and attempted transport calls do not match exactly."""


class ReplayRecord(BaseModel):
    """One sanitized request/response exchange at a deterministic occurrence."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    request_hash: Sha256Digest
    occurrence_index: int = Field(ge=0)
    request: dict[str, Any]
    status: int
    response: Any


class ReplayArtifact(BaseModel):
    """Versioned recording bound to the exact base study manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["conclave_replay_v1"] = REPLAY_SCHEMA_VERSION
    base_manifest_hash: Sha256Digest
    records: tuple[ReplayRecord, ...]


def _is_sensitive_name(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    return lowered in {item.replace("-", "_") for item in _SENSITIVE_NAMES} or any(
        marker in lowered for marker in ("secret", "password", "authorization")
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_sensitive_name(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


def _sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    safe_query = [
        (name, redact(value))
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_sensitive_name(name)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(safe_query), ""))


def _request(url: str, body: dict) -> tuple[dict[str, Any], str]:
    safe = {"url": _sanitize_url(url), "body": _sanitize(body)}
    canonical = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    return safe, digest


class RecordingPostJson:
    """Callable drop-in for :func:`transport.post_json` that records safe artifacts."""

    def __init__(self, delegate: PostJson, *, base_manifest_hash: str) -> None:
        self._delegate = delegate
        self._base_manifest_hash = base_manifest_hash
        self._counts: Counter[str] = Counter()
        self._records: list[ReplayRecord] = []

    async def __call__(
        self, url: str, headers: dict[str, str], json_body: dict, timeout: float
    ) -> tuple[int, object]:
        safe_request, request_hash = _request(url, json_body)
        occurrence = self._counts[request_hash]
        self._counts[request_hash] += 1
        status, response = await self._delegate(url, headers, json_body, timeout)
        self._records.append(
            ReplayRecord(
                request_hash=request_hash,
                occurrence_index=occurrence,
                request=safe_request,
                status=status,
                response=_sanitize(response),
            )
        )
        return status, response

    def artifact(self) -> ReplayArtifact:
        return ReplayArtifact(
            base_manifest_hash=self._base_manifest_hash,
            records=tuple(self._records),
        )


class ReplayingPostJson:
    """Callable zero-network replay that requires an exact complete call set."""

    def __init__(self, artifact: ReplayArtifact, *, base_manifest_hash: str) -> None:
        if artifact.schema_version != REPLAY_SCHEMA_VERSION:
            raise ReplayCompatibilityError(
                f"replay schema version mismatch: {artifact.schema_version!r}"
            )
        if artifact.base_manifest_hash != base_manifest_hash:
            raise ReplayCompatibilityError("replay base manifest hash mismatch")
        self._records = {
            (record.request_hash, record.occurrence_index): record for record in artifact.records
        }
        if len(self._records) != len(artifact.records):
            raise ReplayCompatibilityError("replay contains duplicate request occurrences")
        self._counts: Counter[str] = Counter()
        self._consumed: set[tuple[str, int]] = set()

    async def __call__(
        self, url: str, headers: dict[str, str], json_body: dict, timeout: float
    ) -> tuple[int, object]:
        del headers, timeout
        _safe_request, request_hash = _request(url, json_body)
        occurrence = self._counts[request_hash]
        self._counts[request_hash] += 1
        key = (request_hash, occurrence)
        record = self._records.get(key)
        if record is None:
            raise ReplayMismatchError(f"unmatched request {request_hash} occurrence {occurrence}")
        self._consumed.add(key)
        return record.status, record.response

    def assert_consumed(self) -> None:
        remaining = set(self._records) - self._consumed
        if remaining:
            raise ReplayMismatchError(f"{len(remaining)} unconsumed record(s) remain")
