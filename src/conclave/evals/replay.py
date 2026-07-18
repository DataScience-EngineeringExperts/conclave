"""Strict buffered transport record/replay for offline evaluation runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..adapters.base import redact
from .models import Sha256Digest

REPLAY_SCHEMA_VERSION = "conclave_replay_v1"
_CREDENTIAL_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "signature",
    "x-api-key",
    "x-goog-api-key",
}
_AMBIGUOUS_CREDENTIAL_NAMES = {"key", "token"}

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

    @model_validator(mode="after")
    def validate_integrity(self) -> ReplayArtifact:
        occurrences: dict[str, list[int]] = {}
        for record in self.records:
            sanitized_request = _sanitize_stored_request(record.request)
            if sanitized_request != record.request:
                raise ValueError("replay request must be sanitized")
            if _sanitize(record.response, ()) != record.response:
                raise ValueError("replay response must be sanitized")
            if _hash_request(record.request) != record.request_hash:
                raise ValueError("replay request hash does not match stored request")
            occurrences.setdefault(record.request_hash, []).append(record.occurrence_index)
        if any(sorted(indexes) != list(range(len(indexes))) for indexes in occurrences.values()):
            raise ValueError("replay occurrence indexes must be contiguous from zero")
        return self


def _is_sensitive_name(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    return lowered in {item.replace("-", "_") for item in _CREDENTIAL_NAMES} or any(
        marker in lowered for marker in ("secret", "password", "authorization")
    )


def _redact_exact(text: str, credentials: tuple[str, ...]) -> str:
    cleaned = text
    for credential in sorted(credentials, key=len, reverse=True):
        if credential:
            cleaned = cleaned.replace(credential, "[REDACTED]")
    return redact(cleaned)


def _sanitize(value: Any, credentials: tuple[str, ...], *, body_root: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, credentials)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_sensitive_name(str(key))
            and not (
                body_root and str(key).lower().replace("-", "_") in _AMBIGUOUS_CREDENTIAL_NAMES
            )
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, credentials) for item in value]
    if isinstance(value, str):
        return _redact_exact(value, credentials)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_exact(str(value), credentials)


def _sanitize_url(url: str, credentials: tuple[str, ...] = ()) -> str:
    parts = urlsplit(url)
    safe_query = [
        (name, _redact_exact(value, credentials))
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_sensitive_name(name)
        and name.lower().replace("-", "_") not in _AMBIGUOUS_CREDENTIAL_NAMES
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(safe_query), ""))


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _string_values(nested)]
    if isinstance(value, (list, tuple)):
        return [item for nested in value for item in _string_values(nested)]
    return []


def _credentials(url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for name, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
        normalized = name.lower().replace("-", "_")
        if _is_sensitive_name(name) or normalized in _AMBIGUOUS_CREDENTIAL_NAMES:
            values.extend(_string_values(value))
    for name, value in headers.items():
        if _is_sensitive_name(name):
            values.append(value)
            if value.lower().startswith("bearer "):
                values.append(value[7:])
    for name, value in body.items():
        normalized = str(name).lower().replace("-", "_")
        if _is_sensitive_name(str(name)) or normalized in _AMBIGUOUS_CREDENTIAL_NAMES:
            values.extend(_string_values(value))
    return tuple(dict.fromkeys(item for item in values if item))


def _hash_request(request: Mapping[str, Any]) -> str:
    canonical = json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _sanitize_stored_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if set(request) != {"url", "body"} or not isinstance(request.get("url"), str):
        raise ValueError("sanitized replay request must contain only string url and object body")
    body = request.get("body")
    if not isinstance(body, Mapping):
        raise ValueError("sanitized replay request body must be an object")
    return {"url": _sanitize_url(request["url"]), "body": _sanitize(body, (), body_root=True)}


def _request(
    url: str, headers: Mapping[str, str], body: dict
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    credentials = _credentials(url, headers, body)
    safe = {
        "url": _sanitize_url(url, credentials),
        "body": _sanitize(body, credentials, body_root=True),
    }
    return safe, _hash_request(safe), credentials


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
        safe_request, request_hash, credentials = _request(url, headers, json_body)
        occurrence = self._counts[request_hash]
        self._counts[request_hash] += 1
        status, response = await self._delegate(url, headers, json_body, timeout)
        self._records.append(
            ReplayRecord(
                request_hash=request_hash,
                occurrence_index=occurrence,
                request=safe_request,
                status=status,
                response=_sanitize(response, credentials),
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

    async def __call__(
        self, url: str, headers: dict[str, str], json_body: dict, timeout: float
    ) -> tuple[int, object]:
        del timeout
        _safe_request, request_hash, _credentials_found = _request(url, headers, json_body)
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
