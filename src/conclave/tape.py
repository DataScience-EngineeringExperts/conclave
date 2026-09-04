"""Sanitizing record/replay transport for provider HTTP calls.

Promoted out of :mod:`conclave.evals.replay` (DSE-1517 Task 1) so the product
decision-record feature (``conclave ask --record`` / ``conclave replay``) and
the eval subsystem share one sanitizer instead of two independently maintained
copies. ``conclave.evals.replay`` now delegates to the names defined here; it
must keep working byte-for-byte (see ``tests/evals/test_replay.py``).

A :class:`Tape` is a sequence of sanitized request/response exchanges bound to
one run identity (see :func:`conclave.cache.build_identity`). Recording never
stores a live credential: the URL, headers, and body are hashed for a stable
request key and every credential-shaped name or value is stripped or replaced
with ``[REDACTED]`` before anything is written down. This module MUST NOT
import from :mod:`conclave.evals` -- the dependency runs the other way.

Hardening (DSE-1517 Task 1b, closing CSO findings F2/F3/F15):

* ``_sanitize_url`` strips URL userinfo (``user:pass@host``) from the netloc --
  a bare :func:`urllib.parse.urlsplit` keeps it, so a credential embedded in a
  custom endpoint URL would otherwise reach a published tape verbatim (F2).
* ``_sanitize_url`` also drops the union of this module's own credential names
  and :mod:`conclave.cache`'s wider ``_SECRET_QUERY_KEYS`` /
  ``_SECRET_QUERY_PARTS`` / ``_SECRET_VALUE_MARKERS`` sets, so a tape's query
  scrubbing is never narrower than the cache's endpoint-fingerprint scrubbing
  (F3).
* :class:`RecordingTransport` sanitizes the response body with
  ``body_root=True``, dropping ambiguous top-level keys (``key`` / ``token``)
  from provider responses the same way request bodies already are (F15).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any, Literal
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adapters.base import redact
from .cache import _SECRET_QUERY_KEYS, _SECRET_QUERY_PARTS, _SECRET_VALUE_MARKERS

TAPE_SCHEMA_VERSION = "conclave_tape_v1"

Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

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
    """Base class for fail-closed record/replay errors."""


class ReplayCompatibilityError(ReplayError):
    """The tape cannot be used with the requested run identity."""


class ReplayMismatchError(ReplayError):
    """Recorded and attempted transport calls do not match exactly."""


class TapeRecord(BaseModel):
    """One sanitized request/response exchange at a deterministic occurrence.

    Attributes:
        request_hash: ``sha256:<hex>`` over the sanitized, canonicalized
            request (see :func:`_hash_request`).
        occurrence_index: 0-based count of prior identical requests -- lets a
            tape distinguish two calls with the same URL/body (e.g. retries).
        request: ``{"url": ..., "body": ...}``, already sanitized.
        status: HTTP status code returned by the provider.
        response: The sanitized, decoded JSON response body.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    request_hash: Sha256Digest
    occurrence_index: int = Field(ge=0)
    request: dict[str, Any]
    status: int
    response: Any


def _validate_tape_records(records: Sequence[TapeRecord]) -> None:
    """Enforce sanitization, hash integrity, and contiguous occurrences.

    Shared by :class:`Tape` and (via import) ``conclave.evals.replay.ReplayArtifact``
    so both artifact shapes carry byte-identical integrity guarantees. Raises
    ``ValueError`` naming the first violation found.

    Args:
        records: The candidate records to validate.

    Raises:
        ValueError: A record is unsanitized, its hash does not match its
            stored request, or occurrence indexes for a request hash are not
            contiguous starting at zero.
    """
    occurrences: dict[str, list[int]] = {}
    for record in records:
        sanitized_request = _sanitize_stored_request(record.request)
        if sanitized_request != record.request:
            raise ValueError("tape request must be sanitized")
        if _sanitize(record.response, ()) != record.response:
            raise ValueError("tape response must be sanitized")
        if _hash_request(record.request) != record.request_hash:
            raise ValueError("tape request hash does not match stored request")
        occurrences.setdefault(record.request_hash, []).append(record.occurrence_index)
    if any(sorted(indexes) != list(range(len(indexes))) for indexes in occurrences.values()):
        raise ValueError("tape occurrence indexes must be contiguous from zero")


class Tape(BaseModel):
    """Sanitized request/response records bound to one run identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["conclave_tape_v1"] = TAPE_SCHEMA_VERSION
    run_identity_hash: Sha256Digest
    records: tuple[TapeRecord, ...]

    @model_validator(mode="after")
    def validate_integrity(self) -> Tape:
        _validate_tape_records(self.records)
        return self


def _is_sensitive_name(name: str) -> bool:
    """Return True if ``name`` looks like a credential field/header/param name."""
    lowered = name.lower().replace("-", "_")
    return lowered in {item.replace("-", "_") for item in _CREDENTIAL_NAMES} or any(
        marker in lowered for marker in ("secret", "password", "authorization")
    )


def _redact_exact(text: str, credentials: tuple[str, ...]) -> str:
    """Replace every occurrence of a known credential value, then run :func:`redact`."""
    cleaned = text
    for credential in sorted(credentials, key=len, reverse=True):
        if credential:
            cleaned = cleaned.replace(credential, "[REDACTED]")
    return redact(cleaned)


def _sanitize(value: Any, credentials: tuple[str, ...], *, body_root: bool = False) -> Any:
    """Recursively strip credential-named keys and redact credential-shaped values.

    Args:
        value: Any JSON-decoded value (mapping, sequence, scalar).
        credentials: Exact credential values (e.g. a live env-var key) to blot
            out wherever they appear as a substring.
        body_root: When True, also drop the ambiguous names ``key`` / ``token``
            at this call's top level only (never at nested depths, where
            dropping them would corrupt legitimate content such as a message
            that happens to discuss "token budgeting").

    Returns:
        A sanitized deep copy of ``value``.
    """
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


def _safe_netloc(parts: SplitResult) -> str:
    """Return ``host[:port]`` from a split URL with userinfo stripped (F2).

    A bare ``urlsplit(...).netloc`` keeps ``user:pass@`` verbatim -- this is
    the only safe way to render a netloc into a tape or a sanitized endpoint
    disclosure.

    Args:
        parts: The result of ``urlsplit(url)``.

    Returns:
        ``hostname`` or ``hostname:port``; empty string if there is no host.
        A malformed port is dropped rather than raised, matching
        ``cache._endpoint_fingerprint``'s tolerance of a bad configured port.
    """
    hostname = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{hostname}:{port}" if port is not None else hostname


def _is_secret_query_name(name: str) -> bool:
    """True if a query-parameter name is credential-shaped by any known rule.

    Unions this module's own name/ambiguous-name rules with
    ``cache._SECRET_QUERY_KEYS`` (exact match) and ``cache._SECRET_QUERY_PARTS``
    (substring match), so a tape's URL scrubbing is never narrower than the
    cache's endpoint-fingerprint scrubbing (F3).
    """
    normalized = name.lower().replace("-", "_")
    if _is_sensitive_name(name) or normalized in _AMBIGUOUS_CREDENTIAL_NAMES:
        return True
    lowered = name.lower()
    if lowered in _SECRET_QUERY_KEYS:
        return True
    return any(part in lowered for part in _SECRET_QUERY_PARTS)


def _is_secret_query_value(value: str) -> bool:
    """True if a query-parameter value contains one of ``cache._SECRET_VALUE_MARKERS``."""
    lowered = value.lower()
    return any(marker in lowered for marker in _SECRET_VALUE_MARKERS)


def _sanitize_url(url: str, credentials: tuple[str, ...] = ()) -> str:
    """Return ``url`` with userinfo and every credential-shaped query param removed.

    Args:
        url: The request URL, possibly containing userinfo and/or secret
            query parameters.
        credentials: Exact known credential values to redact from any
            surviving query value.

    Returns:
        The sanitized URL: no userinfo in the netloc, no credential-named or
        credential-shaped-valued query parameters.

    Example:
        >>> _sanitize_url("https://svc:hunter2@litellm.internal:4000/v1/chat?access_key=S&alt=json")
        'https://litellm.internal:4000/v1/chat?alt=json'
    """
    parts = urlsplit(url)
    safe_query = [
        (name, _redact_exact(value, credentials))
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_secret_query_name(name) and not _is_secret_query_value(value)
    ]
    return urlunsplit((parts.scheme, _safe_netloc(parts), parts.path, urlencode(safe_query), ""))


def _string_values(value: Any) -> list[str]:
    """Flatten every string leaf out of a nested mapping/sequence structure."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _string_values(nested)]
    if isinstance(value, (list, tuple)):
        return [item for nested in value for item in _string_values(nested)]
    return []


def _credentials(url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect the exact credential-shaped values present in one live call.

    Used to blot out those exact strings wherever they might otherwise leak
    (e.g. an API key echoed back inside a provider's response body).
    """
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
    """Return the stable ``sha256:<hex>`` identity of a sanitized request."""
    canonical = json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _sanitize_stored_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive the sanitized form of an already-stored ``{"url", "body"}`` request.

    Used by :func:`_validate_tape_records` to prove a stored record was not
    tampered with after sanitization.
    """
    if set(request) != {"url", "body"} or not isinstance(request.get("url"), str):
        raise ValueError("sanitized replay request must contain only string url and object body")
    body = request.get("body")
    if not isinstance(body, Mapping):
        raise ValueError("sanitized replay request body must be an object")
    return {"url": _sanitize_url(request["url"]), "body": _sanitize(body, (), body_root=True)}


def _request(
    url: str, headers: Mapping[str, str], body: dict
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    """Build the sanitized ``{"url", "body"}`` request, its hash, and its credentials."""
    credentials = _credentials(url, headers, body)
    safe = {
        "url": _sanitize_url(url, credentials),
        "body": _sanitize(body, credentials, body_root=True),
    }
    return safe, _hash_request(safe), credentials


class RecordingTransport:
    """Callable drop-in for ``transport.post_json`` that captures a sanitized tape.

    Intended for use as the ``post_json`` field of
    ``conclave.transport.RecordingContext`` (Task 2): every live call this
    wraps is hashed and sanitized before being appended to the tape; the live
    response is returned unchanged to the caller.

    Args:
        delegate: The live transport callable being wrapped.
        run_identity_hash: The digest binding this tape to one run identity
            (see ``cache.build_identity``).
    """

    def __init__(self, delegate: PostJson, *, run_identity_hash: str) -> None:
        self._delegate = delegate
        self._run_identity_hash = run_identity_hash
        self._counts: Counter[str] = Counter()
        self._records: list[TapeRecord] = []

    async def __call__(
        self, url: str, headers: dict[str, str], json_body: dict, timeout: float
    ) -> tuple[int, object]:
        safe_request, request_hash, credentials = _request(url, headers, json_body)
        occurrence = self._counts[request_hash]
        self._counts[request_hash] += 1
        status, response = await self._delegate(url, headers, json_body, timeout)
        self._records.append(
            TapeRecord(
                request_hash=request_hash,
                occurrence_index=occurrence,
                request=safe_request,
                status=status,
                response=_sanitize(response, credentials, body_root=True),
            )
        )
        return status, response

    def tape(self) -> Tape:
        """Return the immutable, self-validating :class:`Tape` recorded so far."""
        return Tape(run_identity_hash=self._run_identity_hash, records=tuple(self._records))


class ReplayingTransport:
    """Callable zero-network replay that requires an exact, complete call set.

    Args:
        tape: The recorded :class:`Tape` to replay against.
        run_identity_hash: Must equal ``tape.run_identity_hash`` -- guards
            against replaying a tape recorded under a different run.

    Raises:
        ReplayCompatibilityError: Schema version or run-identity mismatch, a
            failed re-validation of the tape's integrity, or duplicate
            request occurrences.
    """

    _mismatch_error: type[ReplayError] = ReplayMismatchError

    def __init__(self, tape: Tape, *, run_identity_hash: str) -> None:
        if tape.schema_version != TAPE_SCHEMA_VERSION:
            raise ReplayCompatibilityError(f"tape schema version mismatch: {tape.schema_version!r}")
        if tape.run_identity_hash != run_identity_hash:
            raise ReplayCompatibilityError("tape run identity hash mismatch")
        try:
            tape = Tape.model_validate(tape.model_dump(mode="python"))
        except ValueError as exc:
            raise ReplayCompatibilityError("tape integrity validation failed") from exc
        self._records = {
            (record.request_hash, record.occurrence_index): record for record in tape.records
        }
        if len(self._records) != len(tape.records):
            raise ReplayCompatibilityError("tape contains duplicate request occurrences")
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
            raise self._mismatch_error(f"unmatched request {request_hash} occurrence {occurrence}")
        self._consumed.add(key)
        return record.status, record.response

    def assert_consumed(self) -> None:
        """Raise if any recorded request was never replayed against."""
        remaining = set(self._records) - self._consumed
        if remaining:
            raise self._mismatch_error(f"{len(remaining)} unconsumed record(s) remain")
