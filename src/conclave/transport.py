"""Shared async HTTP transport: the single network boundary for conclave.

Every provider call -- regardless of adapter -- sends its request through
:func:`post_json`. Concentrating all network I/O here gives us exactly one place
to pool connections, one place to normalize timeout/connection failures into a
single internal error type, and one stable patch seam for transport-level tests
(patch ``conclave.transport.post_json``).

The transport is intentionally provider-agnostic: it knows nothing about auth
headers, model ids, or response shapes. Adapters build the request and parse the
response; the transport just moves bytes and reports HTTP status.
"""

from __future__ import annotations

from typing import Optional

import httpx

from .logging import get_logger

logger = get_logger("transport")

# One shared, lazily-created client so connections are pooled across calls
# within a process. httpx.AsyncClient is safe to share across concurrent tasks.
_client: Optional[httpx.AsyncClient] = None


class TransportError(Exception):
    """A network-level failure (timeout, connection refused, DNS, etc.).

    Raised by :func:`post_json` so :func:`conclave.providers.call_model` can turn
    it into a non-raising ``ModelAnswer.error``. The message is built from the
    exception type only -- never from request headers -- so it carries no secret.
    """


def _get_client() -> httpx.AsyncClient:
    """Return the process-wide pooled client, creating it on first use."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient()
    return _client


async def post_json(
    url: str,
    headers: dict[str, str],
    json_body: dict,
    timeout: float,
) -> tuple[int, object]:
    """POST a JSON body and return ``(status_code, parsed_body)``.

    Args:
        url: Fully-qualified endpoint URL built by the adapter.
        headers: Request headers built by the adapter (may carry the API key).
        json_body: The request payload to serialize as JSON.
        timeout: Per-call timeout in seconds (applied to the whole request).

    Returns:
        A ``(status, body)`` tuple. ``body`` is the decoded JSON object when the
        response is valid JSON, otherwise the raw response text (so adapters can
        surface a meaningful error for non-JSON error pages).

    Raises:
        TransportError: On any network-level failure (timeout, connection error,
            or other ``httpx.HTTPError``). The message names only the failure
            kind and never echoes the headers, so no key can leak.
    """
    client = _get_client()
    try:
        response = await client.post(
            url, headers=headers, json=json_body, timeout=timeout
        )
    except httpx.TimeoutException as exc:
        raise TransportError(f"request timed out after {timeout:.0f}s") from exc
    except httpx.HTTPError as exc:
        # Use the exception class name, not str(exc): httpx error strings can
        # include the request URL but never headers, yet we stay conservative.
        raise TransportError(f"network error: {type(exc).__name__}") from exc

    try:
        body: object = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


async def aclose() -> None:
    """Close the shared client. Optional; primarily for clean test teardown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
