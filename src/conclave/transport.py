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

from collections.abc import AsyncIterator

import httpx

from .logging import get_logger

logger = get_logger("transport")

# One shared, lazily-created client so connections are pooled across calls
# within a process. httpx.AsyncClient is safe to share across concurrent tasks.
_client: httpx.AsyncClient | None = None


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
        response = await client.post(url, headers=headers, json=json_body, timeout=timeout)
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


async def stream_sse(
    url: str,
    headers: dict[str, str],
    json_body: dict,
    timeout: float,
) -> AsyncIterator[tuple[str, str]]:
    """POST a JSON body and yield Server-Sent Events as ``(event, data)`` pairs.

    The streaming counterpart of :func:`post_json` and the single streaming
    network boundary for conclave (issue #7). It reuses the same pooled client
    and timeout plumbing, and -- like ``post_json`` -- knows nothing about auth
    headers or provider response shapes: it parses the SSE wire format and hands
    each event back to the adapter to interpret.

    SSE framing parsed here (the subset every supported vendor uses):

    * Events are separated by a blank line.
    * ``event: <name>`` sets the event name for the current event (Anthropic
      uses named events; OpenAI/Gemini do not, so ``event`` is ``""`` there).
    * ``data: <payload>`` lines are accumulated (multiple ``data:`` lines in one
      event are joined with ``\\n``, per the SSE spec).
    * Comment lines (starting ``:``) and other fields are ignored.

    A non-2xx status on the streaming response is surfaced as a
    :class:`TransportError` whose message includes the status and a bounded,
    decoded body snippet (the adapter wraps it as a ``ProviderError`` upstream).
    The body is read fully only on the error path; on success nothing is
    buffered beyond one line at a time.

    Args:
        url: Fully-qualified endpoint URL built by the adapter.
        headers: Request headers built by the adapter (may carry the API key).
        json_body: The request payload to serialize as JSON (already carrying
            the provider's stream-enabling flag).
        timeout: Per-call timeout in seconds (applied to the whole request).

    Yields:
        ``(event_name, data)`` pairs in arrival order. ``event_name`` is ``""``
        when the stream omits ``event:`` lines. ``data`` is the raw payload
        string (typically JSON, or the ``[DONE]`` sentinel for OpenAI-style
        streams); the adapter decodes it.

    Raises:
        TransportError: On any network-level failure (timeout, connection
            error) or a non-2xx streaming status. The message names only the
            failure kind / HTTP status and never echoes the headers.
    """
    client = _get_client()
    try:
        async with client.stream(
            "POST", url, headers=headers, json=json_body, timeout=timeout
        ) as response:
            if response.status_code < 200 or response.status_code >= 300:
                # Drain the error body so we can report a useful, bounded detail.
                # aread() is required before the response is consumed/closed.
                raw = await response.aread()
                detail = raw.decode("utf-8", "replace")[:500]
                raise TransportError(f"HTTP {response.status_code}: {detail}")

            event_name = ""
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                # A blank line terminates the current event -> dispatch it.
                if line == "":
                    if data_lines:
                        yield event_name, "\n".join(data_lines)
                    event_name = ""
                    data_lines = []
                    continue
                if line.startswith(":"):
                    # SSE comment / keep-alive ping; ignore.
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())
                # Any other field (id:, retry:, ...) is irrelevant here.

            # Flush a final event with no trailing blank line (some servers do
            # not emit the terminating newline).
            if data_lines:
                yield event_name, "\n".join(data_lines)
    except httpx.TimeoutException as exc:
        raise TransportError(f"request timed out after {timeout:.0f}s") from exc
    except httpx.HTTPError as exc:
        raise TransportError(f"network error: {type(exc).__name__}") from exc


async def aclose() -> None:
    """Close the shared client. Optional; primarily for clean test teardown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
