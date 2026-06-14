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

import logging
from collections.abc import AsyncIterator
from typing import NoReturn

import httpx

from .logging import get_logger

logger = get_logger("transport")

# One shared, lazily-created client so connections are pooled across calls
# within a process. httpx.AsyncClient is safe to share across concurrent tasks.
_client: httpx.AsyncClient | None = None

# --------------------------------------------------------------------------- #
# httpx/httpcore debug-logging leak guard (key-leak audit, vector 5)
# --------------------------------------------------------------------------- #
#
# SECURITY-CRITICAL, OUT-OF-BAND OF redact(): httpx and httpcore have their own
# `logging` loggers. At DEBUG level httpcore logs the full request headers --
# which include ``Authorization: Bearer <key>`` and ``x-api-key: <key>`` -- to
# whatever handler the host application configured. conclave's ``redact()`` only
# scrubs the error/diagnostic strings *it* produces; it cannot reach inside the
# third-party transport loggers. So a consumer that turns on transport DEBUG
# logging (``logging.basicConfig(level=logging.DEBUG)`` app-wide, or explicitly
# raising the httpx/httpcore loggers) would leak live keys to their own logs,
# entirely bypassing every redaction conclave performs.
#
# We cannot (and should not) globally silence another library's logging for the
# whole process -- that would be surprising and could hide legitimate debugging.
# Instead we expose an explicit, opt-in guard a security-conscious library
# consumer can call once at startup. It installs a filter that drops any
# httpx/httpcore log record at DEBUG severity (the only level that emits header
# content), while leaving INFO+ records untouched. See SECURITY.md "Threat
# model" for the documented trust boundary and accepted limitation.
_TRANSPORT_LOGGER_NAMES = ("httpx", "httpcore")
_GUARD_INSTALLED = False


class _NoDebugHeadersFilter(logging.Filter):
    """Drop DEBUG-level records from a transport logger (where headers appear).

    httpcore emits request/response headers only at ``DEBUG``; INFO and above
    carry no header content. Filtering exactly the DEBUG band stops the header
    leak without suppressing useful higher-severity transport diagnostics.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Returning False discards the record before any handler formats it.
        return record.levelno > logging.DEBUG


def guard_transport_logging() -> None:
    """Block httpx/httpcore DEBUG logging so auth headers can never be logged.

    **Opt-in, library-mode key-leak hardening.** httpx/httpcore log full request
    headers (including the ``Authorization``/``x-api-key`` value) at ``DEBUG``.
    That path is outside :func:`conclave.adapters.base.redact`'s reach, so a host
    application that enables transport DEBUG logging would leak live keys to its
    own log sinks. Calling this once at startup installs a logging filter on the
    ``httpx`` and ``httpcore`` loggers that discards their DEBUG records, closing
    the leak while leaving INFO+ diagnostics intact. Idempotent.

    This is intentionally **not** called automatically: silently reconfiguring a
    third-party library's logging for the whole process would be surprising and
    could mask legitimate debugging. A consumer that handles real keys and also
    runs verbose transport logging should call it explicitly. The default,
    documented guidance (SECURITY.md) is simply: do not enable httpx/httpcore
    DEBUG logging in a process that holds real provider keys.
    """
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    for name in _TRANSPORT_LOGGER_NAMES:
        logging.getLogger(name).addFilter(_NoDebugHeadersFilter())
    _GUARD_INSTALLED = True


class TransportError(Exception):
    """A network-level failure (timeout, connection refused, DNS, etc.).

    Raised by :func:`post_json` so :func:`conclave.providers.call_model` can turn
    it into a non-raising ``ModelAnswer.error``. The message is built from the
    exception type only -- never from request headers -- so it carries no secret.

    KEY-LEAK NOTE (audit RANK 1/5): the raise sites route through
    :func:`_raise_transport_error` (``raise ... from None``) and a boundary clear,
    so the surfaced TransportError retains **no** reference to the underlying httpx
    exception -- not as ``__cause__``, not as ``__context__``. That httpx
    exception's ``.request.headers`` carries the live ``Authorization``/``x-api-key``
    value; had it survived it would leak the key one cause-chain hop away under
    ``traceback.format_exception``, ``logging.exception``, a ``repr`` of the cause
    chain, or a direct ``err.__context__`` attribute walk. Dropping the chain is
    deliberate -- the message already names the failure kind, so no diagnostic
    value is lost.
    """


def _raise_transport_error(message: str) -> NoReturn:
    """Raise a :class:`TransportError` that retains no link to the httpx exception.

    KEY-LEAK NOTE (audit RANK 1/5). The httpx exception active when this is called
    carries a live ``.request`` whose ``.headers`` hold the ``Authorization`` /
    ``x-api-key`` value. We must not let it survive on the surfaced TransportError:

    * ``raise ... from None`` sets ``__cause__ = None`` and
      ``__suppress_context__ = True`` -- enough that ``traceback.format_exception``,
      ``logging.exception``, and a cause-chain ``repr`` render neither the httpx
      exception nor its headers (those formatters honor ``__suppress_context__``).
    * Python's implicit-context machinery still points ``__context__`` at the
      active httpx exception at ``raise`` time, so a *direct attribute walk*
      (``err.__context__.request.headers``) could still reach the key. We therefore
      build the error and raise with ``from None`` here; the **caller** clears
      ``__context__`` at a boundary where no exception is being handled (so Python
      cannot re-chain it), making even a direct walk key-free.

    Centralizing the raise keeps the clear-and-raise contract identical at all four
    transport raise sites. The message names only the failure kind, so dropping the
    chain loses no diagnostic value.
    """
    raise TransportError(message) from None


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
            kind and never echoes the headers, so no key can leak. The underlying
            httpx exception is deliberately dropped from the cause chain
            (``__cause__`` and ``__context__`` both cleared) so its header-bearing
            ``.request`` cannot leak the key via the surfaced error's traceback,
            cause-chain repr, or a direct attribute walk (audit RANK 1/5).
    """
    client = _get_client()
    # Inner try maps httpx failures to TransportError via _raise_transport_error
    # (which raises ``from None``); the outer try clears ``__context__`` at a
    # boundary where no httpx exception is active, so even a direct attribute walk
    # finds no header-bearing httpx exception (key-leak audit RANK 1/5). See
    # _raise_transport_error for the full rationale.
    try:
        try:
            response = await client.post(url, headers=headers, json=json_body, timeout=timeout)
        except httpx.TimeoutException:
            _raise_transport_error(f"request timed out after {timeout:.0f}s")
        except httpx.HTTPError as exc:
            # Use the exception class NAME, not str(exc): httpx error strings can
            # include the request URL but never headers, yet we stay conservative.
            _raise_transport_error(f"network error: {type(exc).__name__}")
    except TransportError as err:
        # Boundary clear: no httpx exception is being handled here, so nulling
        # ``__context__`` sticks (Python will not re-chain) and re-raising
        # ``from None`` keeps ``__cause__``/``__suppress_context__`` clean.
        err.__context__ = None
        raise err from None

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
            failure kind / HTTP status and never echoes the headers. The
            underlying httpx exception is dropped from the cause chain
            (``__cause__`` and ``__context__`` both cleared) so its header-bearing
            ``.request`` cannot leak the key via the surfaced error's traceback,
            cause-chain repr, or a direct attribute walk (audit RANK 1/5).
    """
    client = _get_client()
    # Inner try maps httpx failures to TransportError via _raise_transport_error
    # (raises ``from None``); the outer try clears ``__context__`` at a boundary
    # where no httpx exception is active, so even a direct attribute walk finds no
    # header-bearing httpx exception (key-leak audit RANK 1/5). The intentional
    # ``HTTP <status>: <detail>`` error (not chained from httpx) also passes the
    # boundary harmlessly -- it carries no httpx context to clear.
    try:
        try:
            async with client.stream(
                "POST", url, headers=headers, json=json_body, timeout=timeout
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    # Drain the error body so we can report a useful, bounded detail.
                    # aread() is required before the response is consumed/closed.
                    raw = await response.aread()
                    detail = raw.decode("utf-8", "replace")[:500]
                    # KEY-LEAK NOTE (audit vector 2/4): this raw provider body may echo
                    # request fragments. It is intentionally NOT redacted here -- the
                    # transport stays provider-agnostic and never imports redact(). The
                    # single redaction boundary for the streaming path is
                    # conclave.providers.call_model_stream, which wraps every
                    # TransportError/ProviderError message in redact() before it lands
                    # on ModelAnswer.error or is logged. No streamed text delta is
                    # emitted on this path (deltas carry only parsed answer content),
                    # so the only surface for this string is that redacted final answer.
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
        except httpx.TimeoutException:
            # Map to TransportError with the chain dropped (audit RANK 1/5). The
            # streaming httpx exception also carries ``.request.headers`` with the
            # live auth value; _raise_transport_error raises ``from None``.
            _raise_transport_error(f"request timed out after {timeout:.0f}s")
        except httpx.HTTPError as exc:
            # Drop the httpx exception from the cause chain so its header-bearing
            # ``.request`` cannot leak the key (audit RANK 1/5).
            _raise_transport_error(f"network error: {type(exc).__name__}")
    except TransportError as err:
        # Boundary clear: no httpx exception is active here, so nulling
        # ``__context__`` sticks (Python will not re-chain) and re-raising
        # ``from None`` keeps ``__cause__``/``__suppress_context__`` clean.
        err.__context__ = None
        raise err from None


async def aclose() -> None:
    """Close the shared client. Optional; primarily for clean test teardown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
