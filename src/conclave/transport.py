"""Shared async HTTP transport: the single network boundary for conclave.

Every provider call -- regardless of adapter -- sends its request through
:func:`post_json`. Concentrating all network I/O here gives us exactly one place
to pool connections, one place to normalize timeout/connection failures into a
single internal error type, and one stable patch seam for transport-level tests
(patch ``conclave.transport.post_json``).

The transport is intentionally provider-agnostic: it knows nothing about auth
headers, model ids, or response shapes. Adapters build the request and parse the
response; the transport just moves bytes and reports HTTP status.

**The record/replay seam (DSE-1517 Task 2).** ``post_json`` and ``stream_sse``
consult one :class:`contextvars.ContextVar` before doing anything else. With no
context set, both functions are byte-for-byte their pre-DSE-1517 selves --
nothing about this seam changes the default (no-context) code path. Two
context types share the var: :class:`RecordingContext` (live keys, live
network; wraps ``post_json`` to capture a sanitized tape) and
:class:`ReplayContext` (offline: no keys, no network; ``post_json`` is fully
overridden and ``stream_sse`` refuses outright). A process-global refcount,
``_OFFLINE_STRICT``, backstops the per-task ``ContextVar`` for anything that
escapes normal ``asyncio`` task-context inheritance (e.g. a
``loop.run_in_executor`` hop onto a plain thread), so a strict replay can never
quietly turn into a live call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar, NoReturn

import httpx

from .logging import get_logger
from .models import FailureCategory, categorize_http_status
from .tape import PostJson

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

    ``category`` is typed at the raise site (DSE-1512, :data:`conclave.models.FailureCategory`)
    so a caller can decide "retry a different provider or stop" from a typed
    attribute instead of substring-matching the message.

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

    ``http_status`` (DSE-1512) carries the HTTP status when the failure came from
    a non-2xx response (the streaming path); it stays ``None`` for a network/
    timeout failure, which never produced a response.
    """

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory = "transport",
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category: FailureCategory = category
        self.http_status: int | None = http_status


def _raise_transport_error(message: str, category: FailureCategory = "transport") -> NoReturn:
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
    raise TransportError(message, category=category) from None


@dataclass(frozen=True)
class RecordingContext:
    """Live-key, live-network transport override active during ``ask --record``.

    ``post_json`` is wrapped (typically by a
    :class:`conclave.tape.RecordingTransport`) so every call is captured onto a
    sanitized tape; from the adapter's point of view, behavior is identical to
    an uncontexted call. ``offline`` is ``False`` -- :func:`_resolve_key` always
    falls through to the real environment lookup under this context (never the
    replay sentinel: this is the F1 guard), and ``stream_sse`` is never
    intercepted (``--record`` only drives buffered modes, so the streaming path
    is never reached with a ``RecordingContext`` active, but the contract is
    that it would pass straight through live if it were).
    """

    post_json: PostJson
    offline: ClassVar[bool] = False


@dataclass(frozen=True)
class ReplayContext:
    """Offline transport override active only inside ``conclave replay``.

    No network call and no environment read ever happens under this context.
    ``post_json`` is fully overridden by ``self.post_json`` (typically a
    :class:`conclave.tape.ReplayingTransport`) before ``_get_client()`` is ever
    reached, ``stream_sse`` refuses outright, and
    :func:`conclave.providers._resolve_key` returns ``key_sentinel`` instead of
    reading an env var. The sentinel is placed into request headers by the
    adapter exactly like a real key would be, but the replaying transport
    ignores headers entirely when matching a recorded exchange -- it hashes
    only ``url`` and ``body`` -- so the sentinel never leaves the process. This
    branch is reachable only when ``ctx.offline`` is true; a
    :class:`RecordingContext` never triggers it.
    """

    post_json: PostJson
    key_sentinel: str = "replay"
    offline: ClassVar[bool] = True


# One seam, two context types: set/read by ``recording``/``replaying`` below and
# consulted at the top of ``post_json``/``stream_sse``. ``default=None`` is the
# ordinary, no-context production path -- every call site checks ``is not None``
# before touching ``.offline`` or ``.post_json``.
_TRANSPORT: ContextVar[RecordingContext | ReplayContext | None] = ContextVar(
    "conclave_transport", default=None
)

# Process-global refcount (not per-task): > 0 means "no live call anywhere in
# this process", enforced independently of the ContextVar above so a call that
# escapes normal asyncio task-context inheritance -- e.g. a
# ``loop.run_in_executor`` hop onto a plain OS thread, which does not inherit
# contextvars state -- still cannot reach the network during a strict replay.
_OFFLINE_STRICT = 0


def transport_context() -> RecordingContext | ReplayContext | None:
    """Return the transport context active in the current asyncio task, if any."""
    return _TRANSPORT.get()


@contextmanager
def recording(context: RecordingContext) -> Iterator[None]:
    """Activate ``context`` as the live-key, live-network transport override.

    Scoped via :class:`contextvars.ContextVar`: sibling ``asyncio`` tasks
    created before this context manager is entered do not see the override
    (task isolation), and the previous value is restored on exit even if the
    wrapped run raises.
    """
    token = _TRANSPORT.set(context)
    try:
        yield
    finally:
        _TRANSPORT.reset(token)


@contextmanager
def replaying(context: ReplayContext, *, strict: bool = True) -> Iterator[None]:
    """Activate ``context`` as the offline transport override.

    Args:
        context: The :class:`ReplayContext` to activate for the current task.
        strict: When ``True`` (the default -- what the ``conclave replay`` CLI
            always uses), also increments the process-global
            ``_OFFLINE_STRICT`` refcount for the duration of the context, in a
            ``try``/``finally`` so it is decremented even if the wrapped run
            raises. That refcount backstops the per-task ``ContextVar`` for
            anything that escapes normal task-context inheritance. ``False``
            exists for the embedded/concurrent-replay case
            (``replay_bundle(..., strict=False)``) where other live traffic may
            legitimately share the process; using it is the caller's
            responsibility.
    """
    token = _TRANSPORT.set(context)
    global _OFFLINE_STRICT
    if strict:
        _OFFLINE_STRICT += 1
    try:
        yield
    finally:
        if strict:
            _OFFLINE_STRICT -= 1
        _TRANSPORT.reset(token)


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
            kind and never echoes the headers, so no key can leak. ``category``
            is ``"timeout"`` for a timeout and ``"transport"`` for any other
            ``httpx.HTTPError`` (DSE-1512). The underlying
            httpx exception is deliberately dropped from the cause chain
            (``__cause__`` and ``__context__`` both cleared) so its header-bearing
            ``.request`` cannot leak the key via the surfaced error's traceback,
            cause-chain repr, or a direct attribute walk (audit RANK 1/5).
            Also raised (category ``"unexpected"``) when a strict offline
            replay is active and no context intercepted this call (DSE-1517).

    Note:
        The record/replay seam (DSE-1517) is checked FIRST, before
        ``_get_client()`` is ever called. With no context active and the
        strict backstop at zero, this is byte-for-byte the pre-DSE-1517
        function body.
    """
    ctx = _TRANSPORT.get()
    if ctx is not None:
        return await ctx.post_json(url, headers, json_body, timeout)
    if _OFFLINE_STRICT > 0:
        raise TransportError(
            "live call attempted during a strict offline replay", category="unexpected"
        )

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
            _raise_transport_error(f"request timed out after {timeout:.0f}s", "timeout")
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
            failure kind / HTTP status and never echoes the headers.
            ``category`` (DSE-1512) is ``"timeout"`` for a timeout,
            ``"transport"`` for any other network error, and
            :func:`conclave.models.categorize_http_status` of the status for a
            non-2xx response. The
            underlying httpx exception is dropped from the cause chain
            (``__cause__`` and ``__context__`` both cleared) so its header-bearing
            ``.request`` cannot leak the key via the surfaced error's traceback,
            cause-chain repr, or a direct attribute walk (audit RANK 1/5).
            Also raised (category ``"unexpected"``) under an offline
            :class:`ReplayContext` or a strict offline replay: streaming has no
            recorded override, so it refuses outright rather than reaching the
            network (DSE-1517). A :class:`RecordingContext` does NOT trigger
            this -- ``--record`` only drives buffered modes, so this path is
            never exercised while recording, but the contract is that it would
            stream live if it were.

    Note:
        The record/replay seam (DSE-1517) is checked FIRST, before
        ``_get_client()`` is ever called. With no context active and the
        strict backstop at zero, this is byte-for-byte the pre-DSE-1517
        function body.
    """
    ctx = _TRANSPORT.get()
    if (ctx is not None and ctx.offline) or _OFFLINE_STRICT > 0:
        raise TransportError(
            "live call attempted during a strict offline replay", category="unexpected"
        )

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
                    raise TransportError(
                        f"HTTP {response.status_code}: {detail}",
                        category=categorize_http_status(response.status_code),
                        http_status=response.status_code,
                    )

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
            _raise_transport_error(f"request timed out after {timeout:.0f}s", "timeout")
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
