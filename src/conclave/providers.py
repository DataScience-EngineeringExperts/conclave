"""Single async provider-call path over the owned httpx-based highway.

Every model call in conclave -- both council members and the synthesizer -- flows
through :func:`call_model`. It resolves the right provider adapter, reads the API
key from the environment BY NAME at call time (never storing or logging it),
builds the request, sends it through the one network boundary
(:func:`conclave.transport.post_json`), parses the response, and captures latency
plus token usage. Any provider/network/auth failure becomes ``ModelAnswer.error``
so one model failing never aborts the run.

This module replaces the former LiteLLM dependency. Provider wire formats live in
:mod:`conclave.adapters`; the network boundary lives in :mod:`conclave.transport`.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from . import transport
from .adapters import ProviderError, resolve_adapter
from .adapters.base import ProviderAdapter, redact
from .config import ConclaveConfig, load_config
from .logging import get_logger
from .manifest import ProviderExecutionReceipt
from .models import ModelAnswer, TokenUsage
from .registry import provider_prefix
from .transport import TransportError

logger = get_logger("providers")


def receipt_from_answer(
    answer: ModelAnswer, *, temperature: float, timeout: float
) -> ProviderExecutionReceipt:
    """Map a collected :class:`ModelAnswer` to a :class:`ProviderExecutionReceipt`.

    This is CAC-04's ``providers.py`` wiring: it produces one per-member receipt
    without changing :func:`call_model`'s return type or its never-raises
    contract. The council builds each receipt from an already-collected answer
    plus the council's known generation settings (the same temperature/timeout it
    threaded into :func:`call_model`), so the provider hot path is untouched and
    the manifest internals stay CAC-04's concern (the CAC-01 ticket explicitly
    assigns them here).

    The ``provider`` prefix is derived from ``answer.model_id`` via
    :func:`conclave.registry.provider_prefix`. The ``error`` is re-run through
    :func:`redact` belt-and-suspenders: it is already redacted upstream (every
    :class:`ModelAnswer.error` conclave produces is scrubbed), and ``redact`` is
    idempotent and safe on clean text, so this re-application cannot leak a key
    and cannot raise on the happy path -- preserving the redaction invariant even
    on an unexpected error string.

    Args:
        answer: The collected member answer (success or failure).
        temperature: The sampling temperature the council used for the call.
        timeout: The per-call timeout (seconds) the council used.

    Returns:
        A :class:`ProviderExecutionReceipt` for this member.
    """
    return ProviderExecutionReceipt(
        name=answer.name,
        provider=provider_prefix(answer.model_id),
        model_id=answer.model_id,
        generation_settings={"temperature": temperature, "timeout": timeout},
        latency_ms=answer.latency_ms,
        usage=answer.usage,
        error=redact(answer.error) if answer.error else None,
    )


def _resolve_key(adapter: ProviderAdapter) -> str | None:
    """Read the active key VALUE for an adapter from the environment, or None.

    Walks the adapter's candidate env var names in order and returns the first
    non-empty value. The value is read here, used only to build the request, and
    never stored on any object, logged, or serialized.
    """
    for var in adapter.env_vars:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


async def call_model(
    name: str,
    model_id: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    timeout: float = 120.0,
    config: ConclaveConfig | None = None,
) -> ModelAnswer:
    """Call a single model and return a structured :class:`ModelAnswer`.

    This coroutine never raises for provider-side failures; instead it records the
    error on the returned answer so callers can collect partial results. Errors
    are scrubbed of anything key-shaped before being surfaced.

    Args:
        name: Friendly council member name.
        model_id: Resolved provider model id (e.g. ``"xai/grok-4.3"``).
        messages: OpenAI-style message list.
        temperature: Sampling temperature.
        timeout: Per-call timeout in seconds.
        config: Pre-resolved config to use for adapter resolution (custom
            ``endpoints:``). A caller that already holds the config -- e.g.
            ``Council`` -- threads it in so this hot path does not re-read it.
            When ``None`` (a standalone call) the config is resolved via the
            memoized :func:`conclave.config.load_config`, so even repeated
            standalone calls avoid redundant disk reads (issue #15).

    Returns:
        A ``ModelAnswer`` with either ``answer`` populated or ``error`` set.
    """
    started = time.perf_counter()

    # Resolve the adapter (config-aware for user-declared custom endpoints). A
    # bad/unknown provider id surfaces as a clean, non-raising error. Prefer the
    # injected config; fall back to the memoized loader only when called standalone.
    try:
        resolved_config = config if config is not None else load_config()
        adapter = resolve_adapter(model_id, resolved_config)
    except ProviderError as exc:
        latency = time.perf_counter() - started
        logger.warning("%s (%s) unresolved: %s", name, model_id, exc)
        return ModelAnswer(name=name, model_id=model_id, latency_s=latency, error=str(exc))

    api_key = _resolve_key(adapter)
    if api_key is None:
        latency = time.perf_counter() - started
        names = " or ".join(adapter.env_vars) or "(none)"
        msg = f"no API key in environment (set {names})"
        logger.warning("%s (%s) %s", name, model_id, msg)
        return ModelAnswer(name=name, model_id=model_id, latency_s=latency, error=msg)

    try:
        url, headers, body = adapter.build_request(
            model_id, messages, temperature, timeout, api_key
        )
        status, payload = await transport.post_json(url, headers, body, timeout)
        text, usage = adapter.parse_response(status, payload)
        latency = time.perf_counter() - started
        logger.info("%s (%s) ok in %.2fs", name, model_id, latency)
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer=text,
            latency_s=latency,
            usage=usage,
        )
    except (ProviderError, TransportError) as exc:
        latency = time.perf_counter() - started
        # ProviderError is pre-redacted; redact again belt-and-suspenders for the
        # transport message and any composed string.
        message = redact(str(exc))
        logger.warning("%s (%s) failed: %s", name, model_id, message)
        return ModelAnswer(name=name, model_id=model_id, latency_s=latency, error=message)
    except Exception as exc:  # noqa: BLE001 -- never let an unexpected raise kill the run
        latency = time.perf_counter() - started
        message = redact(f"{type(exc).__name__}: {exc}")
        logger.warning("%s (%s) unexpected error: %s", name, model_id, message)
        return ModelAnswer(name=name, model_id=model_id, latency_s=latency, error=message)


def _merge_usage(acc: TokenUsage | None, frame: TokenUsage | None) -> TokenUsage | None:
    """Merge a per-frame usage accounting into the running total, field-wise.

    Providers split usage across frames differently: OpenAI sends one final
    chunk with all three counts; Anthropic sends ``prompt_tokens`` on
    ``message_start`` and *cumulative* ``completion_tokens`` on each
    ``message_delta``; Gemini repeats a cumulative ``usageMetadata`` per chunk.
    Taking the last non-zero value per field (and recomputing ``total`` when the
    provider did not send one) yields the same final accounting the buffered
    ``parse_response`` would have produced, regardless of which scheme applies.
    """
    if frame is None:
        return acc
    if acc is None:
        acc = TokenUsage()
    prompt = frame.prompt_tokens or acc.prompt_tokens
    completion = frame.completion_tokens or acc.completion_tokens
    # An explicit combined total from a frame wins (OpenAI sends one). Otherwise
    # derive it from the merged components -- Anthropic/Gemini split prompt and
    # completion across frames and never send a true sum, so a stale per-component
    # value must not masquerade as the total.
    if frame.total_tokens:
        total = frame.total_tokens
    elif prompt or completion:
        total = prompt + completion
    else:
        total = acc.total_tokens
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


async def call_model_stream(
    name: str,
    model_id: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    timeout: float = 120.0,
    config: ConclaveConfig | None = None,
) -> AsyncIterator[str | ModelAnswer]:
    """Stream a single model's answer, yielding text deltas then a final answer.

    The streaming counterpart of :func:`call_model` (issue #7). It yields zero or
    more ``str`` text chunks in arrival order, then **exactly one**
    :class:`ModelAnswer` as the final item -- the fully-assembled result whose
    ``answer`` equals the concatenation of every yielded chunk and whose
    ``usage`` matches what the buffered path would produce. A consumer
    distinguishes the two by type::

        async for item in call_model_stream(...):
            if isinstance(item, str):
                render(item)          # live token
            else:
                final = item          # ModelAnswer (last item)

    Never-raises contract (identical to :func:`call_model`): an unknown
    provider, a missing key, a model that cannot stream, or any mid-stream error
    is captured on the final :class:`ModelAnswer.error`; **any partial text seen
    before the failure is preserved** both in the chunks already yielded and in
    the final answer's ``answer`` field. This coroutine never propagates a
    provider/network exception.

    A provider whose adapter reports ``supports_streaming = False`` is served by
    falling back to the buffered :func:`call_model`, whose full answer text is
    emitted as a single chunk -- so a non-streaming provider degrades to a
    one-shot render rather than an error.

    Args mirror :func:`call_model`.

    Yields:
        ``str`` text deltas, then a final :class:`ModelAnswer`.
    """
    started = time.perf_counter()

    try:
        resolved_config = config if config is not None else load_config()
        adapter = resolve_adapter(model_id, resolved_config)
    except ProviderError as exc:
        latency = time.perf_counter() - started
        logger.warning("%s (%s) unresolved: %s", name, model_id, exc)
        yield ModelAnswer(name=name, model_id=model_id, latency_s=latency, error=str(exc))
        return

    # Providers without a streaming path degrade to a single-chunk render so the
    # caller's streaming code path still works uniformly.
    if not getattr(adapter, "supports_streaming", False):
        answer = await call_model(
            name,
            model_id,
            messages,
            temperature=temperature,
            timeout=timeout,
            config=resolved_config,
        )
        if answer.answer:
            yield answer.answer
        yield answer
        return

    api_key = _resolve_key(adapter)
    if api_key is None:
        latency = time.perf_counter() - started
        names = " or ".join(adapter.env_vars) or "(none)"
        msg = f"no API key in environment (set {names})"
        logger.warning("%s (%s) %s", name, model_id, msg)
        yield ModelAnswer(name=name, model_id=model_id, latency_s=latency, error=msg)
        return

    parts: list[str] = []
    usage: TokenUsage | None = None
    try:
        url, headers, body = adapter.stream_request(
            model_id, messages, temperature, timeout, api_key
        )
        async for event, data in transport.stream_sse(url, headers, body, timeout):
            delta = adapter.parse_sse_event(event, data)
            if delta.text:
                parts.append(delta.text)
                yield delta.text
            usage = _merge_usage(usage, delta.usage)
            if delta.done:
                break

        text = "".join(parts)
        latency = time.perf_counter() - started
        if not text:
            # A stream that closed without any text is a failure, mirroring the
            # buffered adapters' "empty response" ProviderError.
            yield ModelAnswer(
                name=name,
                model_id=model_id,
                latency_s=latency,
                error=f"{adapter.prefix}: empty response (no streamed content)",
            )
            return
        logger.info("%s (%s) streamed ok in %.2fs", name, model_id, latency)
        yield ModelAnswer(
            name=name,
            model_id=model_id,
            answer=text,
            latency_s=latency,
            usage=usage,
        )
    except (ProviderError, TransportError) as exc:
        # Mid-stream failure: preserve any partial text already collected, set
        # the (redacted) error, never raise out of the call path.
        latency = time.perf_counter() - started
        message = redact(str(exc))
        logger.warning("%s (%s) stream failed: %s", name, model_id, message)
        yield ModelAnswer(
            name=name,
            model_id=model_id,
            answer="".join(parts) or None,
            latency_s=latency,
            usage=usage,
            error=message,
        )
    except Exception as exc:  # noqa: BLE001 -- never let an unexpected raise kill the run
        latency = time.perf_counter() - started
        message = redact(f"{type(exc).__name__}: {exc}")
        logger.warning("%s (%s) unexpected stream error: %s", name, model_id, message)
        yield ModelAnswer(
            name=name,
            model_id=model_id,
            answer="".join(parts) or None,
            latency_s=latency,
            usage=usage,
            error=message,
        )
