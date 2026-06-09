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

from . import transport
from .adapters import ProviderError, resolve_adapter
from .adapters.base import ProviderAdapter, redact
from .config import ConclaveConfig, load_config
from .logging import get_logger
from .models import ModelAnswer
from .transport import TransportError

logger = get_logger("providers")


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
