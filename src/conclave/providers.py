"""Single async provider-call path built on LiteLLM's ``acompletion``.

Every model call in conclave -- both council members and the synthesizer --
flows through :func:`call_model`. It captures latency, token usage, and any
error so that one provider failing (network/auth/rate-limit) never aborts the
run. LiteLLM resolves the actual API key from the environment per provider.
"""

from __future__ import annotations

import time
from typing import Optional

import litellm

from .logging import get_logger
from .models import ModelAnswer, TokenUsage

logger = get_logger("providers")

# Don't let LiteLLM mutate process-wide retry/telemetry behavior unexpectedly.
litellm.drop_params = True
litellm.telemetry = False


def _extract_usage(response: object) -> Optional[TokenUsage]:
    """Pull token usage from a LiteLLM response if present."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _extract_text(response: object) -> Optional[str]:
    """Pull the assistant message text from a LiteLLM response."""
    try:
        return response.choices[0].message.content  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError):
        return None


async def call_model(
    name: str,
    model_id: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> ModelAnswer:
    """Call a single model and return a structured :class:`ModelAnswer`.

    This coroutine never raises for provider-side failures; instead it records
    the error on the returned answer so callers can collect partial results.

    Args:
        name: Friendly council member name.
        model_id: Resolved LiteLLM model id.
        messages: OpenAI-style message list.
        temperature: Sampling temperature.
        timeout: Per-call timeout in seconds.

    Returns:
        A ``ModelAnswer`` with either ``answer`` populated or ``error`` set.
    """
    started = time.perf_counter()
    try:
        response = await litellm.acompletion(
            model=model_id,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        latency = time.perf_counter() - started
        text = _extract_text(response)
        if text is None:
            logger.warning("%s (%s) returned no content", name, model_id)
            return ModelAnswer(
                name=name,
                model_id=model_id,
                latency_s=latency,
                error="empty response (no message content)",
            )
        logger.info("%s (%s) ok in %.2fs", name, model_id, latency)
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer=text,
            latency_s=latency,
            usage=_extract_usage(response),
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: never kill the run
        latency = time.perf_counter() - started
        logger.warning("%s (%s) failed: %s", name, model_id, exc)
        return ModelAnswer(
            name=name,
            model_id=model_id,
            latency_s=latency,
            error=f"{type(exc).__name__}: {exc}",
        )
