"""Shared pytest fixtures and the offline call-model mock harness.

The whole suite runs offline. Since conclave now owns its provider highway (no
LiteLLM), the single choke point is :func:`conclave.providers.call_model`, which
``conclave.council`` imports as ``call_model``. The ``patch_call_model`` fixture
replaces the name that ``Council.fan_out`` resolves -- ``conclave.council.call_model``
-- with a fake that runs a user-supplied handler and wraps its result in a real
:class:`conclave.models.ModelAnswer`.

A handler has signature ``(model_id, messages) -> _FakeResult`` (via
:func:`make_response`) or it may ``raise`` to simulate a provider failure; the
fixture turns a raise into a ``ModelAnswer.error`` exactly as the real call path
does, and a sleep inside the handler genuinely exercises gather concurrency.

Transport-level tests (in ``test_providers.py``) instead patch
``conclave.transport.post_json`` to exercise the real ``call_model`` end to end.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from conclave.models import ModelAnswer, TokenUsage


@dataclass
class _FakeResult:
    """What a test handler returns: the text and its token usage."""

    text: str
    usage: TokenUsage


def make_response(text: str) -> _FakeResult:
    """Build a fake successful result carrying ``text`` and stock token usage."""
    return _FakeResult(
        text=text,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


@pytest.fixture
def patch_call_model(monkeypatch) -> Callable:
    """Return an installer that patches ``conclave.council.call_model``.

    Usage::

        def handler(model_id, messages, **kwargs):
            return make_response("hi")  # or raise to simulate failure
        patch_call_model(handler)

    The handler is sync (returns a ``_FakeResult`` or raises). The patch wraps it
    so ``await call_model(...)`` works, builds a real ``ModelAnswer`` carrying the
    correct ``name``/``model_id``, and converts a raise into ``ModelAnswer.error``
    -- mirroring the production contract.
    """
    import conclave.council as council_mod

    def install(handler: Callable):
        async def fake_call_model(name, model_id, messages, *, temperature=0.7, timeout=120.0):
            # A tiny await so concurrency is genuinely exercised by gather.
            await asyncio.sleep(0)
            try:
                result = handler(model_id, messages)
            except Exception as exc:  # noqa: BLE001 -- mirror call_model's contract
                return ModelAnswer(
                    name=name,
                    model_id=model_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return ModelAnswer(
                name=name,
                model_id=model_id,
                answer=result.text,
                usage=result.usage,
            )

        monkeypatch.setattr(council_mod, "call_model", fake_call_model)

    return install


@pytest.fixture
def clear_keys(monkeypatch) -> None:
    """Remove all provider env vars so 'missing key' paths are deterministic."""
    for var in (
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
