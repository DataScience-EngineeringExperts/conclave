"""Shared pytest fixtures and a LiteLLM mock harness.

The whole suite runs offline: ``litellm.acompletion`` is patched so no real
network or API keys are needed. A small fake response object mimics the parts of
the LiteLLM response that conclave reads (choices/message/content + usage).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

import pytest


@dataclass
class _FakeUsage:
    prompt_tokens: int = 5
    completion_tokens: int = 7
    total_tokens: int = 12


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


def make_response(text: str) -> _FakeResponse:
    """Build a fake LiteLLM-shaped response carrying ``text``."""
    return _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=text))],
        usage=_FakeUsage(),
    )


@pytest.fixture
def patch_acompletion(monkeypatch) -> Callable:
    """Return an installer that patches ``litellm.acompletion`` with a handler.

    Usage::

        def handler(model, messages, **kwargs):
            return make_response("hi")  # or raise to simulate failure
        patch_acompletion(handler)

    The handler may be sync (returning a response or raising); the patch wraps it
    in a coroutine so ``await litellm.acompletion(...)`` works.
    """
    import litellm

    def install(handler: Callable):
        async def fake_acompletion(*, model, messages, **kwargs):
            # Allow a tiny await so concurrency is genuinely exercised.
            await asyncio.sleep(0)
            return handler(model, messages, **kwargs)

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

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
    ):
        monkeypatch.delenv(var, raising=False)
