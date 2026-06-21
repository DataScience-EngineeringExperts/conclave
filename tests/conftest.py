"""Shared pytest fixtures and the offline call-model mock harness.

The whole suite runs offline. Since conclave now owns its provider highway (no
LiteLLM), the single choke point is :func:`conclave.providers.call_model`. Two
modules import that function as a local name, so there are TWO seams to patch:

* the **council seam** -- ``conclave.council.call_model``, the name
  ``Council.fan_out`` resolves for every member call; and
* the **verdict seam** -- ``conclave.verdict_synthesis.call_model``, the name
  the default-on verdict-extraction step (CAC-05, wired by CAC-06) resolves for
  its synthesizer call.

Because verdict extraction is default-on (the verdict is the product), EVERY
``Council.ask`` now triggers the verdict seam as well as the council seam. The
``patch_call_model`` fixture therefore patches BOTH names so no test touches the
network, and it does so transparently: an existing test that calls
``patch_call_model(handler)`` is unchanged and is now network-safe on both seams.

A handler has signature ``(model_id, messages) -> _FakeResult`` (via
:func:`make_response`) or it may ``raise`` to simulate a provider failure; the
fixture turns a raise into a ``ModelAnswer.error`` exactly as the real call path
does, and a sleep inside the handler genuinely exercises gather concurrency.

**Driving the verdict seam from a handler.** The SAME ``handler`` runs on both
seams, so a handler that returns prose for everything is fine: that prose fails
the verdict-extraction JSON parse, extraction degrades gracefully to
``verdict=None`` (reason "verdict extraction failed schema validation"), and the
member-facing assertions are unaffected. A test that WANTS a real verdict can
branch on the messages: the verdict-extraction call's system message starts with
``"You are the verdict extractor"`` (see
``conclave.verdict_synthesis._EXTRACTION_SYSTEM``), so the handler can return the
extraction JSON for that call and prose otherwise. ``test_council_verdict.py``
shows this pattern.

Transport-level tests (in ``test_providers.py``) instead patch
``conclave.transport.post_json`` to exercise the real ``call_model`` end to end.
"""

from __future__ import annotations

import asyncio
import logging
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


def _make_fake_call_model(handler: Callable) -> Callable:
    """Wrap a sync test ``handler`` as an async ``call_model`` replacement.

    The returned coroutine has a permissive keyword signature
    (``*, temperature=..., timeout=..., config=None, **kwargs``) so it stands in
    for ``call_model`` on BOTH seams: the council seam calls it with
    ``temperature``/``timeout`` keywords, while the verdict seam calls it with a
    ``config`` keyword and no temperature/timeout. The handler itself only ever
    sees ``(model_id, messages)``; it runs identically on both seams, so a handler
    can branch on ``messages`` to drive the verdict path (see module docstring).
    A raise from the handler becomes a ``ModelAnswer.error``, mirroring the real
    contract.
    """

    async def fake_call_model(
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None, **kwargs
    ):
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

    return fake_call_model


@pytest.fixture(autouse=True)
def _offline_verdict_seam(monkeypatch):
    """Keep the default-on verdict-extraction seam offline for EVERY test.

    Verdict extraction (CAC-05, wired by CAC-06) is default-on, so any
    ``Council.ask`` in synthesize mode calls ``conclave.verdict_synthesis.call_model``.
    Tests that use the shared ``patch_call_model`` fixture already cover both
    seams, but several test modules install their OWN local ``call_model`` fake on
    the council seam only (e.g. ``test_cache.py``'s ``counting_call_model``); for
    those, an unpatched verdict seam would make a REAL network call and bind the
    pooled httpx client to a doomed event loop, corrupting later sync tests.

    This autouse fixture installs a safe, no-network DEFAULT on the verdict seam:
    a stub that returns innocuous prose, so verdict extraction parse-fails and
    degrades gracefully to ``verdict=None`` (reason "verdict extraction failed
    schema validation") with no network and no assertion impact. Any test that
    needs a real verdict overrides this default -- ``patch_call_model`` re-patches
    the same name on top, and ``test_verdict_synthesis.py`` sets its own seam via
    ``monkeypatch.setattr`` (both run after this fixture, so they win). Tests that
    exercise the real provider highway (``test_providers.py``) patch
    ``conclave.transport.post_json`` and never reach this seam.
    """
    import conclave.verdict_synthesis as verdict_synthesis_mod

    async def _offline_default(name, model_id, messages, *, config=None, **kwargs):
        await asyncio.sleep(0)
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="offline verdict-seam default (not JSON)",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(verdict_synthesis_mod, "call_model", _offline_default)


@pytest.fixture
def patch_call_model(monkeypatch) -> Callable:
    """Return an installer that patches BOTH ``call_model`` seams.

    Usage::

        def handler(model_id, messages, **kwargs):
            return make_response("hi")  # or raise to simulate failure
        patch_call_model(handler)

    The handler is sync (returns a ``_FakeResult`` or raises). The patch wraps it
    so ``await call_model(...)`` works, builds a real ``ModelAnswer`` carrying the
    correct ``name``/``model_id``, and converts a raise into ``ModelAnswer.error``
    -- mirroring the production contract.

    BOTH ``conclave.council.call_model`` (the member fan-out seam) and
    ``conclave.verdict_synthesis.call_model`` (the default-on verdict-extraction
    seam) are patched with the SAME wrapped handler, so a test is network-safe on
    both paths without any change. To return a real verdict from a test, branch on
    ``messages`` (the verdict-extraction system message starts with "You are the
    verdict extractor") and return extraction JSON for that call, prose otherwise.
    """
    import conclave.council as council_mod
    import conclave.verdict_synthesis as verdict_synthesis_mod

    def install(handler: Callable):
        fake = _make_fake_call_model(handler)
        monkeypatch.setattr(council_mod, "call_model", fake)
        monkeypatch.setattr(verdict_synthesis_mod, "call_model", fake)

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


@pytest.fixture
def conclave_caplog(caplog):
    """caplog that reliably captures the non-propagating ``conclave`` logger.

    ``get_logger`` sets ``conclave.propagate = False`` (a key-leak defense), so
    pytest's root-attached caplog handler stops seeing these records once any
    prior test in the suite has configured the logger. Attaching the capture
    handler directly to the ``conclave`` logger makes capture order- and
    propagation-independent.
    """
    logger = logging.getLogger("conclave")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="conclave")
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
