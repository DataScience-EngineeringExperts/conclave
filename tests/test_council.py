"""Tests for the Council fan-out, partial-failure, skip, and synthesis paths.

All tests run offline via the ``patch_acompletion`` fixture; no real keys are
required. Provider env vars are set/cleared explicitly per test.
"""

from __future__ import annotations

import asyncio

import pytest

from conclave import Council
from conclave.config import ConclaveConfig
from tests.conftest import make_response


def _all_keys(monkeypatch) -> None:
    """Set every provider key to a dummy non-empty value."""
    for var in (
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(var, "dummy-key")


def _config() -> ConclaveConfig:
    """A deterministic config independent of any on-disk ~/.conclave file."""
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": "anthropic/claude-sonnet-4-6",
            "perplexity": "perplexity/sonar-pro",
        },
        councils={"default": ["grok", "gemini", "claude", "perplexity"]},
        synthesizer="claude",
    )


async def test_fan_out_collects_all_members(monkeypatch, patch_acompletion):
    """All members run concurrently and each raw answer is captured."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        # Synthesizer is anthropic with the system+merge prompt; members are single-turn.
        if model == "anthropic/claude-sonnet-4-6" and len(messages) == 2:
            return make_response("MERGED")
        return make_response(f"answer from {model}")

    patch_acompletion(handler)

    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.ask("What is 2+2?")

    assert len(result.answers) == 3
    assert {a.name for a in result.answers} == {"grok", "gemini", "perplexity"}
    assert all(a.ok for a in result.answers)
    assert all(a.usage and a.usage.total_tokens == 12 for a in result.answers)
    assert result.synthesis == "MERGED"
    assert result.synthesizer == "claude"


async def test_concurrency_is_real(monkeypatch, patch_acompletion):
    """Members run concurrently: total time ~= slowest call, not the sum."""
    _all_keys(monkeypatch)

    async def slow_handler_wrapper():
        pass

    def handler(model, messages, **kwargs):
        return make_response(f"ok {model}")

    # Replace acompletion with one that sleeps, to prove gather concurrency.
    import litellm

    async def sleepy_acompletion(*, model, messages, **kwargs):
        await asyncio.sleep(0.2)
        return handler(model, messages, **kwargs)

    monkeypatch.setattr(litellm, "acompletion", sleepy_acompletion)

    council = Council(
        models=["grok", "gemini", "perplexity"], config=_config()
    )
    start = asyncio.get_event_loop().time()
    result = await council.ask("hi", synthesize=False)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(result.answers) == 3
    # 3 sequential calls would be ~0.6s; concurrent should be well under 0.45s.
    assert elapsed < 0.45, f"expected concurrent execution, took {elapsed:.2f}s"


async def test_partial_failure_one_provider_raises(monkeypatch, patch_acompletion):
    """One member raising does not kill the run; others still return."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if model == "gemini/gemini-2.5-pro":
            raise RuntimeError("simulated gemini 500")
        if model == "anthropic/claude-sonnet-4-6" and len(messages) == 2:
            return make_response("MERGED FROM SURVIVORS")
        return make_response(f"answer from {model}")

    patch_acompletion(handler)

    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.ask("question")

    assert len(result.answers) == 3
    assert len(result.successful_answers) == 2
    assert len(result.failed_answers) == 1
    failed = result.failed_answers[0]
    assert failed.name == "gemini"
    assert "simulated gemini 500" in failed.error
    # Synthesis still runs over the two survivors.
    assert result.synthesis == "MERGED FROM SURVIVORS"


async def test_missing_key_is_skipped(monkeypatch, patch_acompletion, clear_keys):
    """Members without a key are skipped with a warning, run proceeds."""
    # Only grok + perplexity have keys.
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "dummy")

    def handler(model, messages, **kwargs):
        if model == "perplexity/sonar-pro" and len(messages) == 2:
            return make_response("MERGED")  # perplexity as synthesizer here
        return make_response(f"answer from {model}")

    patch_acompletion(handler)

    council = Council(
        models=["grok", "gemini", "claude", "perplexity"],
        synthesizer="perplexity",
        config=_config(),
    )
    result = await council.ask("q")

    assert {a.name for a in result.answers} == {"grok", "perplexity"}
    assert set(result.skipped) == {"gemini", "claude"}
    assert result.synthesis == "MERGED"


async def test_synthesizer_without_key_returns_raw(monkeypatch, patch_acompletion, clear_keys):
    """If the synthesizer's key is absent, raw answers return with an error note."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")  # only grok has a key

    def handler(model, messages, **kwargs):
        return make_response(f"answer from {model}")

    patch_acompletion(handler)

    council = Council(
        models=["grok"], synthesizer="claude", config=_config()
    )
    result = await council.ask("q")

    assert len(result.successful_answers) == 1
    assert result.synthesis is None
    assert result.synthesis_error is not None
    assert "no API key" in result.synthesis_error


async def test_no_members_available(monkeypatch, patch_acompletion, clear_keys):
    """Zero available members yields an empty result, not an exception."""
    def handler(model, messages, **kwargs):  # pragma: no cover - never called
        return make_response("unused")

    patch_acompletion(handler)

    council = Council(models=["grok", "claude"], config=_config())
    result = await council.ask("q")

    assert result.answers == []
    assert set(result.skipped) == {"grok", "claude"}
    assert result.synthesis is None


async def test_synthesis_over_no_survivors(monkeypatch, patch_acompletion):
    """When every member fails, synthesis reports it has nothing to merge."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        raise RuntimeError("everything is down")

    patch_acompletion(handler)

    council = Council(
        models=["grok", "gemini"], synthesizer="claude", config=_config()
    )
    result = await council.ask("q")

    assert len(result.failed_answers) == 2
    assert result.synthesis is None
    assert "no successful member answers" in result.synthesis_error


def test_ask_sync_wrapper(monkeypatch, patch_acompletion):
    """The sync entry point works from non-async code."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def handler(model, messages, **kwargs):
        if model == "anthropic/claude-sonnet-4-6" and len(messages) == 2:
            return make_response("SYNC MERGE")
        return make_response(f"answer from {model}")

    patch_acompletion(handler)

    council = Council(models=["grok"], synthesizer="claude", config=_config())
    result = council.ask_sync("hello")

    assert result.synthesis == "SYNC MERGE"
    assert len(result.successful_answers) == 1


async def test_ask_sync_raises_inside_loop(monkeypatch):
    """ask_sync from within a running loop raises a clear error."""
    council = Council(models=["grok"], config=_config())
    with pytest.raises(RuntimeError, match="running event loop"):
        council.ask_sync("hi")
