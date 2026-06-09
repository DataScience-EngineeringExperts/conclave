"""Tests for the Council fan-out, partial-failure, skip, and synthesis paths.

All tests run offline via the ``patch_call_model`` fixture; no real keys are
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


async def test_fan_out_collects_all_members(monkeypatch, patch_call_model):
    """All members run concurrently and each raw answer is captured."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        # Synthesizer is anthropic with the system+merge prompt; members are single-turn.
        if model == "anthropic/claude-sonnet-4-6" and len(messages) == 2:
            return make_response("MERGED")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

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


async def test_concurrency_is_real(monkeypatch):
    """Members run concurrently: total time ~= slowest call, not the sum."""
    _all_keys(monkeypatch)

    import conclave.council as council_mod
    from conclave.models import ModelAnswer

    # Replace call_model with a coroutine that sleeps, to prove gather concurrency.
    async def sleepy_call_model(name, model_id, messages, *, temperature=0.7, timeout=120.0):
        await asyncio.sleep(0.2)
        return ModelAnswer(name=name, model_id=model_id, answer=f"ok {model_id}")

    monkeypatch.setattr(council_mod, "call_model", sleepy_call_model)

    council = Council(models=["grok", "gemini", "perplexity"], config=_config())
    start = asyncio.get_event_loop().time()
    result = await council.ask("hi", synthesize=False)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(result.answers) == 3
    # 3 sequential calls would be ~0.6s; concurrent should be well under 0.45s.
    assert elapsed < 0.45, f"expected concurrent execution, took {elapsed:.2f}s"


async def test_partial_failure_one_provider_raises(monkeypatch, patch_call_model):
    """One member raising does not kill the run; others still return."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if model == "gemini/gemini-2.5-pro":
            raise RuntimeError("simulated gemini 500")
        if model == "anthropic/claude-sonnet-4-6" and len(messages) == 2:
            return make_response("MERGED FROM SURVIVORS")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

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


async def test_missing_key_is_skipped(monkeypatch, patch_call_model, clear_keys):
    """Members without a key are skipped with a warning, run proceeds."""
    # Only grok + perplexity have keys.
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "dummy")

    def handler(model, messages, **kwargs):
        if model == "perplexity/sonar-pro" and len(messages) == 2:
            return make_response("MERGED")  # perplexity as synthesizer here
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(
        models=["grok", "gemini", "claude", "perplexity"],
        synthesizer="perplexity",
        config=_config(),
    )
    result = await council.ask("q")

    assert {a.name for a in result.answers} == {"grok", "perplexity"}
    assert set(result.skipped) == {"gemini", "claude"}
    assert result.synthesis == "MERGED"


async def test_synthesizer_without_key_returns_raw(monkeypatch, patch_call_model, clear_keys):
    """If the synthesizer's key is absent, raw answers return with an error note."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")  # only grok has a key

    def handler(model, messages, **kwargs):
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(models=["grok"], synthesizer="claude", config=_config())
    result = await council.ask("q")

    assert len(result.successful_answers) == 1
    assert result.synthesis is None
    assert result.synthesis_error is not None
    assert "no API key" in result.synthesis_error


async def test_no_members_available(monkeypatch, patch_call_model, clear_keys):
    """Zero available members yields an empty result, not an exception."""

    def handler(model, messages, **kwargs):  # pragma: no cover - never called
        return make_response("unused")

    patch_call_model(handler)

    council = Council(models=["grok", "claude"], config=_config())
    result = await council.ask("q")

    assert result.answers == []
    assert set(result.skipped) == {"grok", "claude"}
    assert result.synthesis is None


async def test_synthesis_over_no_survivors(monkeypatch, patch_call_model):
    """When every member fails, synthesis reports it has nothing to merge."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        raise RuntimeError("everything is down")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("q")

    assert len(result.failed_answers) == 2
    assert result.synthesis is None
    assert "no successful member answers" in result.synthesis_error


def test_ask_sync_wrapper(monkeypatch, patch_call_model):
    """The sync entry point works from non-async code."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def handler(model, messages, **kwargs):
        if model == "anthropic/claude-sonnet-4-6" and len(messages) == 2:
            return make_response("SYNC MERGE")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(models=["grok"], synthesizer="claude", config=_config())
    result = council.ask_sync("hello")

    assert result.synthesis == "SYNC MERGE"
    assert len(result.successful_answers) == 1


async def test_ask_sync_raises_inside_loop(monkeypatch):
    """ask_sync from within a running loop raises a clear error."""
    council = Council(models=["grok"], config=_config())
    with pytest.raises(RuntimeError, match="running event loop"):
        council.ask_sync("hi")


async def test_config_disk_read_at_most_once_per_ask(monkeypatch, tmp_path):
    """A full Council.ask run hits the config file on disk at most once (issue #15).

    Exercises the REAL call path (transport patched, not call_model) so every
    member call plus synthesis flows through providers.call_model -> load_config.
    With the memoized loader, the underlying disk read happens at most once across
    the whole run rather than once per model call.
    """
    import conclave.config as config_mod

    _all_keys(monkeypatch)

    # Synthesizer is openai so the OpenAI-shaped transport stub serves both the
    # members and the synthesis call (all OpenAI-compatible).
    config_file = tmp_path / "conclave.yml"
    config_file.write_text("synthesizer: openai\n", encoding="utf-8")
    monkeypatch.setenv("CONCLAVE_CONFIG", str(config_file))

    config_mod.clear_config_cache()

    reads = {"n": 0}
    real_read_yaml = config_mod._read_yaml

    def counting_read_yaml(path):
        reads["n"] += 1
        return real_read_yaml(path)

    monkeypatch.setattr(config_mod, "_read_yaml", counting_read_yaml)

    async def fake_post(url, headers, json_body, timeout):
        return 200, {"choices": [{"message": {"content": "answer"}}]}

    monkeypatch.setattr("conclave.transport.post_json", fake_post)

    # Council built with no injected config -> resolves via load_config; every
    # member + synthesis call then also calls load_config from providers.
    council = Council(models=["grok", "perplexity", "openai"], synthesizer="openai")
    result = await council.ask("what is 2+2?")

    assert result.synthesis == "answer"
    assert len(result.answers) == 3
    assert reads["n"] <= 1, f"expected at most one disk read for the run, got {reads['n']}"

    config_mod.clear_config_cache()
