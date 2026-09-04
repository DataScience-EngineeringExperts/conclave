"""Tests for the Council fan-out, partial-failure, skip, and synthesis paths.

All tests run offline via the ``patch_call_model`` fixture; no real keys are
required. Provider env vars are set/cleared explicitly per test.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from conclave import Council
from conclave.config import ConclaveConfig
from tests.conftest import install_council_script, make_failed_answer, make_ok_answer, make_response


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


async def test_resolved_config_identity_reaches_every_model_call(monkeypatch):
    """One resolved config instance reaches members, synthesis, and verdict repair."""
    _all_keys(monkeypatch)

    import conclave.council as council_mod
    import conclave.verdict_synthesis as verdict_synthesis_mod
    from conclave.models import ModelAnswer

    resolved_config = _config()
    received: list[tuple[str, ConclaveConfig | None]] = []

    async def config_spy(
        name,
        model_id,
        messages,
        *,
        temperature=0.7,
        timeout=120.0,
        config=None,
        **kwargs,
    ):
        system = messages[0]["content"] if messages else ""
        if system.startswith("You are the verdict extractor"):
            phase = "verdict_repair" if len(messages) == 3 else "verdict_extraction"
            answer = "not valid json"
        elif system == council_mod._SYNTH_SYSTEM:
            phase = "synthesis"
            answer = "merged"
        else:
            phase = "member"
            answer = "member answer"
        received.append((phase, config))
        return ModelAnswer(name=name, model_id=model_id, answer=answer)

    monkeypatch.setattr(council_mod, "call_model", config_spy)
    monkeypatch.setattr(verdict_synthesis_mod, "call_model", config_spy)

    council = Council(
        models=["grok", "gemini"],
        synthesizer="claude",
        config=resolved_config,
    )
    await council.ask("Should we proceed?")

    assert [phase for phase, _config_value in received].count("member") == 2
    assert {phase for phase, _config_value in received} == {
        "member",
        "synthesis",
        "verdict_extraction",
        "verdict_repair",
    }
    assert all(config_value is resolved_config for _phase, config_value in received)


async def test_concurrency_is_real(monkeypatch):
    """Members run concurrently: total time ~= slowest call, not the sum."""
    _all_keys(monkeypatch)

    import conclave.council as council_mod
    from conclave.models import ModelAnswer

    # Replace call_model with a coroutine that sleeps, to prove gather concurrency.
    async def sleepy_call_model(
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None, **kwargs
    ):
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


# --------------------------------------------------------------------------- #
# synthesizer_chain resolution (DSE-1512)
# --------------------------------------------------------------------------- #


def test_council_chain_defaults_to_single_synthesizer():
    c = Council(models=["grok"], config=ConclaveConfig(synthesizer="claude"))
    assert c.synthesizer_chain == ["claude"] and c.synthesizer == "claude"


def test_council_chain_from_constructor_string():
    c = Council(models=["grok"], synthesizer="claude>grok", config=ConclaveConfig())
    assert c.synthesizer_chain == ["claude", "grok"] and c.synthesizer == "claude"


def test_council_chain_from_constructor_list():
    c = Council(models=["grok"], synthesizer=["gemini", "grok"], config=ConclaveConfig())
    assert c.synthesizer_chain == ["gemini", "grok"] and c.synthesizer == "gemini"


def test_council_chain_from_config_overrides_scalar():
    cfg = ConclaveConfig(synthesizer="claude", synthesizer_chain=["grok", "gemini"])
    c = Council(models=["claude"], config=cfg)
    assert c.synthesizer_chain == ["grok", "gemini"] and c.synthesizer == "grok"


def test_council_constructor_arg_beats_config_chain():
    cfg = ConclaveConfig(synthesizer_chain=["grok", "gemini"])
    c = Council(models=["claude"], synthesizer="claude", config=cfg)
    assert c.synthesizer_chain == ["claude"]


# --------------------------------------------------------------------------- #
# max_spend_usd / max_output_tokens constructor validation (DSE-1514 review,
# F1/F2): a library caller that bypasses the CLI's own format check must get
# the identical rejection at construction, never a crash or a silently
# disabled gate deeper in the call.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_cap", [Decimal("NaN"), Decimal("Infinity"), Decimal("0")])
def test_a_non_finite_or_non_positive_spend_cap_is_a_value_error(bad_cap):
    with pytest.raises(ValueError, match="max_spend_usd must be a finite positive Decimal"):
        Council(
            models=["grok"],
            config=ConclaveConfig(),
            max_output_tokens=1_000,
            max_spend_usd=bad_cap,
        )


@pytest.mark.parametrize("bad_output_cap", [0, -1])
def test_a_non_positive_output_cap_is_a_value_error(bad_output_cap):
    with pytest.raises(ValueError, match="max_output_tokens must be a positive integer"):
        Council(models=["grok"], config=ConclaveConfig(), max_output_tokens=bad_output_cap)


# --------------------------------------------------------------------------- #
# prose synthesis routed through the adjudication succession seam (DSE-1512, task 5)
# --------------------------------------------------------------------------- #

CFG = ConclaveConfig(
    models={
        "claude": "anthropic/c",
        "grok": "xai/g",
        "gemini": "gemini/m",
        "openai": "openai/o",
        "mistral": "mistral/m",
    }
)


async def test_synthesize_mode_fails_over_and_is_not_degraded(monkeypatch, keys):
    calls = install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "quota", 402),
            "grok": make_ok_answer("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.synthesis == "grok says yes" and r.synthesis_error is None and r.degraded is False
    assert (r.synthesizer, r.synthesizer_model_id) == ("grok", "xai/g")
    ledger = r.manifest.adjudication_succession
    assert [(a.role, a.candidate, a.outcome) for a in ledger] == [
        ("synthesis", "claude", "failed_over"),
        ("synthesis", "grok", "success"),
    ]
    assert [
        (x.phase, x.attempt, x.name) for x in r.manifest.receipts if x.phase == "synthesis"
    ] == [
        ("synthesis", 1, "claude"),
        ("synthesis", 2, "grok"),
    ]
    assert r.manifest.secret_safety == "verified_no_secrets"
    assert calls == ["gemini", "claude", "grok"]


async def test_synthesize_mode_exhausted_is_degraded(monkeypatch, keys):
    calls = install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "quota", 429),
            "grok": make_failed_answer("grok", "xai/g", "unavailable", 503),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.synthesis is None
    assert r.synthesis_error == "grok failed"
    assert r.degraded is True
    ledger = r.manifest.adjudication_succession
    assert [a.outcome for a in ledger] == ["failed_over", "exhausted"]
    assert (r.synthesizer, r.synthesizer_model_id) == ("grok", "xai/g")
    assert calls == ["gemini", "claude", "grok"]


async def test_synthesize_chain_of_one_no_key_message_unchanged(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    calls = install_council_script(monkeypatch, {"gemini": make_ok_answer("gemini", "gemini/m")})
    c = Council(models=["gemini"], synthesizer="claude", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.synthesis_error == (
        "synthesizer 'claude' (anthropic/c) has no API key; returning raw answers only"
    )
    ledger = r.manifest.adjudication_succession
    assert [(a.role, a.candidate, a.outcome) for a in ledger] == [
        ("synthesis", "claude", "skipped_unkeyed")
    ]
    assert [x for x in r.manifest.receipts if x.phase == "synthesis"] == []
    assert calls == ["gemini"]


async def test_synthesize_chain_all_unkeyed_message(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    calls = install_council_script(monkeypatch, {"gemini": make_ok_answer("gemini", "gemini/m")})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.synthesis_error == (
        "synthesizer chain [claude, grok] has no API key for any candidate; "
        "returning raw answers only"
    )
    ledger = r.manifest.adjudication_succession
    assert [a.outcome for a in ledger] == ["skipped_unkeyed", "skipped_unkeyed"]
    assert r.degraded is True
    assert calls == ["gemini"]


# --------------------------------------------------------------------------- #
# CouncilResult.primary_failed_over computed field (DSE-1512 review, Unit F)
# --------------------------------------------------------------------------- #


async def test_successor_run_is_primary_failed_over_but_not_degraded(monkeypatch, keys):
    install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "quota", 402),
            "grok": make_ok_answer("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.primary_failed_over is True
    assert r.degraded is False
    assert "primary_failed_over" in r.model_dump(mode="json")


async def test_chain_of_one_clean_run_is_not_primary_failed_over(monkeypatch, keys):
    install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_ok_answer("claude", "anthropic/c"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.primary_failed_over is False
    assert "primary_failed_over" in r.model_dump(mode="json")


async def test_terminal_failure_primary_is_not_primary_failed_over(monkeypatch, keys):
    """A primary that answered unusably (``terminal_failure`` at index 1) is NOT
    primary_failed_over -- it adjudicated, just badly, so failover never fires and
    the run must stay cacheable exactly like any other content failure.
    """
    install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "bad_request", 400),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.primary_failed_over is False
    assert r.degraded is True
    assert [a.outcome for a in r.manifest.adjudication_succession] == ["terminal_failure"]


async def test_successor_after_unkeyed_primary_is_primary_failed_over(monkeypatch):
    """DSE-1512 review, Unit A3: a keyed successor after a SKIPPED (not failed) primary
    still counts as primary_failed_over -- no candidate ever errored on a live call.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "grok": make_ok_answer("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.primary_failed_over is True
    assert r.degraded is False
    ledger = r.manifest.adjudication_succession
    assert [(a.candidate, a.outcome, a.attempt_index) for a in ledger] == [
        ("claude", "skipped_unkeyed", 1),
        ("grok", "success", 2),
    ]


async def test_chain_of_one_unkeyed_is_primary_failed_over(monkeypatch):
    """A chain-of-one unkeyed synthesizer IS primary_failed_over (DSE-1512 review,
    uniform rule): the primary's attempt_index==1 outcome is "skipped_unkeyed" --
    it never adjudicated, for an infrastructure reason (no key) -- exactly like a
    live failover or an exhausted ladder, even though there is no successor to
    have adjudicated instead.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    install_council_script(monkeypatch, {"gemini": make_ok_answer("gemini", "gemini/m")})
    c = Council(models=["gemini"], synthesizer="claude", config=CFG, extract_verdict=False)
    r = await c.ask("q")
    assert r.primary_failed_over is True
    assert r.degraded is True
    assert [a.outcome for a in r.manifest.adjudication_succession] == ["skipped_unkeyed"]


async def test_chain_of_one_unkeyed_is_primary_failed_over_with_verdict_extraction(monkeypatch):
    """Same as above with the default ``extract_verdict=True``: the rule must not
    depend on which roles ran or whether verdict extraction's separate
    unkeyed-candidate handling (see ``Council._apply_verdict``) happened to run.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    install_council_script(monkeypatch, {"gemini": make_ok_answer("gemini", "gemini/m")})
    c = Council(models=["gemini"], synthesizer="claude", config=CFG)
    r = await c.ask("q")
    assert r.primary_failed_over is True
    assert r.degraded is True
