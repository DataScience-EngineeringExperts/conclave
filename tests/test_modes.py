"""Tests for the debate and adversarial deliberation modes.

All tests run offline via the ``patch_call_model`` fixture; no real keys are
required. Provider env vars are set/cleared explicitly per test. The handlers
inspect the message list to tell roles apart: members get a single user message
in round 1, the synthesizer/judge gets a 2-message system+user prompt, and
debate rounds >= 2 / critics get a system+user prompt as well -- so handlers key
off the system-prompt text to disambiguate.
"""

from __future__ import annotations

import pytest

from conclave import AdversarialResult, Council, DebateRound
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


def _system_text(messages) -> str:
    """Return the system-role content of a message list, or '' if none."""
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


# --------------------------------------------------------------------------- #
# Debate
# --------------------------------------------------------------------------- #


async def test_debate_multi_round_flow(monkeypatch, patch_call_model):
    """Three members, two rounds: each round captured, final synthesis produced."""
    _all_keys(monkeypatch)

    seen_systems: list[str] = []

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        seen_systems.append(system)
        if "synthesizer concluding a multi-round" in system:
            return make_response("DEBATE SYNTHESIS")
        if "debating a prompt over several rounds" in system:
            return make_response(f"round2 from {model}")
        # round 1: bare user prompt, no system message
        return make_response(f"round1 from {model}")

    patch_call_model(handler)

    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.debate("Is P=NP?", rounds=2)

    assert result.mode == "debate"
    assert len(result.rounds) == 2
    assert all(isinstance(r, DebateRound) for r in result.rounds)
    # Each round ran all three members.
    assert {a.name for a in result.rounds[0].answers} == {"grok", "gemini", "perplexity"}
    assert {a.name for a in result.rounds[1].answers} == {"grok", "gemini", "perplexity"}
    # Round 1 has no system prompt; round 2 used the debate system prompt.
    assert any("round1 from" in a.answer for a in result.rounds[0].answers)
    assert any("round2 from" in a.answer for a in result.rounds[1].answers)
    # answers mirrors the final round for backward-compatible consumers.
    assert {a.name for a in result.answers} == {"grok", "gemini", "perplexity"}
    assert result.answers == result.rounds[-1].answers
    assert result.synthesis == "DEBATE SYNTHESIS"
    assert result.synthesizer == "claude"


async def test_debate_single_round_is_independent(monkeypatch, patch_call_model):
    """rounds=1 behaves like one independent fan-out plus synthesis."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if "synthesizer concluding a multi-round" in _system_text(messages):
            return make_response("ONE ROUND SYNTH")
        # Round 1 must never carry a debate system prompt.
        assert "debating a prompt" not in _system_text(messages)
        return make_response(f"answer {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.debate("q", rounds=1)

    assert len(result.rounds) == 1
    assert result.synthesis == "ONE ROUND SYNTH"


async def test_debate_partial_failure_mid_round(monkeypatch, patch_call_model):
    """A member that errors in round 1 drops out of round 2; debate continues."""
    _all_keys(monkeypatch)

    round2_members: list[str] = []

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "synthesizer concluding a multi-round" in system:
            return make_response("SYNTH OF SURVIVORS")
        if "debating a prompt over several rounds" in system:
            round2_members.append(model)
            return make_response(f"round2 {model}")
        # Round 1: gemini blows up.
        if model == "gemini/gemini-2.5-pro":
            raise RuntimeError("gemini 500 in round 1")
        return make_response(f"round1 {model}")

    patch_call_model(handler)

    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.debate("q", rounds=2)

    # Round 1: all three attempted, gemini failed.
    r1 = result.rounds[0]
    assert len(r1.answers) == 3
    assert {a.name for a in r1.answers if not a.ok} == {"gemini"}
    # Round 2: only the two survivors participate.
    r2 = result.rounds[1]
    assert {a.name for a in r2.answers} == {"grok", "perplexity"}
    assert "gemini/gemini-2.5-pro" not in round2_members
    # Synthesis runs over survivors.
    assert result.synthesis == "SYNTH OF SURVIVORS"


async def test_debate_all_fail_first_round(monkeypatch, patch_call_model):
    """If everyone fails round 1, no survivors remain and synthesis reports it."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        raise RuntimeError("everything down")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.debate("q", rounds=3)

    # Round 1 attempted both; both failed -> debate ends after round 1.
    assert len(result.rounds) == 1
    assert all(not a.ok for a in result.rounds[0].answers)
    assert result.synthesis is None
    assert "no surviving member answers" in result.synthesis_error


async def test_debate_peers_anonymized_in_round2(monkeypatch, patch_call_model):
    """Round 2 prompts attribute peers as 'Model A/B/...', not by brand name.

    Anonymization relabels the *attribution header* on each prior answer; the
    answer body is passed verbatim. So we give round-1 answers brand-neutral
    bodies and assert no friendly name or model id leaks into the attribution.
    """
    _all_keys(monkeypatch)

    captured_round2: list[str] = []
    # Map each model id to a brand-neutral round-1 body.
    bodies = {
        "xai/grok-4.3": "the answer is forty two",
        "gemini/gemini-2.5-pro": "the answer is seven",
    }

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "synthesizer concluding a multi-round" in system:
            return make_response("S")
        if "debating a prompt over several rounds" in system:
            user = next(m["content"] for m in messages if m["role"] == "user")
            captured_round2.append(user)
            return make_response("revised neutral body")
        return make_response(bodies[model])

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    await council.debate("q", rounds=2)

    assert captured_round2, "expected round-2 prompts to be captured"
    joined = "\n".join(captured_round2)
    # Anonymized attribution labels are present...
    assert "Model A" in joined
    assert "Model B" in joined
    # ...and no friendly name or model id leaks into the peer attributions.
    assert "grok" not in joined.lower()
    assert "gemini" not in joined.lower()
    # The verbatim peer answer bodies still cross-pollinate.
    assert "the answer is forty two" in joined
    assert "the answer is seven" in joined


async def test_debate_no_members_available(monkeypatch, patch_call_model, clear_keys):
    """Zero available members yields an empty debate result, not an exception."""

    def handler(model, messages, **kwargs):  # pragma: no cover - never called
        return make_response("unused")

    patch_call_model(handler)

    council = Council(models=["grok", "claude"], config=_config())
    result = await council.debate("q", rounds=2)

    assert result.mode == "debate"
    assert result.rounds == []
    assert result.answers == []
    assert set(result.skipped) == {"grok", "claude"}


def test_debate_sync_wrapper(monkeypatch, patch_call_model):
    """The sync debate entry point works from non-async code."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def handler(model, messages, **kwargs):
        if "synthesizer concluding a multi-round" in _system_text(messages):
            return make_response("SYNC DEBATE")
        return make_response(f"answer {model}")

    patch_call_model(handler)

    council = Council(models=["grok"], synthesizer="claude", config=_config())
    result = council.debate_sync("hi", rounds=2)

    assert result.mode == "debate"
    assert result.synthesis == "SYNC DEBATE"


# --------------------------------------------------------------------------- #
# Adversarial
# --------------------------------------------------------------------------- #


async def test_adversarial_proposer_critic_verdict(monkeypatch, patch_call_model):
    """Default proposer answers, others critique, judge issues a verdict."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            return make_response("VERDICT TEXT")
        if "critic on an adversarial review" in system:
            return make_response(f"critique from {model}")
        # Proposal: bare user prompt, no system message.
        return make_response(f"proposal from {model}")

    patch_call_model(handler)

    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.adversarial("Defend microservices.")

    assert result.mode == "adversarial"
    adv = result.adversarial
    assert isinstance(adv, AdversarialResult)
    # Default proposer is the first requested member.
    assert adv.proposer == "grok"
    assert adv.proposal.ok
    assert "proposal from xai/grok-4.3" in adv.proposal.answer
    # The other two members are critics.
    assert {c.name for c in adv.critiques} == {"gemini", "perplexity"}
    assert all("critique from" in c.answer for c in adv.critiques)
    # Judge verdict mirrors into synthesis for backward-compatible consumers.
    assert adv.verdict == "VERDICT TEXT"
    assert result.synthesis == "VERDICT TEXT"
    assert adv.judge == "claude"
    # answers carries proposal + critiques.
    assert {a.name for a in result.answers} == {"grok", "gemini", "perplexity"}


async def test_adversarial_explicit_proposer(monkeypatch, patch_call_model):
    """--proposer selects a non-default member; the rest become critics."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            return make_response("V")
        if "critic on an adversarial review" in system:
            return make_response(f"crit {model}")
        return make_response(f"prop {model}")

    patch_call_model(handler)

    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.adversarial("q", proposer="perplexity")

    assert result.adversarial.proposer == "perplexity"
    assert {c.name for c in result.adversarial.critiques} == {"grok", "gemini"}


async def test_adversarial_critic_failure_still_verdicts(monkeypatch, patch_call_model):
    """One critic failing does not abort; the judge still issues a verdict."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            return make_response("VERDICT DESPITE FAILURE")
        if "critic on an adversarial review" in system:
            if model == "gemini/gemini-2.5-pro":
                raise RuntimeError("critic gemini crashed")
            return make_response(f"crit {model}")
        return make_response(f"prop {model}")

    patch_call_model(handler)

    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.adversarial("q")  # proposer grok

    adv = result.adversarial
    assert {c.name for c in adv.critiques} == {"gemini", "perplexity"}
    assert {c.name for c in adv.critiques if not c.ok} == {"gemini"}
    assert len(adv.successful_critiques) == 1
    assert adv.verdict == "VERDICT DESPITE FAILURE"


async def test_adversarial_proposal_failure_skips_critics(monkeypatch, patch_call_model):
    """If the proposal itself fails, critics are skipped and no verdict is made."""
    _all_keys(monkeypatch)

    critic_calls: list[str] = []

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:  # pragma: no cover
            return make_response("should-not-run")
        if "critic on an adversarial review" in system:  # pragma: no cover
            critic_calls.append(model)
            return make_response("crit")
        # Proposal (grok) fails.
        raise RuntimeError("proposer down")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.adversarial("q")

    adv = result.adversarial
    assert not adv.proposal.ok
    assert adv.critiques == []
    assert critic_calls == []
    assert adv.verdict is None
    assert "proposal from 'grok' failed" in adv.verdict_error
    assert result.synthesis is None


async def test_adversarial_proposer_no_key_falls_back(monkeypatch, patch_call_model, clear_keys):
    """A requested proposer without a key falls back to the first available member."""
    # Only gemini + claude have keys; requested proposer grok does not.
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            return make_response("V")
        if "critic on an adversarial review" in system:
            return make_response("crit")
        return make_response(f"prop {model}")

    patch_call_model(handler)

    council = Council(
        models=["grok", "gemini", "claude"],
        synthesizer="claude",
        config=_config(),
    )
    result = await council.adversarial("q", proposer="grok")

    # grok skipped (no key); proposer falls back to the first available (gemini).
    assert "grok" in result.skipped
    assert result.adversarial.proposer == "gemini"


async def test_adversarial_judge_no_key_returns_structure(
    monkeypatch, patch_call_model, clear_keys
):
    """If the judge has no key, proposal + critiques return with a verdict error."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    # claude (judge) intentionally has no key.

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "critic on an adversarial review" in system:
            return make_response(f"crit {model}")
        return make_response(f"prop {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.adversarial("q")

    adv = result.adversarial
    assert adv.proposal.ok
    assert len(adv.successful_critiques) == 1
    assert adv.verdict is None
    assert "no API key" in adv.verdict_error


async def test_adversarial_no_members_available(monkeypatch, patch_call_model, clear_keys):
    """Zero available members yields an empty adversarial result, not an error."""

    def handler(model, messages, **kwargs):  # pragma: no cover - never called
        return make_response("unused")

    patch_call_model(handler)

    council = Council(models=["grok", "claude"], config=_config())
    result = await council.adversarial("q")

    assert result.mode == "adversarial"
    assert result.adversarial is None
    assert result.answers == []
    assert set(result.skipped) == {"grok", "claude"}


def test_adversarial_sync_wrapper(monkeypatch, patch_call_model):
    """The sync adversarial entry point works from non-async code."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            return make_response("SYNC VERDICT")
        if "critic on an adversarial review" in system:
            return make_response("crit")
        return make_response("prop")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = council.adversarial_sync("hi")

    assert result.mode == "adversarial"
    assert result.synthesis == "SYNC VERDICT"


def test_debate_sync_raises_inside_loop(monkeypatch):
    """debate_sync from within a running loop raises a clear error (async test = loop)."""

    async def _inner():
        council = Council(models=["grok"], config=_config())
        with pytest.raises(RuntimeError, match="running event loop"):
            council.debate_sync("hi")

    import asyncio

    asyncio.run(_inner())


def test_adversarial_sync_raises_inside_loop():
    """adversarial_sync from within a running loop raises a clear error."""
    import asyncio

    async def _inner():
        council = Council(models=["grok"], config=_config())
        with pytest.raises(RuntimeError, match="running event loop"):
            council.adversarial_sync("hi")

    asyncio.run(_inner())
