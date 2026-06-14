"""Regression tests pinning the SYNTHESIZER behavior (readiness must-do #5).

The synthesizer/judge path is the heart of conclave's "council" value prop, so
its contract is pinned here explicitly rather than left implicit across
``test_council``/``test_modes``:

* **selection** -- which model synthesizes by default, and that the constructor
  arg, config, and CLI ``--synthesizer`` all override it (a, b, c);
* **observable degradation** -- the synthesizer failing or being unkeyed is
  signaled on the result (``synthesis_error`` / ``verdict_error``), never a
  silent quiet-concat degrade (c, d);
* **prompt versioning** -- the synthesis prompt is a stable, versioned constant,
  stamped onto every result, so a wording change is detectable downstream (e).

All tests run offline via the ``patch_call_model`` fixture (mocking at the same
httpx-transport boundary the rest of the suite uses); the CLI override test
drives Typer's ``CliRunner`` with no network and no real keys.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from conclave import Council, cli
from conclave.config import ConclaveConfig
from conclave.council import _SYNTH_SYSTEM
from conclave.council import SYNTHESIS_PROMPT_VERSION as COUNCIL_VERSION
from conclave.models import CouncilResult
from conclave.prompts import (
    DEBATE_FINAL_SYSTEM,
    JUDGE_SYSTEM,
    SYNTHESIS_PROMPT_VERSION,
)
from conclave.registry import DEFAULT_SYNTHESIZER
from tests.conftest import make_response

runner = CliRunner()


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


def _config(synthesizer: str = "claude") -> ConclaveConfig:
    """A deterministic config independent of any on-disk ~/.conclave file."""
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": "anthropic/claude-sonnet-4-6",
            "perplexity": "perplexity/sonar-pro",
            "openai": "openai/gpt-4.1",
        },
        councils={"default": ["grok", "gemini", "claude", "perplexity"]},
        synthesizer=synthesizer,
    )


def _system_text(messages) -> str:
    """Return the system-role content of a message list, or '' if none."""
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


# --------------------------------------------------------------------------- #
# (a) Default synthesizer selection
# --------------------------------------------------------------------------- #


def test_default_synthesizer_is_config_default():
    """No synthesizer arg -> the config's synthesizer is used (here 'claude')."""
    council = Council(models=["grok", "gemini"], config=_config())
    assert council.synthesizer == "claude"


def test_default_synthesizer_falls_back_to_registry_default():
    """A config with the built-in default synthesizer resolves to 'claude'.

    Pins the bottom of the precedence chain: with no constructor arg and a config
    whose synthesizer is the registry default, the council synthesizes with the
    documented built-in (``DEFAULT_SYNTHESIZER``).
    """
    council = Council(
        models=["grok"],
        config=ConclaveConfig(models={"grok": "xai/grok-4.3"}),  # synthesizer defaults
    )
    assert DEFAULT_SYNTHESIZER == "claude"
    assert council.synthesizer == DEFAULT_SYNTHESIZER


async def test_default_synthesizer_runs_and_is_recorded(monkeypatch, patch_call_model):
    """The default synthesizer actually performs the merge and is named on the result."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        # The synthesizer is anthropic/claude with the 2-message system+merge prompt.
        if model == "anthropic/claude-sonnet-4-6" and _system_text(messages) == _SYNTH_SYSTEM:
            return make_response("DEFAULT MERGE")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], config=_config())  # no synthesizer arg
    result = await council.ask("q")

    assert result.synthesizer == "claude"
    assert result.synthesizer_model_id == "anthropic/claude-sonnet-4-6"
    assert result.synthesis == "DEFAULT MERGE"


# --------------------------------------------------------------------------- #
# (b) Configurable synthesizer override -- constructor arg + config
# --------------------------------------------------------------------------- #


def test_constructor_arg_overrides_config_synthesizer():
    """The constructor ``synthesizer=`` wins over the config default."""
    council = Council(models=["grok"], synthesizer="openai", config=_config("claude"))
    assert council.synthesizer == "openai"


def test_config_synthesizer_used_when_no_arg():
    """With no constructor arg the config's synthesizer is honored (not the registry default)."""
    council = Council(models=["grok"], config=_config("perplexity"))
    assert council.synthesizer == "perplexity"


async def test_overridden_synthesizer_performs_the_merge(monkeypatch, patch_call_model):
    """An overridden synthesizer (openai) is the model that runs the merge."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if model == "openai/gpt-4.1" and _system_text(messages) == _SYNTH_SYSTEM:
            return make_response("OPENAI MERGE")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="openai", config=_config("claude"))
    result = await council.ask("q")

    assert result.synthesizer == "openai"
    assert result.synthesizer_model_id == "openai/gpt-4.1"
    assert result.synthesis == "OPENAI MERGE"


# --------------------------------------------------------------------------- #
# (c) Configurable synthesizer override -- CLI ``--synthesizer``
# --------------------------------------------------------------------------- #


def test_cli_synthesizer_flag_overrides(monkeypatch, patch_call_model):
    """``--synthesizer openai`` makes openai the synthesizer end-to-end via the CLI."""
    monkeypatch.setattr(cli, "load_config", lambda: _config("claude"))
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")
    # claude (the config default) intentionally has NO key: if the flag were
    # ignored, synthesis would degrade to a no-key error instead of merging.

    def handler(model, messages, **kwargs):
        if model == "openai/gpt-4.1" and _system_text(messages) == _SYNTH_SYSTEM:
            return make_response("CLI OPENAI MERGE")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    result = runner.invoke(
        cli.app,
        ["ask", "q", "--council", "grok,gemini", "--synthesizer", "openai", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["synthesizer"] == "openai"
    assert payload["synthesizer_model_id"] == "openai/gpt-4.1"
    assert payload["synthesis"] == "CLI OPENAI MERGE"


# --------------------------------------------------------------------------- #
# (d) Degraded / fallback path is SIGNALED, never silent -- synthesize mode
# --------------------------------------------------------------------------- #


async def test_synthesizer_unkeyed_is_signaled_not_silent(
    monkeypatch, patch_call_model, clear_keys
):
    """Synthesizer with no key -> synthesis is None AND synthesis_error explains it.

    The degraded path must be observable: a caller can tell synthesis did not run
    (no quietly-concatenated output masquerading as a synthesis). Member answers
    are preserved; the selected synthesizer identity is still recorded.
    """
    monkeypatch.setenv("XAI_API_KEY", "dummy")  # only grok has a key; claude (synth) does not

    def handler(model, messages, **kwargs):
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(models=["grok"], synthesizer="claude", config=_config())
    result = await council.ask("q")

    # Happy-path member output is untouched...
    assert len(result.successful_answers) == 1
    # ...but synthesis is explicitly NOT produced, and the reason is observable.
    assert result.synthesis is None
    assert result.synthesis_error is not None
    assert "no API key" in result.synthesis_error
    # The selected synthesizer is still recorded even though it could not run.
    assert result.synthesizer == "claude"
    assert result.synthesizer_model_id == "anthropic/claude-sonnet-4-6"


async def test_synthesizer_call_failure_is_signaled(monkeypatch, patch_call_model):
    """Synthesizer keyed but the call errors -> synthesis None, error surfaced verbatim."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if model == "anthropic/claude-sonnet-4-6" and _system_text(messages) == _SYNTH_SYSTEM:
            raise RuntimeError("synthesizer 503 from provider")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("q")

    # Members succeeded; only the synthesis step failed, and it is signaled.
    assert len(result.successful_answers) == 2
    assert result.synthesis is None
    assert result.synthesis_error is not None
    assert "synthesizer 503 from provider" in result.synthesis_error


async def test_no_usable_answers_is_signaled(monkeypatch, patch_call_model):
    """Every member fails -> synthesis None with a 'nothing to merge' signal."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        raise RuntimeError("all members down")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("q")

    assert result.synthesis is None
    assert result.synthesis_error is not None
    assert "no successful member answers" in result.synthesis_error


# --------------------------------------------------------------------------- #
# (d') Degraded path is SIGNALED -- adversarial JUDGE (the analogous role)
# --------------------------------------------------------------------------- #


async def test_adversarial_judge_unkeyed_is_signaled(monkeypatch, patch_call_model, clear_keys):
    """Judge (synthesizer) with no key -> verdict None, verdict_error + mirror set.

    The adversarial judge is the same model as the synthesizer; its degraded path
    must be just as observable. The proposal and critiques survive; the missing
    verdict is signaled on both ``adversarial.verdict_error`` and the mirrored
    ``result.synthesis_error``.
    """
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    # claude (the judge) intentionally has no key.

    def handler(model, messages, **kwargs):
        if "critic on an adversarial review" in _system_text(messages):
            return make_response(f"crit {model}")
        return make_response(f"prop {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.adversarial("q")

    adv = result.adversarial
    assert adv is not None
    assert adv.proposal.ok  # proposal survived
    assert len(adv.successful_critiques) == 1  # a critique survived
    assert adv.verdict is None
    assert adv.verdict_error is not None
    assert "no API key" in adv.verdict_error
    # The selected judge identity is recorded, and the error mirrors to synthesis_error.
    assert adv.judge == "claude"
    assert adv.judge_model_id == "anthropic/claude-sonnet-4-6"
    assert result.synthesis_error == adv.verdict_error


async def test_adversarial_judge_call_failure_is_signaled(monkeypatch, patch_call_model):
    """Judge keyed but the verdict call errors -> verdict None, error surfaced."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            raise RuntimeError("judge 500 from provider")
        if "critic on an adversarial review" in system:
            return make_response(f"crit {model}")
        return make_response(f"prop {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.adversarial("q")

    adv = result.adversarial
    assert adv is not None
    assert adv.verdict is None
    assert adv.verdict_error is not None
    assert "judge 500 from provider" in adv.verdict_error


async def test_debate_synthesizer_unkeyed_is_signaled(monkeypatch, patch_call_model, clear_keys):
    """Debate's final synthesizer with no key -> synthesis None + observable error."""
    monkeypatch.setenv("XAI_API_KEY", "dummy")  # only grok; claude (synth) has no key

    def handler(model, messages, **kwargs):
        return make_response(f"answer {model}")

    patch_call_model(handler)

    council = Council(models=["grok"], synthesizer="claude", config=_config())
    result = await council.debate("q", rounds=1)

    assert result.synthesis is None
    assert result.synthesis_error is not None
    assert "no API key" in result.synthesis_error


# --------------------------------------------------------------------------- #
# (e) Prompt-version constant is stable + asserted
# --------------------------------------------------------------------------- #


def test_prompt_version_is_a_stable_nonempty_string():
    """The version tag is a non-empty string and re-exported consistently."""
    assert isinstance(SYNTHESIS_PROMPT_VERSION, str)
    assert SYNTHESIS_PROMPT_VERSION
    # council re-exports the same object the prompts module owns.
    assert COUNCIL_VERSION == SYNTHESIS_PROMPT_VERSION


def test_prompt_version_is_pinned():
    """Pin the exact version so a prompt change without a version bump fails CI.

    This is the tripwire: editing any synthesizer-facing prompt below WITHOUT
    bumping ``SYNTHESIS_PROMPT_VERSION`` leaves this assertion (and the prompt-text
    pins) inconsistent, so the change cannot land silently.
    """
    assert SYNTHESIS_PROMPT_VERSION == "2026-06-14"


def test_synthesis_prompt_text_is_pinned():
    """Pin the synthesize/debate/judge prompt wording.

    Guards the happy-path output contract: the synthesizer-facing prompts are
    byte-stable. Any intentional edit must update this test AND bump
    ``SYNTHESIS_PROMPT_VERSION`` (see ``test_prompt_version_is_pinned``).
    """
    assert _SYNTH_SYSTEM.startswith("You are the synthesizer of a council of AI models.")
    assert "rely only on the answers provided" in _SYNTH_SYSTEM
    assert DEBATE_FINAL_SYSTEM.startswith("You are the synthesizer concluding")
    assert JUDGE_SYSTEM.startswith("You are the judge of an adversarial review.")


def test_every_result_carries_the_prompt_version():
    """A bare CouncilResult defaults prompt_version to the current tag."""
    result = CouncilResult(prompt="x")
    assert result.prompt_version == SYNTHESIS_PROMPT_VERSION


async def test_live_run_stamps_prompt_version(monkeypatch, patch_call_model):
    """A real synthesize run stamps the version onto the result (and into JSON)."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if model == "anthropic/claude-sonnet-4-6" and _system_text(messages) == _SYNTH_SYSTEM:
            return make_response("MERGE")
        return make_response(f"answer from {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("q")

    assert result.prompt_version == SYNTHESIS_PROMPT_VERSION
    # The version survives JSON serialization for downstream eval pipelines.
    assert result.model_dump(mode="json")["prompt_version"] == SYNTHESIS_PROMPT_VERSION


@pytest.mark.parametrize("mode", ["raw", "debate", "adversarial"])
def test_prompt_version_stamped_in_every_mode(monkeypatch, patch_call_model, mode):
    """Every mode's result carries prompt_version, even when synthesis does not run."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        return make_response(f"answer {model}")

    patch_call_model(handler)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    if mode == "raw":
        result = council.ask_sync("q", synthesize=False)
    elif mode == "debate":
        result = council.debate_sync("q", rounds=1)
    else:
        result = council.adversarial_sync("q")

    assert result.prompt_version == SYNTHESIS_PROMPT_VERSION
