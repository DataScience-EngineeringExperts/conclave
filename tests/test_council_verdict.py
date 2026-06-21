"""Tests for CAC-06: verdict extraction wired into ``Council.ask``.

The verdict-extraction step (CAC-05 engine) is default-on. CAC-06 wires it into
the buffered ``ask``/``_ask_uncached`` path via the single shared helper
:meth:`conclave.council.Council._apply_verdict`, hoists the canonical verdict's
fields onto :class:`conclave.models.CouncilResult`, and records provenance on the
manifest. These tests drive both seams offline through ``patch_call_model`` (which
patches the council seam AND the verdict seam) and assert the wiring, the
mirroring, the manifest provenance, the opt-out, the raw-mode skip, and graceful
degradation -- all without a network call.

A handler returns prose for member calls and, when it sees the verdict-extraction
system message ("You are the verdict extractor"), returns extraction JSON, so a
real verdict flows end to end. See ``conftest`` for the dual-seam design.
"""

from __future__ import annotations

import json

from conclave import Council
from conclave.config import ConclaveConfig
from conclave.manifest import SECRET_SAFETY_VERIFIED, ModelHarnessManifest
from conclave.models import CouncilResult, ModelAnswer
from conclave.verdict import CONSENSUS_METHOD
from conclave.verdict_synthesis import (
    _REASON_EXTRACTION_FAILED,
    _REASON_OPEN_ENDED,
    _REASON_TOO_FEW,
)
from tests.conftest import make_response

# The synthesizer/extractor resolved id for the "claude" friendly name below.
_SYNTH_MODEL_ID = "anthropic/claude-sonnet-4-6"


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
            "claude": _SYNTH_MODEL_ID,
            "perplexity": "perplexity/sonar-pro",
        },
        councils={"default": ["grok", "gemini", "claude", "perplexity"]},
        synthesizer=synthesizer,
    )


def _is_verdict_call(messages) -> bool:
    """True when ``messages`` is the verdict-extraction call (vs a member/prose call)."""
    return bool(messages) and messages[0].get("content", "").startswith(
        "You are the verdict extractor"
    )


def _extraction_json(
    *,
    verdict_applies: bool = True,
    verdict_type: str = "decision",
    members: tuple[str, ...] = ("grok", "gemini", "perplexity"),
    position_label: str = "yes",
) -> str:
    """Build valid verdict-extraction JSON where every member votes the same way.

    With all members on one ``position_label`` the deterministic consensus is
    1.0 ("unanimous"), giving a real, non-None consensus_score to prove mirroring.
    ``provider_votes[*].provider`` matches the member names so the engine's
    per-member sequence resolves.
    """
    return json.dumps(
        {
            "verdict_applies": verdict_applies,
            "verdict_type": verdict_type,
            "headline": "Yes.",
            "recommendation": "Proceed with yes.",
            "positions": [
                {
                    "label": position_label,
                    "summary": "All members agree: yes.",
                    "providers": list(members),
                    "evidence_answer_ids": [],
                }
            ],
            "provider_votes": [
                {"provider": name, "position_label": position_label} for name in members
            ],
            "minority_reports": [],
            "conflicts": [],
            "caveats": [],
            "dissent_summary": None,
        }
    )


# --------------------------------------------------------------------------- #
# 1. Verdict present (happy path)
# --------------------------------------------------------------------------- #
async def test_verdict_present_happy_path(monkeypatch, patch_call_model):
    """A valid extraction yields a verdict, hoisted mirrors, and manifest provenance."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    # Verdict present.
    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    assert result.verdict.consensus_label == "unanimous"
    assert result.verdict.consensus_method == CONSENSUS_METHOD

    # Top-level fields mirror the canonical verdict exactly.
    assert result.consensus_score == result.verdict.consensus_score
    assert result.consensus_method == result.verdict.consensus_method
    assert result.consensus_label == result.verdict.consensus_label
    assert result.conflicts == result.verdict.conflicts
    assert result.provider_votes == result.verdict.provider_votes
    assert result.minority_reports == result.verdict.minority_reports

    # Manifest provenance populated and self-consistent with the verdict.
    assert result.manifest is not None
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.verdict_extraction.prompt_version is not None
    assert result.manifest.verdict_type == result.verdict.verdict_type
    assert result.manifest.consensus_method == result.verdict.consensus_method
    assert result.manifest.verdict_absent_reason is None

    # The VERIFIED stamp survives populating the verdict provenance.
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED

    # Prose synthesis is unaffected (additive).
    assert result.synthesis is not None


# --------------------------------------------------------------------------- #
# 2. Verdict absent -- open-ended
# --------------------------------------------------------------------------- #
async def test_verdict_absent_open_ended(monkeypatch, patch_call_model):
    """verdict_applies=false leaves verdict None but still records the extractor."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(verdict_applies=False, verdict_type="synthesis"))
        return make_response(f"some prose from {model}")

    patch_call_model(handler)
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("Write me a poem about the sea.")

    assert result.verdict is None
    # Mirror fields stay at their None/empty defaults.
    assert result.consensus_score is None
    assert result.consensus_method is None
    assert result.consensus_label is None
    assert result.conflicts == []
    assert result.provider_votes == []
    assert result.minority_reports == []
    # Provenance: extractor still recorded; absent reason explains why.
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_OPEN_ENDED
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.consensus_method is None
    assert result.manifest.verdict_type is None
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# 3. Verdict absent -- extraction fails (prose, can't parse)
# --------------------------------------------------------------------------- #
async def test_verdict_absent_extraction_fails(monkeypatch, patch_call_model):
    """Prose for the verdict call -> parse+repair fail -> verdict None, reason set."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        # Prose for EVERY call, including the verdict call and its repair retry.
        return make_response(f"this is prose, not JSON, from {model}")

    patch_call_model(handler)
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("Should we adopt Rust?")

    assert result.verdict is None
    assert result.consensus_score is None
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_EXTRACTION_FAILED
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# 4. Verdict absent -- N<2 responders (verdict seam never called)
# --------------------------------------------------------------------------- #
async def test_verdict_absent_too_few_responders(monkeypatch, patch_call_model):
    """One responding member -> extract_verdict short-circuits before any LLM call."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        # If the verdict-extraction call ever fires with N<2, that's a bug.
        if _is_verdict_call(messages):
            raise AssertionError("verdict extraction must not be called with <2 responders")
        # Only grok responds; gemini errors out so just one member is "responding".
        if model == "gemini/gemini-2.5-pro":
            raise RuntimeError("provider down")
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    assert result.verdict is None
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_TOO_FEW
    # Provenance still recorded even though no extraction call was made.
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# 5. Opt-out: extract_verdict=False
# --------------------------------------------------------------------------- #
async def test_extract_verdict_opt_out(monkeypatch, patch_call_model):
    """With extract_verdict=False the verdict is skipped entirely; prose still runs."""
    _all_keys(monkeypatch)
    verdict_call_seen = {"hit": False}

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            verdict_call_seen["hit"] = True
            return make_response(_extraction_json())
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
        extract_verdict=False,
    )
    result = await council.ask("Should we ship?")

    # No verdict extraction attempted at all.
    assert verdict_call_seen["hit"] is False
    assert result.verdict is None
    assert result.consensus_score is None
    assert result.consensus_method is None
    assert result.consensus_label is None
    assert result.conflicts == []
    assert result.provider_votes == []
    assert result.minority_reports == []
    # Manifest verdict-provenance left at defaults.
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason is None
    assert result.manifest.verdict_extraction.model_id is None
    assert result.manifest.verdict_type is None
    assert result.manifest.consensus_method is None
    # Prose synthesis still works.
    assert result.synthesis is not None


# --------------------------------------------------------------------------- #
# 6. Raw mode: no verdict extraction
# --------------------------------------------------------------------------- #
async def test_raw_mode_no_verdict(monkeypatch, patch_call_model):
    """synthesize=False never triggers verdict extraction (no synthesizer call)."""
    _all_keys(monkeypatch)
    verdict_call_seen = {"hit": False}

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            verdict_call_seen["hit"] = True
            return make_response(_extraction_json())
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?", synthesize=False)

    assert verdict_call_seen["hit"] is False
    assert result.verdict is None
    assert result.synthesis is None  # raw mode never synthesizes
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason is None


# --------------------------------------------------------------------------- #
# 7. Shared helper reusable / single path (proves CAC-06-STREAM can reuse it)
# --------------------------------------------------------------------------- #
async def test_apply_verdict_helper_directly(monkeypatch, patch_call_model):
    """Calling _apply_verdict directly on a hand-built result populates everything."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response("unused")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())

    # Hand-build a result with answers + a minimal manifest (as the stream path will).
    result = CouncilResult(
        prompt="Should we ship?",
        answers=[
            ModelAnswer(name="grok", model_id="xai/grok-4.3", answer="yes", answer_id="grok-1"),
            ModelAnswer(
                name="gemini",
                model_id="gemini/gemini-2.5-pro",
                answer="yes",
                answer_id="gemini-1",
            ),
        ],
        manifest=ModelHarnessManifest(request_id="r1", conclave_version="0.0.0", mode="synthesize"),
    )

    await council._apply_verdict(result)

    assert result.verdict is not None
    assert result.consensus_score == result.verdict.consensus_score == 1.0
    assert result.provider_votes == result.verdict.provider_votes
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.verdict_type == result.verdict.verdict_type
    assert result.manifest.consensus_method == CONSENSUS_METHOD
    assert result.manifest.verdict_absent_reason is None


async def test_apply_verdict_helper_no_manifest(monkeypatch, patch_call_model):
    """_apply_verdict tolerates a result with manifest=None (mirrors still hoisted)."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response("unused")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = CouncilResult(
        prompt="Should we ship?",
        answers=[
            ModelAnswer(name="grok", model_id="xai/grok-4.3", answer="yes"),
            ModelAnswer(name="gemini", model_id="gemini/gemini-2.5-pro", answer="yes"),
        ],
        manifest=None,
    )

    await council._apply_verdict(result)

    assert result.verdict is not None
    assert result.consensus_score == 1.0
    assert result.manifest is None  # no manifest to populate, and no crash


# --------------------------------------------------------------------------- #
# 8. Backward-compat smoke: prose-only handler degrades gracefully
# --------------------------------------------------------------------------- #
async def test_backward_compat_prose_handler_degrades(monkeypatch, patch_call_model):
    """An old-style prose handler keeps synthesis/answers and adds a graceful None verdict."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        # Classic handler: prose everywhere, no verdict-call branch.
        if model == _SYNTH_MODEL_ID and len(messages) == 2:
            return make_response("MERGED")
        return make_response(f"answer from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"], synthesizer="claude", config=_config()
    )
    result = await council.ask("What is 2+2?")

    # Existing behavior intact.
    assert len(result.answers) == 3
    assert all(a.ok for a in result.answers)
    assert result.synthesis == "MERGED"
    assert result.synthesizer == "claude"
    # Default-on verdict degrades gracefully (prose can't be parsed as extraction).
    assert result.verdict is None
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_EXTRACTION_FAILED


def test_apply_verdict_opt_out_is_noop_no_calls(monkeypatch, patch_call_model):
    """Sync guard: opt-out short-circuits before importing/calling the engine."""
    import asyncio

    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            raise AssertionError("verdict seam must not be called when opted out")
        return make_response("prose")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini"],
        synthesizer="claude",
        config=_config(),
        extract_verdict=False,
    )
    result = CouncilResult(
        prompt="x",
        answers=[
            ModelAnswer(name="grok", model_id="xai/grok-4.3", answer="yes"),
            ModelAnswer(name="gemini", model_id="gemini/gemini-2.5-pro", answer="yes"),
        ],
        manifest=ModelHarnessManifest(request_id="r1", conclave_version="0.0.0", mode="synthesize"),
    )
    asyncio.run(council._apply_verdict(result))
    assert result.verdict is None
    assert result.manifest.verdict_extraction.model_id is None
