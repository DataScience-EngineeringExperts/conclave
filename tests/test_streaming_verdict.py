"""Tests for CAC-06-STREAM: the streamed ``done`` result carries manifest + verdict.

All tests run offline. The streaming path resolves member tokens through the
MEMBER stream seam (``conclave.streaming.call_model_stream``) and the structured
verdict through the verdict seam (``conclave.verdict_synthesis.call_model``). The
buffered path resolves members through ``conclave.council.call_model`` instead.
So a streaming-verdict test patches BOTH:

* the member stream seam -- via a local ``fake_stream`` installed with
  ``monkeypatch.setattr(streaming_mod, "call_model_stream", ...)``, replicating
  ``test_streaming.py``'s ``_patch_stream`` helper signature exactly, and
* the verdict ``call_model`` seam -- via the shared ``patch_call_model`` fixture
  (which also covers the council member seam used by the buffered ``Council.ask``
  comparison run in the parity test).

A real verdict flows when the handler returns extraction JSON for the call whose
system message starts with "You are the verdict extractor" (see ``conftest`` and
``test_council_verdict.py``); prose otherwise degrades gracefully to
``verdict=None``. The autouse ``_offline_verdict_seam`` fixture keeps the verdict
seam network-safe by default, so no test reaches the network. Warning assertions
use the ``conclave_caplog`` fixture because the ``conclave`` logger does not
propagate.

The load-bearing test is ``test_streaming_nonstreaming_verdict_parity``: it drives
the SAME member text + SAME extraction JSON through both the streamed path and
``Council.ask`` and asserts the two verdicts (and manifest provenance) match
field by field -- the proof that CAC-06-STREAM made the "byte-for-byte identical"
claim literal.
"""

from __future__ import annotations

import json

from conclave import Council
from conclave.config import ConclaveConfig
from conclave.manifest import SECRET_SAFETY_VERIFIED
from conclave.models import ModelAnswer
from conclave.verdict import CONSENSUS_METHOD
from conclave.verdict_synthesis import (
    _REASON_EXTRACTION_FAILED,
    _REASON_OPEN_ENDED,
)
from tests.conftest import make_response

# The synthesizer/extractor resolved id for the "claude" friendly name below.
_SYNTH_MODEL_ID = "anthropic/claude-sonnet-4-6"


# --------------------------------------------------------------------------- #
# Shared offline harness helpers (mock data lives only in tests)
# --------------------------------------------------------------------------- #


def _all_keys(monkeypatch) -> None:
    """Set every provider key used by the deterministic config to a dummy value."""
    for var in (
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
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
    1.0 ("unanimous"), giving a real, non-None consensus_score to prove the
    streamed ``done`` carries a true verdict. ``provider_votes[*].provider``
    matches the member names so the engine's per-member sequence resolves.
    Mirrors the shape in ``tests/test_council_verdict.py``.
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


def _patch_member_stream(monkeypatch, deltas_by_model, errors_by_model=None) -> None:
    """Patch the MEMBER stream seam with canned deltas + a final answer per model.

    Replicates ``test_streaming.py``'s ``_patch_stream`` signature exactly so the
    council's ``_drive_member`` consumes it unchanged: ``fake_stream`` yields each
    text part then a terminal :class:`ModelAnswer`. When a model id is in
    ``errors_by_model`` the terminal answer carries an ``error`` (partial deltas
    are still yielded first), exercising the mid-stream-failure contract.
    """
    import conclave.streaming as streaming_mod

    errors_by_model = errors_by_model or {}

    async def fake_stream(
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None, **kwargs
    ):
        text_parts = deltas_by_model.get(model_id, ["x"])
        for part in text_parts:
            yield part
        err = errors_by_model.get(model_id)
        if err is not None:
            # Partial text preserved on the errored answer, just like the real
            # call_model_stream mid-stream-failure contract.
            yield ModelAnswer(name=name, model_id=model_id, answer="".join(text_parts), error=err)
        else:
            yield ModelAnswer(name=name, model_id=model_id, answer="".join(text_parts))

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)


# --------------------------------------------------------------------------- #
# 1. Verdict present in the terminal done event
# --------------------------------------------------------------------------- #
async def test_streamed_done_carries_verdict(monkeypatch, patch_call_model):
    """A streamed synthesize run's ``done`` result carries a real, hoisted verdict."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    _patch_member_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["yes ", "grok"],
            "gemini/gemini-2.5-pro": ["yes ", "gemini"],
            "perplexity/sonar-pro": ["yes ", "perplexity"],
            _SYNTH_MODEL_ID: ["SYN"],  # synthesizer prose stream
        },
    )

    def verdict_handler(model, messages, **kwargs):
        # Only the verdict-extraction call flows through the call_model seam here;
        # members stream via the seam patched above.
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response("unused prose")

    patch_call_model(verdict_handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("Should we ship?", synthesize=True)]

    done = events[-1]
    assert done.type == "done"
    result = done.result
    assert result is not None

    # Verdict present and hoisted onto the terminal result.
    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    assert result.verdict.consensus_label == "unanimous"
    assert result.verdict.consensus_method == CONSENSUS_METHOD
    assert result.consensus_score == 1.0
    assert result.consensus_label == "unanimous"
    assert result.consensus_method == CONSENSUS_METHOD


# --------------------------------------------------------------------------- #
# 2. Streaming / non-streaming PARITY (load-bearing)
# --------------------------------------------------------------------------- #
async def test_streaming_nonstreaming_verdict_parity(monkeypatch, patch_call_model):
    """The streamed ``done`` verdict equals ``Council.ask``'s for identical inputs.

    THE parity proof. A single shared handler produces the same member text + the
    same extraction JSON, wired into both the ``call_model`` seam (members for
    ``Council.ask`` plus the verdict seam for both runs) AND a ``call_model_stream``
    fake (members for the streamed run) that returns the SAME member text. The two
    resulting verdict objects -- and the manifest verdict-provenance slots -- are
    compared field by field.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")
    member_text = {
        "xai/grok-4.3": "yes grok",
        "gemini/gemini-2.5-pro": "yes gemini",
        "perplexity/sonar-pro": "yes perplexity",
    }

    def shared_handler(model, messages, **kwargs):
        # Same logic on every call_model seam: extraction JSON for the verdict
        # call, the member's canned text for everything else.
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response(member_text.get(model, "SYN"))

    # Buffered run: call_model seam covers members + verdict.
    patch_call_model(shared_handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    buffered = await council.ask("Should we ship?", synthesize=True)

    # Streamed run: same verdict seam (still patched) + matching member stream.
    _patch_member_stream(
        monkeypatch,
        {model_id: [text] for model_id, text in member_text.items()} | {_SYNTH_MODEL_ID: ["SYN"]},
    )
    council_stream = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council_stream.ask_stream("Should we ship?", synthesize=True)]
    streamed = events[-1].result

    # Both produced a verdict.
    assert buffered.verdict is not None
    assert streamed.verdict is not None

    # Verdict object fields match field-by-field.
    assert streamed.verdict.verdict_type == buffered.verdict.verdict_type
    assert streamed.verdict.consensus_score == buffered.verdict.consensus_score
    assert streamed.verdict.consensus_method == buffered.verdict.consensus_method
    assert streamed.verdict.consensus_label == buffered.verdict.consensus_label
    assert streamed.verdict.conflicts == buffered.verdict.conflicts
    assert streamed.verdict.provider_votes == buffered.verdict.provider_votes
    assert streamed.verdict.minority_reports == buffered.verdict.minority_reports

    # Hoisted top-level mirrors match too.
    assert streamed.consensus_score == buffered.consensus_score
    assert streamed.consensus_method == buffered.consensus_method
    assert streamed.consensus_label == buffered.consensus_label

    # Manifest verdict-provenance slots match between the two paths.
    assert streamed.manifest is not None and buffered.manifest is not None
    assert (
        streamed.manifest.verdict_extraction.model_id
        == buffered.manifest.verdict_extraction.model_id
        == _SYNTH_MODEL_ID
    )
    assert streamed.manifest.consensus_method == buffered.manifest.consensus_method
    assert streamed.manifest.verdict_type == buffered.manifest.verdict_type
    assert streamed.manifest.verdict_absent_reason == buffered.manifest.verdict_absent_reason


# --------------------------------------------------------------------------- #
# 3. Manifest present on the streamed done result
# --------------------------------------------------------------------------- #
async def test_streamed_done_carries_manifest(monkeypatch, patch_call_model):
    """The streamed ``done`` result has a populated, VERIFIED-stamped manifest."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    _patch_member_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["yes"],
            "gemini/gemini-2.5-pro": ["yes"],
            "perplexity/sonar-pro": ["yes"],
            _SYNTH_MODEL_ID: ["SYN"],
        },
    )

    def verdict_handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response("unused")

    patch_call_model(verdict_handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("Should we ship?", synthesize=True)]
    result = events[-1].result

    assert result.manifest is not None
    # Provider provenance populated from the membership.
    assert result.manifest.providers_called == ["grok", "gemini", "perplexity"]
    assert len(result.manifest.receipts) == 3
    # Verdict provenance + VERIFIED stamp survive _apply_verdict.
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.consensus_method == CONSENSUS_METHOD
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# 4. Mid-stream member failure: partial text preserved, verdict still present
# --------------------------------------------------------------------------- #
async def test_midstream_member_failure_preserves_partial_and_verdict(
    monkeypatch, patch_call_model
):
    """A failing member keeps its partial deltas; >=2 others succeed -> verdict present."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    _patch_member_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["par", "tial"],  # this one errors after partial text
            "gemini/gemini-2.5-pro": ["yes"],
            "perplexity/sonar-pro": ["yes"],
            _SYNTH_MODEL_ID: ["SYN"],
        },
        errors_by_model={"xai/grok-4.3": "network error: stream dropped"},
    )

    def verdict_handler(model, messages, **kwargs):
        # gemini + perplexity respond successfully -> >=2 responders -> real verdict.
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=("gemini", "perplexity")))
        return make_response("unused")

    patch_call_model(verdict_handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("Should we ship?", synthesize=True)]

    # The failing member's partial deltas were emitted before its error.
    grok_deltas = [e.text for e in events if e.type == "member_delta" and e.name == "grok"]
    assert grok_deltas == ["par", "tial"]

    # The run completed cleanly with a terminal done event.
    assert events[-1].type == "done"
    result = events[-1].result

    # The failed member is recorded as an error; the others succeeded.
    by_name = {a.name: a for a in result.answers}
    assert not by_name["grok"].ok
    assert by_name["gemini"].ok
    assert by_name["perplexity"].ok

    # With >=2 successful responders a verdict is still present.
    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    assert result.manifest is not None
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# 5a. Verdict absent -- open-ended (verdict_applies=False)
# --------------------------------------------------------------------------- #
async def test_streamed_verdict_absent_open_ended(monkeypatch, patch_call_model):
    """verdict_applies=false -> done result verdict None, manifest reason OPEN_ENDED."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini")

    _patch_member_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["prose"],
            "gemini/gemini-2.5-pro": ["prose"],
            _SYNTH_MODEL_ID: ["SYN"],
        },
    )

    def verdict_handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(
                _extraction_json(verdict_applies=False, verdict_type="synthesis", members=members)
            )
        return make_response("unused")

    patch_call_model(verdict_handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("Write me a poem.", synthesize=True)]

    result = events[-1].result
    assert events[-1].type == "done"
    assert result.verdict is None
    assert result.consensus_score is None
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_OPEN_ENDED
    # Extractor still recorded; stamp intact.
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# 5b. Verdict absent -- extraction fails (prose can't parse), warning emitted
# --------------------------------------------------------------------------- #
async def test_streamed_verdict_absent_extraction_fails(
    monkeypatch, patch_call_model, conclave_caplog
):
    """Prose for the verdict call -> verdict None, reason EXTRACTION_FAILED, clean done."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini")

    _patch_member_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["prose"],
            "gemini/gemini-2.5-pro": ["prose"],
            _SYNTH_MODEL_ID: ["SYN"],
        },
    )

    def prose_handler(model, messages, **kwargs):
        # Prose for EVERY call_model call, including the verdict call + repair retry.
        return make_response(f"this is prose, not JSON, from {model}")

    patch_call_model(prose_handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("Should we adopt Rust?", synthesize=True)]

    result = events[-1].result
    assert events[-1].type == "done"
    assert result.verdict is None
    assert result.consensus_score is None
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_EXTRACTION_FAILED
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED
    # The extraction-failure path logs a warning on the non-propagating logger.
    assert any(rec.levelname == "WARNING" for rec in conclave_caplog.records)


# --------------------------------------------------------------------------- #
# 6. Opt-out: extract_verdict=False -> no verdict, manifest still present
# --------------------------------------------------------------------------- #
async def test_streamed_opt_out_no_verdict_but_manifest_present(monkeypatch, patch_call_model):
    """extract_verdict=False streamed run: verdict None, provenance at defaults, manifest present."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")
    verdict_call_seen = {"hit": False}

    _patch_member_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["yes ", "grok"],
            "gemini/gemini-2.5-pro": ["yes ", "gemini"],
            "perplexity/sonar-pro": ["yes ", "perplexity"],
            _SYNTH_MODEL_ID: ["SYN"],
        },
    )

    def verdict_handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            verdict_call_seen["hit"] = True
            return make_response(_extraction_json(members=members))
        return make_response("unused")

    patch_call_model(verdict_handler)
    council = Council(
        models=list(members),
        synthesizer="claude",
        config=_config(),
        extract_verdict=False,
    )
    events = [e async for e in council.ask_stream("Should we ship?", synthesize=True)]

    # Member deltas still streamed.
    grok_deltas = [e.text for e in events if e.type == "member_delta" and e.name == "grok"]
    assert grok_deltas == ["yes ", "grok"]

    result = events[-1].result
    assert events[-1].type == "done"
    # Opt-out: verdict extraction never attempted, no verdict.
    assert verdict_call_seen["hit"] is False
    assert result.verdict is None
    assert result.consensus_score is None
    # Manifest still present (proves _apply_verdict no-ops cleanly), provenance at defaults.
    assert result.manifest is not None
    assert result.manifest.verdict_extraction.model_id is None
    assert result.manifest.verdict_absent_reason is None
    assert result.manifest.verdict_type is None
    assert result.manifest.consensus_method is None
