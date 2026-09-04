"""CAC-07 integration sweep: auditability, parity, and structured-output degradation.

A hardening cross-cut over the default-on verdict pipeline. Every test drives a
REAL verdict end-to-end through :meth:`conclave.council.Council.ask` (and, for the
parity item, also through ``ask_stream``) offline via the dual-seam harness in
``conftest`` (``patch_call_model`` patches BOTH the council member seam and the
verdict seam) plus the member stream seam fake copied from
``tests/test_streaming_verdict.py``. No network, ever.

Sweep items covered here (see the CAC-07 brief):

* **Item 2 — auditability invariant.** The consensus number on the result is the
  deterministic ``position_cluster_ratio_v1`` arithmetic over the model's
  clustering (:mod:`conclave.agreement`), NEVER a number the extractor model
  emitted. A BOGUS top-level ``consensus_score`` smuggled into the extraction JSON
  is dropped by :class:`conclave.verdict.VerdictExtractionModel` (Pydantic
  ``extra="ignore"``) and never reaches the verdict — proven for a unanimous
  clustering (computed 1.0 beats smuggled 0.123) AND a split clustering (computed
  2/3 beats smuggled 1.0).
* **Item 5 — streaming<->non-streaming parity (NEW scenarios).** A SPLIT (2 vs 1)
  verdict and a verdict-ABSENT (open-ended) run each produce a field-by-field
  identical verdict + manifest verdict-provenance through ``ask`` and
  ``ask_stream``. These are scenarios the existing ``test_streaming_verdict.py``
  (which only covers the unanimous case) does not exercise.
* **Item 6 — structured-output degradation.** The extractor's JSON wrapped in a
  ```` ```json ... ``` ```` markdown code fence still yields a valid verdict via the
  ``_strip_code_fence`` path (the plain-text case is the default elsewhere).

The test-local helpers (``_all_keys``, ``_config``, ``_is_verdict_call``,
``_extraction_json``, ``_patch_member_stream``) are copied VERBATIM from
``tests/test_council_verdict.py`` / ``tests/test_streaming_verdict.py`` because
those helpers are per-file, not shared. Mock data lives only in tests.
"""

from __future__ import annotations

import json

from conclave import Council
from conclave.config import ConclaveConfig
from conclave.models import ModelAnswer
from conclave.verdict import CONSENSUS_METHOD
from conclave.verdict_synthesis import _REASON_OPEN_ENDED
from tests.conftest import make_response

# The synthesizer/extractor resolved id for the "claude" friendly name below.
_SYNTH_MODEL_ID = "anthropic/claude-sonnet-4-6"


# --------------------------------------------------------------------------- #
# Test-local harness helpers (copied verbatim from the canonical templates).
# --------------------------------------------------------------------------- #
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
    1.0 ("unanimous"). ``provider_votes[*].provider`` matches the member names so
    the engine's per-member sequence resolves. Copied from
    ``tests/test_council_verdict.py``.
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
    council's ``_drive_member`` consumes it unchanged. Copied verbatim from
    ``tests/test_streaming_verdict.py``.
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
            yield ModelAnswer(name=name, model_id=model_id, answer="".join(text_parts), error=err)
        else:
            yield ModelAnswer(name=name, model_id=model_id, answer="".join(text_parts))

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)


# --------------------------------------------------------------------------- #
# Custom extraction-JSON builders for clustering that differs across members.
# --------------------------------------------------------------------------- #
def _smuggled_unanimous_extraction(members: tuple[str, ...], *, smuggled_score: float) -> str:
    """Unanimous clustering (all members on "yes") WITH a smuggled consensus_score.

    Every member votes "yes" so the deterministic ratio is 1.0. The top-level
    ``consensus_score`` is a BOGUS number the validator must drop (``extra="ignore"``);
    it must never reach the assembled verdict. Drives sweep item 2 (unanimous arm).
    """
    payload = {
        # The smuggled field the model must never get to set — sits at the top
        # level exactly where a real consensus field would be, so dropping it is
        # the load-bearing auditability proof.
        "consensus_score": smuggled_score,
        "verdict_applies": True,
        "verdict_type": "decision",
        "headline": "Yes.",
        "recommendation": "Proceed.",
        "positions": [
            {
                "label": "yes",
                "summary": "All members agree: yes.",
                "providers": list(members),
                "evidence_answer_ids": [],
            }
        ],
        "provider_votes": [{"provider": name, "position_label": "yes"} for name in members],
        "minority_reports": [],
        "conflicts": [],
        "caveats": [],
        "dissent_summary": None,
    }
    return json.dumps(payload)


def _smuggled_split_extraction(*, smuggled_score: float) -> str:
    """A 2-yes / 1-no clustering WITH a smuggled consensus_score.

    Three members: grok+gemini vote "yes", perplexity votes "no". The engine maps
    every responding member to its ``provider_votes`` entry (keyed on member name),
    so the per-member sequence is ["yes","yes","no"] -> 2/3 -> "majority". The
    smuggled top-level ``consensus_score`` (1.0) must be dropped; the computed 2/3
    must win. positions[] mirrors the same split for human-readability; only
    provider_votes drives the arithmetic (confirmed in verdict_synthesis
    ``_member_vote_sequence``). Drives sweep item 2 (split arm).
    """
    payload = {
        "consensus_score": smuggled_score,  # BOGUS — must be ignored.
        "verdict_applies": True,
        "verdict_type": "decision",
        "headline": "Mostly yes.",
        "recommendation": "Proceed with one dissent.",
        "positions": [
            {
                "label": "yes",
                "summary": "Two members say yes.",
                "providers": ["grok", "gemini"],
                "evidence_answer_ids": [],
            },
            {
                "label": "no",
                "summary": "One member says no.",
                "providers": ["perplexity"],
                "evidence_answer_ids": [],
            },
        ],
        "provider_votes": [
            {"provider": "grok", "position_label": "yes"},
            {"provider": "gemini", "position_label": "yes"},
            {"provider": "perplexity", "position_label": "no"},
        ],
        "minority_reports": [],
        "conflicts": [],
        "caveats": [],
        "dissent_summary": None,
    }
    return json.dumps(payload)


# --------------------------------------------------------------------------- #
# Item 2 — auditability: consensus is deterministic, never the LLM's number.
# --------------------------------------------------------------------------- #
async def test_smuggled_consensus_score_ignored_unanimous(monkeypatch, patch_call_model):
    """A bogus top-level consensus_score (0.123) is dropped; the computed 1.0 wins.

    Sweep item 2 (unanimous arm). The extractor JSON smuggles
    ``consensus_score=0.123`` at the top level alongside a real all-"yes"
    clustering. ``VerdictExtractionModel``'s ``extra="ignore"`` drops the smuggled
    field, so the engine's deterministic arithmetic (3/3 = 1.0) is what surfaces.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_smuggled_unanimous_extraction(members, smuggled_score=0.123))
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    assert result.verdict is not None
    # The computed unanimous ratio wins; the smuggled 0.123 never appears.
    assert result.verdict.consensus_score == 1.0
    assert result.consensus_score == 1.0
    assert result.verdict.consensus_label == "unanimous"
    assert result.verdict.consensus_method == CONSENSUS_METHOD
    assert result.consensus_score != 0.123


async def test_smuggled_consensus_score_ignored_split(monkeypatch, patch_call_model):
    """A bogus top-level consensus_score (1.0) is dropped; the computed 2/3 wins.

    Sweep item 2 (split arm). The extractor JSON smuggles ``consensus_score=1.0``
    alongside a 2-yes/1-no clustering. The engine computes 2/3 -> "majority"
    deterministically from ``provider_votes``; the smuggled 1.0 ("unanimous") must
    never stand. Uses ``pytest.approx(2/3)`` for the float.
    """
    import pytest

    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_smuggled_split_extraction(smuggled_score=1.0))
        return make_response(f"answer from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    assert result.verdict is not None
    # Deterministic 2/3 from the clustering — NOT the smuggled 1.0.
    assert result.verdict.consensus_score == pytest.approx(2 / 3)
    assert result.consensus_score == pytest.approx(2 / 3)
    assert result.verdict.consensus_label == "majority"
    assert result.verdict.consensus_method == CONSENSUS_METHOD
    # The smuggled unanimous value is provably gone.
    assert result.verdict.consensus_score != 1.0
    assert result.verdict.consensus_label != "unanimous"


# --------------------------------------------------------------------------- #
# Item 6 — structured-output degradation: fenced extraction JSON still resolves.
# --------------------------------------------------------------------------- #
async def test_fenced_extraction_json_yields_verdict(monkeypatch, patch_call_model):
    """Extraction JSON wrapped in a ```json fence still resolves via _strip_code_fence.

    Sweep item 6. The offline seam returns whatever the handler gives, so we wrap a
    valid extraction object in a Markdown ```json ... ``` fence (what a model may
    emit despite the "JSON only" instruction). The engine's ``_strip_code_fence``
    unwraps it, so the verdict resolves end-to-end exactly as the plain-text case.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")
    inner = _extraction_json(members=members)
    fenced = f"```json\n{inner}\n```"

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(fenced)
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    assert result.verdict.consensus_label == "unanimous"
    assert result.verdict.consensus_method == CONSENSUS_METHOD
    assert result.consensus_score == 1.0


async def test_plain_text_extraction_json_yields_verdict(monkeypatch, patch_call_model):
    """Control for item 6: the same JSON WITHOUT a fence also resolves.

    Pins that the fenced/plain pair both succeed, so a regression that broke
    ``_strip_code_fence`` (fenced fails) would be distinguishable from a broken
    extraction parse (both fail).
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0


# --------------------------------------------------------------------------- #
# Item 5 — streaming<->non-streaming parity: NEW scenarios (split + absent).
# --------------------------------------------------------------------------- #
async def test_streaming_nonstreaming_parity_split(monkeypatch, patch_call_model):
    """A SPLIT (2 vs 1) verdict is field-by-field identical via ask and ask_stream.

    Sweep item 5 (NEW scenario the existing unanimous parity test does not cover).
    The SAME member text + SAME split extraction JSON drive both paths; the two
    verdict objects, the hoisted mirrors, and the manifest verdict-provenance must
    match exactly, including the computed 2/3 "majority" consensus.
    """
    import pytest

    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")
    member_text = {
        "xai/grok-4.3": "yes grok",
        "gemini/gemini-2.5-pro": "yes gemini",
        "perplexity/sonar-pro": "no perplexity",
    }

    def shared_handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_smuggled_split_extraction(smuggled_score=1.0))
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

    assert buffered.verdict is not None
    assert streamed.verdict is not None

    # The split was actually computed (not a unanimous accident).
    assert buffered.verdict.consensus_score == pytest.approx(2 / 3)
    assert buffered.verdict.consensus_label == "majority"

    # Verdict object fields match field-by-field across the two paths.
    assert streamed.verdict.verdict_type == buffered.verdict.verdict_type
    assert streamed.verdict.consensus_score == buffered.verdict.consensus_score
    assert streamed.verdict.consensus_method == buffered.verdict.consensus_method
    assert streamed.verdict.consensus_label == buffered.verdict.consensus_label
    assert streamed.verdict.conflicts == buffered.verdict.conflicts
    assert streamed.verdict.provider_votes == buffered.verdict.provider_votes
    assert streamed.verdict.minority_reports == buffered.verdict.minority_reports

    # Hoisted top-level mirrors match.
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


async def test_streaming_nonstreaming_parity_verdict_absent_open_ended(
    monkeypatch, patch_call_model
):
    """A verdict-ABSENT (open-ended) run is identical via ask and ask_stream.

    Sweep item 5 (NEW absent-parity scenario). With ``verdict_applies=false`` both
    paths must produce ``verdict is None``, ``consensus_score is None``, and the
    SAME manifest verdict-provenance: same extractor id, same ``OPEN_ENDED`` absent
    reason, and ``consensus_method``/``verdict_type`` both None. synthesis +
    member_answers survive on both.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini")
    member_text = {
        "xai/grok-4.3": "a poem about the sea",
        "gemini/gemini-2.5-pro": "another poem",
    }

    def shared_handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(
                _extraction_json(verdict_applies=False, verdict_type="synthesis", members=members)
            )
        return make_response(member_text.get(model, "SYN"))

    patch_call_model(shared_handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    buffered = await council.ask("Write me a poem about the sea.", synthesize=True)

    _patch_member_stream(
        monkeypatch,
        {model_id: [text] for model_id, text in member_text.items()} | {_SYNTH_MODEL_ID: ["SYN"]},
    )
    council_stream = Council(models=list(members), synthesizer="claude", config=_config())
    events = [
        e
        async for e in council_stream.ask_stream("Write me a poem about the sea.", synthesize=True)
    ]
    streamed = events[-1].result

    # Both absent, identically.
    assert buffered.verdict is None
    assert streamed.verdict is None
    assert buffered.consensus_score is None
    assert streamed.consensus_score is None

    # synthesis + member answers survive on both paths (DD-2 absent rule).
    assert buffered.member_answers and streamed.member_answers
    assert len(buffered.member_answers) == len(streamed.member_answers) == 2
    assert buffered.synthesis is not None
    assert streamed.synthesis is not None

    # Manifest verdict-provenance matches: extractor recorded, OPEN_ENDED reason,
    # method/type None on both.
    assert streamed.manifest is not None and buffered.manifest is not None
    assert (
        streamed.manifest.verdict_extraction.model_id
        == buffered.manifest.verdict_extraction.model_id
        == _SYNTH_MODEL_ID
    )
    assert (
        streamed.manifest.verdict_absent_reason
        == buffered.manifest.verdict_absent_reason
        == _REASON_OPEN_ENDED
    )
    assert streamed.manifest.consensus_method is buffered.manifest.consensus_method is None
    assert streamed.manifest.verdict_type is buffered.manifest.verdict_type is None
