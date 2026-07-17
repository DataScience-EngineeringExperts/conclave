"""CAC-07 verdict edge-case sweep (items 3 + 4), end-to-end through ``Council.ask``.

Every test drives a real verdict (or a real verdict-absence) through
:meth:`conclave.council.Council.ask` offline via the dual-seam ``patch_call_model``
harness in ``conftest``. No network. The constants for the three absent reasons are
IMPORTED from :mod:`conclave.verdict_synthesis` (never hardcoded) so a wording change
there fails these tests loudly rather than silently passing on a stale literal.

Sweep items covered here (see the CAC-07 brief):

* **Item 3 — clustering/partial-failure edge cases.**
  - ``test_n1_responder_no_verdict_seam_call``: N=1 responding member -> verdict
    None, manifest reason ``_REASON_TOO_FEW``, and the verdict seam is NEVER called
    (the handler raises ``AssertionError`` if it sees a verdict call).
  - ``test_n2_tie_is_split``: N=2 with each member on a DIFFERENT position_label ->
    ``consensus_label == "split"`` and ``consensus_score == 0.5`` (a 1-of-2 tie).
  - ``test_partial_failure_verdict_over_responders_only``: >=2 succeed, 1 errors ->
    consensus computed over responders only; the errored member is recorded in the
    manifest's ``redacted_errors`` + as a receipt carrying ``error`` (NOT in
    ``providers_skipped`` — that slot is for missing-key skips only); verdict present.
  - ``test_all_members_fail_graceful``: every member errors -> no verdict, no crash,
    ``synthesis is None``, ``synthesis_error`` set, manifest attached + VERIFIED.

* **Item 4 — verdict-absent paths end-to-end.**
  - ``test_extraction_fails_both_calls``: non-JSON on the initial call AND the repair
    retry -> verdict None, reason ``_REASON_EXTRACTION_FAILED``, synthesis/answers
    survive.
  - ``test_repair_success_bad_then_good``: bad JSON the FIRST time the verdict seam
    is hit, GOOD extraction JSON the SECOND time (closure-counted) -> verdict
    PRESENT, proving the one-repair path actually repairs (a second seam call is
    made — confirmed in ``verdict_synthesis.extract_verdict`` step 3).
  - (the open-ended absent path is covered in ``test_integration_verdict.py`` via
    the parity test; ``test_council_verdict.py`` also covers it directly.)
"""

from __future__ import annotations

import json

from conclave import Council
from conclave.config import ConclaveConfig
from conclave.manifest import SECRET_SAFETY_VERIFIED
from conclave.verdict import CONSENSUS_METHOD
from conclave.verdict_synthesis import (
    _REASON_EXTRACTION_FAILED,
    _REASON_TOO_FEW,
)
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

    Copied from ``tests/test_council_verdict.py`` (per-file helper, not shared).
    All members on one ``position_label`` -> deterministic consensus 1.0
    ("unanimous").
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


def _tie_extraction_json(members: tuple[str, str], labels: tuple[str, str]) -> str:
    """A 2-member tie: each member on a DIFFERENT position_label.

    The engine maps each responding member to its ``provider_votes`` entry (keyed
    on member name), so the per-member sequence is ``[labels[0], labels[1]]`` ->
    1-of-2 -> 0.5 -> "split". Two distinct positions[] clusters mirror the tie for
    human-readability; only provider_votes drives the arithmetic.
    """
    payload = {
        "verdict_applies": True,
        "verdict_type": "decision",
        "headline": "Split.",
        "recommendation": "No consensus.",
        "positions": [
            {
                "label": labels[0],
                "summary": f"{members[0]} says {labels[0]}.",
                "providers": [members[0]],
                "evidence_answer_ids": [],
            },
            {
                "label": labels[1],
                "summary": f"{members[1]} says {labels[1]}.",
                "providers": [members[1]],
                "evidence_answer_ids": [],
            },
        ],
        "provider_votes": [
            {"provider": members[0], "position_label": labels[0]},
            {"provider": members[1], "position_label": labels[1]},
        ],
        "minority_reports": [],
        "conflicts": [],
        "caveats": [],
        "dissent_summary": None,
    }
    return json.dumps(payload)


# --------------------------------------------------------------------------- #
# Item 3 — N=1 responder: verdict absent, seam never called.
# --------------------------------------------------------------------------- #
async def test_n1_responder_no_verdict_seam_call(monkeypatch, patch_call_model):
    """One responding member -> verdict None, reason TOO_FEW, verdict seam unused.

    The N<2 gate in ``extract_verdict`` short-circuits BEFORE any LLM call, so the
    handler raises ``AssertionError`` if the verdict-extraction call ever fires —
    pinning that the gate makes no synthesizer round-trip.
    """
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            raise AssertionError("verdict extraction must not run with <2 responders")
        # Only grok responds; gemini errors out so just one member responds.
        if model == "gemini/gemini-2.5-pro":
            raise RuntimeError("provider down")
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    assert result.verdict is None
    assert result.consensus_score is None
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_TOO_FEW
    # Provenance still recorded even though no extraction call was made.
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# Item 3 — N=2 tie: split / 0.5.
# --------------------------------------------------------------------------- #
async def test_n2_tie_is_split(monkeypatch, patch_call_model):
    """Two members on different labels -> 1-of-2 tie -> 0.5 -> "split"."""
    _all_keys(monkeypatch)
    members = ("grok", "gemini")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_tie_extraction_json(members, ("yes", "no")))
        return make_response(f"answer from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    assert result.verdict is not None
    assert result.verdict.consensus_score == 0.5
    assert result.verdict.consensus_label == "split"
    assert result.verdict.consensus_method == CONSENSUS_METHOD
    # Hoisted mirrors agree.
    assert result.consensus_score == 0.5
    assert result.consensus_label == "split"


# --------------------------------------------------------------------------- #
# Item 3 — partial failure: consensus over responders only; manifest records error.
# --------------------------------------------------------------------------- #
async def test_partial_failure_verdict_over_responders_only(monkeypatch, patch_call_model):
    """One member errors, two succeed -> verdict over responders; manifest shows error.

    The errored member produces a ``ModelAnswer`` with ``error`` set, so it appears
    in the manifest's ``redacted_errors`` and as a receipt carrying ``error`` — but
    NOT in ``providers_skipped`` (that slot is reserved for missing-key skips,
    confirmed in council ``_build_manifest`` / ``_available_members``). The two
    responders give a unanimous verdict.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    def handler(model, messages, **kwargs):
        # perplexity errors at call time; grok + gemini respond.
        if model == "perplexity/sonar-pro":
            raise RuntimeError("provider 503")
        if _is_verdict_call(messages):
            # Only the two responders are in the extraction clustering.
            return make_response(_extraction_json(members=("grok", "gemini")))
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    # Three attempted; one failed, two ok.
    by_name = {a.name: a for a in result.answers}
    assert not by_name["perplexity"].ok
    assert by_name["grok"].ok
    assert by_name["gemini"].ok

    # Verdict present, computed over the two responders only (unanimous).
    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    assert result.verdict.consensus_label == "unanimous"

    manifest = result.manifest
    assert manifest is not None
    # The errored member is called-but-failed: it is NOT a no-key skip.
    assert manifest.providers_skipped == []
    assert manifest.providers_called == ["grok", "gemini", "perplexity"]
    # Its redacted error is recorded, and it has a receipt carrying that error.
    assert manifest.redacted_errors == ["provider_error"]
    assert len(manifest.redacted_errors) == 1
    perplexity_receipt = next(r for r in manifest.receipts if r.name == "perplexity")
    assert perplexity_receipt.error == "provider_error"
    assert perplexity_receipt.error_category == "provider_error"
    # Every actual call is retained: three members, prose synthesis, verdict.
    assert len(manifest.receipts) == 5
    assert [receipt.phase for receipt in manifest.receipts[-2:]] == [
        "synthesis",
        "verdict_extraction",
    ]
    assert manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# Item 3 — all members fail: graceful, no verdict, synthesis_error set.
# --------------------------------------------------------------------------- #
async def test_all_members_fail_graceful(monkeypatch, patch_call_model):
    """Every member errors -> no verdict, no crash, synthesis None + error set, manifest VERIFIED.

    With zero usable answers there is nothing to synthesize, so ``_synthesize``
    short-circuits leaving ``synthesis is None`` and
    ``synthesis_error == "no successful member answers to synthesize"`` (confirmed
    field name in models.py / council.py). ``extract_verdict``'s N<2 gate then
    leaves the verdict absent. A complete, VERIFIED manifest is still attached.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            raise AssertionError("verdict extraction must not run with 0 responders")
        # Every member call raises.
        raise RuntimeError(f"down: {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    # No member produced a usable answer.
    assert result.answers
    assert all(not a.ok for a in result.answers)

    # Graceful degradation: no verdict, synthesis absent + reason recorded.
    assert result.verdict is None
    assert result.consensus_score is None
    assert result.synthesis is None
    assert result.synthesis_error == "no successful member answers to synthesize"

    # Manifest still attached, complete, and VERIFIED.
    assert result.manifest is not None
    assert result.manifest.providers_called == ["grok", "gemini", "perplexity"]
    assert len(result.manifest.redacted_errors) == 3
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# Item 4 — extraction fails on BOTH the initial call AND the repair retry.
# --------------------------------------------------------------------------- #
async def test_extraction_fails_both_calls(monkeypatch, patch_call_model):
    """Non-JSON on initial + repair -> verdict None, reason EXTRACTION_FAILED, prose survives.

    The handler returns prose for EVERY call (members, the verdict-extraction call,
    and its repair retry), so parse+repair both fail and the verdict degrades to
    None with ``_REASON_EXTRACTION_FAILED``; member answers + synthesis are intact.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini")

    def handler(model, messages, **kwargs):
        return make_response(f"this is prose, not JSON, from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we adopt Rust?")

    assert result.verdict is None
    assert result.consensus_score is None
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason == _REASON_EXTRACTION_FAILED
    assert result.manifest.verdict_extraction.model_id == _SYNTH_MODEL_ID
    # Member answers + synthesis survive the absent verdict (DD-2).
    assert len(result.member_answers) == 2
    assert all(a.ok for a in result.member_answers)
    assert result.synthesis is not None
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# Item 4 — repair SUCCESS: bad JSON first, good JSON on the second seam call.
# --------------------------------------------------------------------------- #
async def test_repair_success_bad_then_good(monkeypatch, patch_call_model):
    """First verdict call returns bad JSON, the repair retry returns good JSON -> verdict PRESENT.

    Proves the one-repair path actually repairs. ``extract_verdict`` makes a SECOND
    ``call_model`` on the verdict seam when the first parse fails (confirmed: step 3
    of ``extract_verdict`` builds ``repair_messages`` and re-calls), so a
    closure-counted handler that fails once then succeeds drives a successful
    repair. We assert the seam was hit exactly twice.
    """
    _all_keys(monkeypatch)
    members = ("grok", "gemini")
    verdict_calls = {"count": 0}

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            verdict_calls["count"] += 1
            if verdict_calls["count"] == 1:
                # First extraction attempt: malformed JSON -> triggers one repair.
                return make_response("{ this is not valid json ]")
            # Repair retry: valid extraction JSON.
            return make_response(_extraction_json(members=members))
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    # The verdict seam was invoked exactly twice: initial + one repair.
    assert verdict_calls["count"] == 2
    # The repair succeeded -> a real verdict is present.
    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    assert result.verdict.consensus_label == "unanimous"
    assert result.verdict.consensus_method == CONSENSUS_METHOD
    assert result.consensus_score == 1.0
    assert result.manifest is not None
    assert result.manifest.verdict_absent_reason is None
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED
