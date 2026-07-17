"""Regression guard: the ModelHarnessManifest rides on EVERY mode's result.

The auditable manifest (CAC-04) is documented as first-class on every
:class:`conclave.models.CouncilResult` -- "it rides on every result, not behind
a debug flag." Historically only the synthesize/raw path (built in
``Council._ask_uncached``) satisfied this; ``debate``/``adversarial``/``vote``
constructed their result directly in :mod:`conclave.modes` and returned with
``manifest is None``, so the invariant silently drifted for 3 of 5 modes and no
test guarded it.

The single-site fix attaches the manifest in ``Council._cached_run`` (the one
chokepoint every mode funnels through) via ``Council._ensure_manifest``. These
tests pin the invariant for debate, adversarial, and vote -- including the
zero-members early-return and cache-hit paths -- so it cannot drift again. All
run offline through the shared ``patch_call_model`` seam; no network, no keys.
"""

from __future__ import annotations

import pytest

from conclave import Council
from conclave.config import ConclaveConfig
from conclave.manifest import (
    SECRET_SAFETY_VERIFIED,
    ModelHarnessManifest,
    ProviderExecutionReceipt,
)
from conclave.models import ELITE_PROTOCOL_VERSION, ModelAnswer, TokenUsage
from conclave.prompts import ELITE_PROMPT_VERSION, SYNTHESIS_PROMPT_VERSION
from conclave.providers import receipt_from_answer
from conclave.verdict import VERDICT_EXTRACTION_PROMPT_VERSION, VERDICT_SCHEMA_VERSION
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


def _assert_verified_manifest(result, expected_mode: str) -> None:
    """The result carries a secret-safety-VERIFIED manifest stamped with the mode."""
    manifest = result.manifest
    assert manifest is not None, f"{expected_mode} result must carry a manifest"
    assert isinstance(manifest, ModelHarnessManifest)
    assert manifest.secret_safety == SECRET_SAFETY_VERIFIED
    assert manifest.mode == expected_mode
    # request_id is a uuid4 hex (32 lowercase hex chars) -- a real, assembled manifest.
    assert len(manifest.request_id) == 32
    assert all(c in "0123456789abcdef" for c in manifest.request_id)


def _elite_phase(messages) -> str:
    """Identify the elite phase from its system prompt."""
    system = _system_text(messages)
    if "claim auditor" in system:
        return "critique"
    if "revise your original answer" in system:
        return "revision"
    return "initial"


def test_receipt_phase_is_backward_compatible_and_optional() -> None:
    """Existing callers get phase=None; elite callers can stamp provenance."""
    answer = ModelAnswer(name="grok", model_id="xai/grok-4.3", answer="A")

    normal = receipt_from_answer(answer, temperature=0.7, timeout=120.0)
    elite = receipt_from_answer(answer, temperature=0.7, timeout=120.0, phase="critique")

    assert normal.phase is None
    assert elite.phase == "critique"


def test_receipt_audit_fields_are_additive_and_conservative() -> None:
    """Old construction remains valid while new accounting fields default safely."""
    receipt = ProviderExecutionReceipt(name="grok", provider="xai", model_id="xai/grok-4.3")

    assert receipt.attempt == 1
    assert receipt.outcome == "success"
    assert receipt.error_category is None
    assert receipt.protocol_version is None
    assert receipt.prompt_version is None
    assert receipt.schema_version is None
    assert receipt.estimated_cost is None


def test_receipt_empty_error_string_is_still_a_failed_call() -> None:
    """ModelAnswer.error presence, not truthiness, determines call failure."""
    answer = ModelAnswer(name="grok", model_id="xai/grok-4.3", error="")

    receipt = receipt_from_answer(answer, temperature=0.7, timeout=120.0)

    assert receipt.outcome == "failed"
    assert receipt.error_category == "provider_error"
    assert receipt.error == "provider_error"


def _valid_elite_verdict_json() -> str:
    return """{
      "verdict_applies": true,
      "verdict_type": "decision",
      "headline": "Choose A",
      "recommendation": "Choose A",
      "positions": [{
        "label": "A", "summary": "A wins", "providers": ["grok", "gemini", "perplexity"],
        "evidence_answer_ids": []
      }],
      "provider_votes": [
        {"provider": "grok", "position_label": "A"},
        {"provider": "gemini", "position_label": "A"},
        {"provider": "perplexity", "position_label": "A"}
      ],
      "minority_reports": [], "conflicts": [], "caveats": [], "dissent_summary": null
    }"""


async def test_elite_manifest_captures_synthesis_and_verdict_repair_calls(monkeypatch) -> None:
    """Every actual Elite model call has one versioned, correctly-accounted receipt."""
    _all_keys(monkeypatch)
    verdict_attempt = 0
    calls: list[tuple[float, float]] = []

    async def fake_call_model(
        name,
        model,
        messages,
        *,
        temperature=0.7,
        timeout=120.0,
        config=None,
        **kwargs,
    ):
        nonlocal verdict_attempt
        calls.append((temperature, timeout))
        system = _system_text(messages)
        if system.startswith("You are the verdict extractor"):
            verdict_attempt += 1
            text = "not json" if verdict_attempt == 1 else _valid_elite_verdict_json()
        elif system.startswith("You are the synthesizer of a council"):
            text = "elite synthesis"
        else:
            text = f"{_elite_phase(messages)} from {model}"
        return ModelAnswer(
            name=name,
            model_id=model,
            answer=text,
            latency_s=len(calls) / 1000,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
        )

    monkeypatch.setattr("conclave.council.call_model", fake_call_model)
    monkeypatch.setattr("conclave.verdict_synthesis.call_model", fake_call_model)
    result = await Council(
        models=["grok", "gemini", "perplexity"],
        config=_config(),
        temperature=0.25,
        timeout=37.0,
    ).elite("Choose the strongest option.")

    manifest = result.manifest
    assert manifest is not None
    assert [receipt.phase for receipt in manifest.receipts] == (
        ["initial"] * 3
        + ["critique"] * 3
        + ["revision"] * 3
        + ["synthesis", "verdict_extraction", "verdict_repair"]
    )
    first_verdict, repair = manifest.receipts[-2:]
    assert (first_verdict.attempt, first_verdict.outcome, first_verdict.schema_valid) == (
        1,
        "schema_invalid",
        False,
    )
    assert first_verdict.error_category == "schema_validation"
    assert (repair.attempt, repair.outcome, repair.schema_valid) == (2, "success", True)
    assert repair.error_category is None
    assert all(
        receipt.generation_settings == {"temperature": 0.25, "timeout": 37.0}
        for receipt in manifest.receipts
    )
    assert calls == [(0.25, 37.0)] * 12
    assert all(receipt.protocol_version == ELITE_PROTOCOL_VERSION for receipt in manifest.receipts)
    assert all(receipt.prompt_version is None for receipt in manifest.receipts[:3])
    assert all(receipt.prompt_version == ELITE_PROMPT_VERSION for receipt in manifest.receipts[3:9])
    assert manifest.receipts[9].prompt_version == SYNTHESIS_PROMPT_VERSION
    assert all(
        receipt.prompt_version == VERDICT_EXTRACTION_PROMPT_VERSION
        for receipt in manifest.receipts[-2:]
    )
    assert all(
        receipt.schema_version == VERDICT_SCHEMA_VERSION for receipt in manifest.receipts[-2:]
    )
    assert manifest.total_usage is not None
    assert manifest.total_usage.total_tokens == 12 * 12
    assert manifest.total_latency_ms == sum(r.latency_ms for r in manifest.receipts)
    assert manifest.total_latency_ms == 78.0
    assert manifest.estimated_cost is None


async def test_elite_manifest_keeps_failed_synthesis_call_without_raw_error(
    monkeypatch, patch_call_model
) -> None:
    """A failed synthesis is counted, categorized, and stripped of raw provider detail."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if _system_text(messages).startswith("You are the synthesizer of a council"):
            raise RuntimeError("POST https://secret.example/v1 body={'token':'do-not-store'}")
        return make_response(f"{_elite_phase(messages)} from {model}")

    patch_call_model(handler)
    result = await Council(
        models=["grok", "gemini", "perplexity"],
        config=_config(),
    ).elite("Choose.")

    assert result.manifest is not None
    receipt = result.manifest.receipts[-1]
    assert receipt.phase == "synthesis"
    assert receipt.outcome == "failed"
    assert receipt.error_category == "provider_error"
    assert receipt.error == "provider_error"
    assert "secret.example" not in result.manifest.model_dump_json()
    assert "do-not-store" not in result.manifest.model_dump_json()


async def test_elite_manifest_keeps_failed_verdict_attempt_and_successful_repair(
    monkeypatch, patch_call_model
) -> None:
    """A provider-error extraction and its repair remain distinct receipts."""
    _all_keys(monkeypatch)
    verdict_attempt = 0

    def handler(model, messages, **kwargs):
        nonlocal verdict_attempt
        system = _system_text(messages)
        if system.startswith("You are the verdict extractor"):
            verdict_attempt += 1
            if verdict_attempt == 1:
                raise RuntimeError("upstream provider unavailable")
            return make_response(_valid_elite_verdict_json())
        if system.startswith("You are the synthesizer of a council"):
            return make_response("elite synthesis")
        return make_response(f"{_elite_phase(messages)} from {model}")

    patch_call_model(handler)
    result = await Council(
        models=["grok", "gemini", "perplexity"],
        config=_config(),
    ).elite("Choose.")

    assert result.manifest is not None
    failed, repaired = result.manifest.receipts[-2:]
    assert (failed.phase, failed.attempt, failed.outcome, failed.error_category) == (
        "verdict_extraction",
        1,
        "failed",
        "provider_error",
    )
    assert failed.schema_valid is None
    assert (repaired.phase, repaired.attempt, repaired.outcome, repaired.schema_valid) == (
        "verdict_repair",
        2,
        "success",
        True,
    )


async def test_elite_manifest_audits_every_phase(monkeypatch, patch_call_model):
    """A complete elite run records all nine member calls and their usage."""
    _all_keys(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    def handler(model, messages, **kwargs):
        return make_response(f"{_elite_phase(messages)} from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "claude", "perplexity"],
        synthesizer="claude",
        config=_config(),
        extract_verdict=False,
    )

    result = await council.elite("Choose the strongest option.")

    _assert_verified_manifest(result, "elite")
    assert result.manifest.providers_called == ["grok", "gemini", "perplexity"]
    assert len(result.manifest.providers_called) == len(set(result.manifest.providers_called))
    assert [(skip.name, skip.reason) for skip in result.manifest.providers_skipped] == [
        ("claude", "no API key in environment")
    ]
    assert [receipt.phase for receipt in result.manifest.receipts] == (
        ["initial"] * 3 + ["critique"] * 3 + ["revision"] * 3
    )
    assert result.manifest.total_usage is not None
    assert result.manifest.total_usage.prompt_tokens == 45
    assert result.manifest.total_usage.completion_tokens == 63
    assert result.manifest.total_usage.total_tokens == 108


async def test_incomplete_elite_manifest_keeps_attempted_phase_receipts(
    monkeypatch, patch_call_model
):
    """A failed critique gate retains initial and attempted critique receipts."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        phase = _elite_phase(messages)
        if phase == "critique" and model == "gemini/gemini-2.5-pro":
            raise RuntimeError("critic unavailable")
        return make_response(f"{phase} from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_config(),
        extract_verdict=False,
    )

    result = await council.elite("Choose.")

    assert result.elite is not None and result.elite.completed is False
    _assert_verified_manifest(result, "elite")
    assert [receipt.phase for receipt in result.manifest.receipts] == (
        ["initial"] * 3 + ["critique"] * 3
    )
    assert result.manifest.total_usage is not None
    assert result.manifest.total_usage.total_tokens == 60


async def test_elite_cache_hit_preserves_phase_receipts(monkeypatch, patch_call_model, tmp_path):
    """Elite phase provenance survives cache serialization and reload."""
    _all_keys(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def handler(model, messages, **kwargs):
        return make_response(f"{_elite_phase(messages)} from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_config(),
        extract_verdict=False,
        cache=True,
    )

    first = await council.elite("Choose.")
    second = await council.elite("Choose.")

    assert first.cached is False
    assert second.cached is True
    _assert_verified_manifest(second, "elite")
    assert [receipt.phase for receipt in second.manifest.receipts] == (
        ["initial"] * 3 + ["critique"] * 3 + ["revision"] * 3 + ["synthesis"]
    )


# --------------------------------------------------------------------------- #
# Debate
# --------------------------------------------------------------------------- #


async def test_debate_result_carries_verified_manifest(monkeypatch, patch_call_model):
    """A debate run attaches a secret-safe manifest (was manifest=None before the fix)."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        if "synthesizer concluding a multi-round" in _system_text(messages):
            return make_response("DEBATE SYNTHESIS")
        return make_response(f"answer from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"], synthesizer="claude", config=_config()
    )
    result = await council.debate("Is P=NP?", rounds=2)

    _assert_verified_manifest(result, "debate")
    # Full resolved membership is recorded even though only survivors answer.
    assert result.manifest.providers_considered == ["grok", "gemini", "perplexity"]
    assert result.manifest.providers_called == ["grok", "gemini", "perplexity"]


async def test_debate_dropped_member_still_in_manifest_membership(monkeypatch, patch_call_model):
    """A member that drops mid-debate stays in providers_called (full membership)."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "synthesizer concluding a multi-round" in system:
            return make_response("SYNTH")
        if model == "gemini/gemini-2.5-pro":  # gemini fails every round -> drops out
            raise RuntimeError("gemini down")
        return make_response(f"answer from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"], synthesizer="claude", config=_config()
    )
    result = await council.debate("q", rounds=2)

    _assert_verified_manifest(result, "debate")
    # Membership reflects everyone that was called, not just final-round survivors.
    assert set(result.manifest.providers_called) == {"grok", "gemini", "perplexity"}


async def test_debate_no_members_still_carries_manifest(monkeypatch, patch_call_model, clear_keys):
    """The zero-members early return in run_debate still yields a manifest."""

    def handler(model, messages, **kwargs):  # pragma: no cover - never called
        return make_response("unused")

    patch_call_model(handler)
    council = Council(models=["grok", "claude"], config=_config())
    result = await council.debate("q", rounds=2)

    _assert_verified_manifest(result, "debate")
    assert result.manifest.receipts == []
    assert result.manifest.providers_called == []
    assert {s.name for s in result.manifest.providers_skipped} == {"grok", "claude"}


# --------------------------------------------------------------------------- #
# Adversarial
# --------------------------------------------------------------------------- #


async def test_adversarial_result_carries_verified_manifest(monkeypatch, patch_call_model):
    """An adversarial run attaches a secret-safe manifest (was manifest=None before)."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            return make_response("VERDICT TEXT")
        if "critic on an adversarial review" in system:
            return make_response(f"critique from {model}")
        return make_response(f"proposal from {model}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"], synthesizer="claude", config=_config()
    )
    result = await council.adversarial("Defend microservices.")

    _assert_verified_manifest(result, "adversarial")
    assert result.manifest.providers_considered == ["grok", "gemini", "perplexity"]


async def test_adversarial_no_members_still_carries_manifest(
    monkeypatch, patch_call_model, clear_keys
):
    """The zero-members early return in run_adversarial still yields a manifest."""

    def handler(model, messages, **kwargs):  # pragma: no cover - never called
        return make_response("unused")

    patch_call_model(handler)
    council = Council(models=["grok", "claude"], config=_config())
    result = await council.adversarial("q")

    _assert_verified_manifest(result, "adversarial")
    assert result.manifest.providers_called == []
    assert {s.name for s in result.manifest.providers_skipped} == {"grok", "claude"}


# --------------------------------------------------------------------------- #
# Vote
# --------------------------------------------------------------------------- #


async def test_vote_result_carries_verified_manifest(monkeypatch, patch_call_model):
    """A vote run attaches a secret-safe manifest (was manifest=None before the fix)."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        return make_response("A")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"], synthesizer="claude", config=_config()
    )
    result = await council.vote("Best option?", ["Alpha", "Beta"])

    _assert_verified_manifest(result, "vote")
    assert result.manifest.providers_called == ["grok", "gemini", "perplexity"]
    # One receipt per member that voted.
    assert len(result.manifest.receipts) == len(result.answers)


async def test_vote_no_members_still_carries_manifest(monkeypatch, patch_call_model, clear_keys):
    """The zero-members early return in run_vote still yields a manifest."""

    def handler(model, messages, **kwargs):  # pragma: no cover - never called
        return make_response("A")

    patch_call_model(handler)
    council = Council(models=["grok", "claude"], config=_config())
    result = await council.vote("Pick", ["Yes", "No"])

    _assert_verified_manifest(result, "vote")
    assert result.manifest.receipts == []
    assert {s.name for s in result.manifest.providers_skipped} == {"grok", "claude"}


# --------------------------------------------------------------------------- #
# Cache-hit path also carries the manifest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["debate", "adversarial", "vote"])
async def test_cache_hit_carries_manifest(monkeypatch, patch_call_model, tmp_path, mode):
    """A cached result for each mode still carries a manifest on the hit path.

    Two identical runs with caching enabled: the second is served from cache
    (``cached is True``) and must still expose a VERIFIED manifest, proving the
    invariant holds on the cache-hit branch of ``_cached_run`` too.
    """
    _all_keys(monkeypatch)
    # Point the on-disk cache (XDG_CACHE_HOME/conclave) at an isolated tmp dir.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def handler(model, messages, **kwargs):
        system = _system_text(messages)
        if "judge of an adversarial review" in system:
            return make_response("VERDICT")
        if "critic on an adversarial review" in system:
            return make_response("critique")
        if "synthesizer concluding a multi-round" in system:
            return make_response("SYNTH")
        return make_response("A")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
        cache=True,
    )

    async def run():
        if mode == "debate":
            return await council.debate("q", rounds=2)
        if mode == "adversarial":
            return await council.adversarial("q")
        return await council.vote("q", ["Alpha", "Beta"])

    first = await run()
    assert first.cached is False
    _assert_verified_manifest(first, mode)

    second = await run()
    assert second.cached is True
    _assert_verified_manifest(second, mode)
