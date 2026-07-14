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
from conclave.manifest import SECRET_SAFETY_VERIFIED, ModelHarnessManifest
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
