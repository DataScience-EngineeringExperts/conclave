"""Tests for CAC-04: ModelHarnessManifest + per-member execution receipts.

The manifest records WHAT ran (providers considered/called/skipped, concrete
model ids, generation settings, latency, token usage) and HOW the verdict was
made (verdict_extraction provenance, verdict_type, consensus_method,
verdict-absent reason — all populated later by CAC-05). Its load-bearing
acceptance criterion is the secret-safety self-scan: no key/header/raw-body
material may ever appear in a serialized manifest.

Every test runs offline. Member fan-out is driven by patching
``conclave.transport.post_json`` (the real ``call_model`` path, mirroring
``test_providers.py``) so the real ``receipt_from_answer`` wiring is exercised
end to end; no real network calls happen and no real keys are required.
"""

from __future__ import annotations

import pytest

import conclave
from conclave import (
    Council,
    CouncilResult,
    ModelHarnessManifest,
    ProviderExecutionReceipt,
    ProviderSkip,
    VerdictExtraction,
)
from conclave.config import ConclaveConfig, clear_config_cache
from conclave.manifest import (
    SECRET_SAFETY_UNVERIFIED,
    SECRET_SAFETY_VERIFIED,
    scan_for_secret_material,
    verified_secret_safety,
)
from conclave.models import TokenUsage
from conclave.registry import provider_prefix


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Each test starts and ends with an empty config memo for isolation."""
    clear_config_cache()
    yield
    clear_config_cache()


def _config() -> ConclaveConfig:
    """A deterministic config independent of any on-disk ~/.conclave file."""
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": "anthropic/claude-sonnet-4-6",
            "perplexity": "perplexity/sonar-pro",
            "openai": "openai/gpt-4.1",
        },
        synthesizer="claude",
    )


def _clear_all_keys(monkeypatch) -> None:
    """Remove every provider env var so 'no key' paths are deterministic."""
    for var in (
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _patch_openai_style_transport(monkeypatch) -> None:
    """Patch the network boundary to return a canned OpenAI-style payload.

    Every provider in ``_config`` is OpenAI-compatible at the wire level for the
    members we call here (xai/openai/perplexity), so one canned chat-completions
    payload serves them all. This keeps the real ``call_model`` ->
    ``receipt_from_answer`` path live with no real network.
    """

    async def fake_post_json(url, headers, json_body, timeout):
        return 200, {
            "choices": [{"message": {"content": "canned answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    monkeypatch.setattr("conclave.transport.post_json", fake_post_json)


# --------------------------------------------------------------------------- #
# 1. Normal run assembly
# --------------------------------------------------------------------------- #


async def test_manifest_assembled_on_normal_run(monkeypatch):
    """A council with >=2 keyed members produces a populated manifest."""
    _clear_all_keys(monkeypatch)
    # Two members keyed (grok, openai); synthesizer claude unkeyed so we exercise
    # the raw member path without a synthesis call confusing the receipts.
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai"], config=_config())
    result = await council.ask("What is 2+2?", synthesize=False)

    manifest = result.manifest
    assert manifest is not None
    assert isinstance(manifest, ModelHarnessManifest)
    # request_id is a uuid4 hex (32 lowercase hex chars).
    assert len(manifest.request_id) == 32
    assert all(c in "0123456789abcdef" for c in manifest.request_id)
    assert manifest.conclave_version == conclave.__version__
    assert manifest.mode == "raw"
    assert manifest.providers_considered == ["grok", "openai"]
    assert manifest.providers_called == ["grok", "openai"]
    assert manifest.model_ids == ["xai/grok-4.3", "openai/gpt-4.1"]
    assert len(manifest.receipts) == len(result.answers) == 2
    assert manifest.total_latency_ms >= 0.0


async def test_manifest_mode_is_synthesize_when_synthesizing(monkeypatch):
    """The deliberation mode is recorded as ``synthesize`` for a synth run."""
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-test")
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai"], synthesizer="claude", config=_config())
    result = await council.ask("What is 2+2?", synthesize=True)

    assert result.manifest is not None
    assert result.manifest.mode == "synthesize"


# --------------------------------------------------------------------------- #
# 2. Providers skipped with reasons
# --------------------------------------------------------------------------- #


async def test_providers_skipped_with_reasons(monkeypatch):
    """Members without a key land in ``providers_skipped`` with the no-key reason."""
    _clear_all_keys(monkeypatch)
    # Request 3 members; key only one.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai", "perplexity"], config=_config())
    result = await council.ask("hi", synthesize=False)

    manifest = result.manifest
    assert manifest is not None
    assert manifest.providers_considered == ["grok", "openai", "perplexity"]
    assert manifest.providers_called == ["openai"]
    assert len(manifest.providers_skipped) == 2
    assert {s.name for s in manifest.providers_skipped} == {"grok", "perplexity"}
    for skip in manifest.providers_skipped:
        assert isinstance(skip, ProviderSkip)
        assert "no API key" in skip.reason


# --------------------------------------------------------------------------- #
# 3. Secret-safety matrix (the critical AC)
# --------------------------------------------------------------------------- #


def _fully_populated_manifest() -> ModelHarnessManifest:
    """Construct a manifest with every field set to representative values.

    Includes receipts carrying redacted error strings and a populated
    verdict_extraction so the scan has real content to inspect — yet nothing
    secret-shaped, proving the assembled object is clean by construction.
    """
    return ModelHarnessManifest(
        request_id="0123456789abcdef0123456789abcdef",
        conclave_version="1.0.0",
        mode="synthesize",
        providers_considered=["grok", "openai", "perplexity"],
        providers_called=["grok", "openai"],
        providers_skipped=[ProviderSkip(name="perplexity", reason="no API key in environment")],
        model_ids=["xai/grok-4.3", "openai/gpt-4.1"],
        generation_settings={"temperature": 0.7, "timeout": 120.0},
        receipts=[
            ProviderExecutionReceipt(
                name="grok",
                provider="xai",
                model_id="xai/grok-4.3",
                generation_settings={"temperature": 0.7, "timeout": 120.0},
                latency_ms=812.5,
                usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                error=None,
                schema_valid=None,
            ),
            ProviderExecutionReceipt(
                name="openai",
                provider="openai",
                model_id="openai/gpt-4.1",
                generation_settings={"temperature": 0.7, "timeout": 120.0},
                latency_ms=640.0,
                usage=None,
                # An already-redacted error string: legitimately carries the
                # [REDACTED] marker, which the scan must tolerate.
                error="openai: HTTP 401: invalid api key: [REDACTED]",
                schema_valid=None,
            ),
        ],
        total_latency_ms=1452.5,
        total_usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        redacted_errors=["openai: HTTP 401: invalid api key: [REDACTED]"],
        verdict_extraction=VerdictExtraction(model_id=None, prompt_version=None),
    )


def test_secret_safety_clean_manifest_passes_scan():
    """A representative fully-populated manifest is clean and scans True."""
    manifest = _fully_populated_manifest()
    dumped = manifest.model_dump_json()
    lowered = dumped.lower()
    for forbidden in ("sk-", "bearer", "authorization", "api_key", "x-api-key"):
        assert forbidden not in lowered, f"forbidden substring {forbidden!r} present"
    assert scan_for_secret_material(manifest) is True
    # The redacted marker is fine and must not trip the scan.
    assert "[redacted]" in lowered


def test_secret_safety_helper_sets_verified_status_when_clean():
    """``verified_secret_safety`` returns the VERIFIED literal for a clean manifest."""
    manifest = _fully_populated_manifest()
    assert verified_secret_safety(manifest) == SECRET_SAFETY_VERIFIED


def test_secret_safety_negative_scan_detects_pollution():
    """A manifest deliberately polluted with key material scans False.

    Proves the scanner actually detects, rather than always passing.
    """
    polluted = _fully_populated_manifest()
    # Inject a leak into a free-text field (a redacted_errors entry).
    polluted.redacted_errors.append("leaked credential sk-leaked123abc")
    assert scan_for_secret_material(polluted) is False
    assert verified_secret_safety(polluted) == SECRET_SAFETY_UNVERIFIED


async def test_council_sets_secret_safety_verified(monkeypatch):
    """A real council run stamps ``secret_safety`` VERIFIED after the scan."""
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai"], config=_config())
    result = await council.ask("hi", synthesize=False)

    assert result.manifest is not None
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


# --------------------------------------------------------------------------- #
# 4. Receipt-per-member
# --------------------------------------------------------------------------- #


async def test_receipt_per_member_fields(monkeypatch):
    """One receipt per answer; provider + generation_settings are correct."""
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai"], config=_config(), temperature=0.3, timeout=42.0)
    result = await council.ask("hi", synthesize=False)

    manifest = result.manifest
    assert manifest is not None
    assert len(manifest.receipts) == len(result.answers)
    for receipt, answer in zip(manifest.receipts, result.answers, strict=True):
        assert isinstance(receipt, ProviderExecutionReceipt)
        assert receipt.name == answer.name
        assert receipt.model_id == answer.model_id
        assert receipt.provider == provider_prefix(answer.model_id)
        assert receipt.generation_settings == {"temperature": 0.3, "timeout": 42.0}
        assert receipt.latency_ms == answer.latency_ms
        assert receipt.schema_valid is None
    # Council-level generation settings mirror the receipts.
    assert manifest.generation_settings == {"temperature": 0.3, "timeout": 42.0}


async def test_total_usage_summed_across_receipts(monkeypatch):
    """``total_usage`` sums per-member usage when at least one member reports it."""
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai"], config=_config())
    result = await council.ask("hi", synthesize=False)

    manifest = result.manifest
    assert manifest is not None
    # Each canned response reports total_tokens=5; two members -> 10.
    assert manifest.total_usage is not None
    assert manifest.total_usage.total_tokens == 10
    assert manifest.total_usage.prompt_tokens == 6
    assert manifest.total_usage.completion_tokens == 4


# --------------------------------------------------------------------------- #
# 5. Verdict-provenance + cost fields default None (CAC-05 / §8 fill later)
# --------------------------------------------------------------------------- #


async def test_verdict_provenance_and_cost_default_none(monkeypatch):
    """The CAC-05/§8 slots are left unfilled by CAC-04."""
    _clear_all_keys(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai"], config=_config())
    result = await council.ask("hi", synthesize=False)

    manifest = result.manifest
    assert manifest is not None
    assert isinstance(manifest.verdict_extraction, VerdictExtraction)
    assert manifest.verdict_extraction.model_id is None
    assert manifest.verdict_extraction.prompt_version is None
    assert manifest.verdict_type is None
    assert manifest.consensus_method is None
    assert manifest.verdict_absent_reason is None
    assert manifest.estimated_cost is None
    assert manifest.pricing_snapshot_date is None
    assert manifest.schema_valid is None


# --------------------------------------------------------------------------- #
# 6. Backward-compat CouncilResult construction
# --------------------------------------------------------------------------- #


def test_council_result_constructs_without_manifest():
    """An existing-shape CouncilResult still validates; manifest defaults None."""
    result = CouncilResult(prompt="x")
    assert result.manifest is None
    # Re-validating a dumped result round-trips (additive field is optional).
    again = CouncilResult.model_validate(result.model_dump())
    assert again.manifest is None


# --------------------------------------------------------------------------- #
# 7. Empty-members path still carries a manifest
# --------------------------------------------------------------------------- #


async def test_empty_members_path_carries_manifest(monkeypatch):
    """A council where NO member has a key still produces a manifest."""
    _clear_all_keys(monkeypatch)
    _patch_openai_style_transport(monkeypatch)

    council = Council(models=["grok", "openai", "perplexity"], config=_config())
    result = await council.ask("hi", synthesize=False)

    manifest = result.manifest
    assert manifest is not None
    assert manifest.receipts == []
    assert manifest.providers_called == []
    assert {s.name for s in manifest.providers_skipped} == {"grok", "openai", "perplexity"}
    assert manifest.providers_considered == ["grok", "openai", "perplexity"]
    assert manifest.total_usage is None
    assert manifest.total_latency_ms == 0.0
    assert manifest.secret_safety == SECRET_SAFETY_VERIFIED
