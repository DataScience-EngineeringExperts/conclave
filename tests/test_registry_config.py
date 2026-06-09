"""Tests for the provider registry and config merge logic."""

from __future__ import annotations

import pytest

from conclave.config import load_config
from conclave.registry import (
    DEFAULT_MODELS,
    NATIVE_PROVIDERS,
    OPENAI_COMPAT_PROVIDERS,
    PROVIDER_ENV_VARS,
    RegistryError,
    _assert_metadata_consistent,
    key_present,
    key_source,
    provider_prefix,
    required_env_vars,
)


def test_provider_prefix():
    assert provider_prefix("xai/grok-4.3") == "xai"
    assert provider_prefix("gemini/gemini-2.5-pro") == "gemini"
    assert provider_prefix("bare-model") == "bare-model"


def test_required_env_vars_known_and_unknown():
    assert required_env_vars("anthropic/claude-sonnet-4-6") == ["ANTHROPIC_API_KEY"]
    assert required_env_vars("gemini/gemini-2.5-pro") == ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    # Unknown provider -> no statically-known var.
    assert required_env_vars("mystery/model") == []


def test_key_present_and_source(monkeypatch):
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert key_present("xai/grok-4.3") is False
    assert key_source("xai/grok-4.3") is None

    monkeypatch.setenv("XAI_API_KEY", "abc")
    assert key_present("xai/grok-4.3") is True
    assert key_source("xai/grok-4.3") == "XAI_API_KEY"

    # Gemini falls back to GOOGLE_API_KEY.
    monkeypatch.setenv("GOOGLE_API_KEY", "xyz")
    assert key_present("gemini/gemini-2.5-pro") is True
    assert key_source("gemini/gemini-2.5-pro") == "GOOGLE_API_KEY"


def test_key_present_blank_is_absent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert key_present("openai/gpt-4.1") is False


def test_unknown_provider_assumed_present(monkeypatch):
    # No env var known -> don't pre-skip; let the live call decide.
    assert key_present("mystery/model") is True


def test_load_config_defaults_when_absent(tmp_path):
    cfg = load_config(path=tmp_path / "does-not-exist.yml")
    # Built-in defaults always present.
    for name, model_id in DEFAULT_MODELS.items():
        assert cfg.models[name] == model_id
    assert "default" in cfg.councils
    assert cfg.synthesizer == "claude"


def test_load_config_merges_file(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        "models:\n"
        "  grok: xai/grok-4.3-fast\n"
        "  myllm: openai/gpt-4o\n"
        "councils:\n"
        "  fast: [grok, perplexity]\n"
        "synthesizer: gemini\n",
        encoding="utf-8",
    )
    cfg = load_config(path=cfg_path)

    # Overridden default.
    assert cfg.models["grok"] == "xai/grok-4.3-fast"
    # New custom model.
    assert cfg.models["myllm"] == "openai/gpt-4o"
    # Untouched defaults survive.
    assert cfg.models["claude"] == DEFAULT_MODELS["claude"]
    # Custom council + synthesizer.
    assert cfg.resolve_council("fast") == ["grok", "perplexity"]
    assert cfg.synthesizer == "gemini"


def test_resolve_council_csv_and_named(tmp_path):
    cfg = load_config(path=tmp_path / "missing.yml")
    assert cfg.resolve_council("grok,claude") == ["grok", "claude"]
    assert cfg.resolve_council("grok, claude , perplexity") == [
        "grok",
        "claude",
        "perplexity",
    ]


def test_resolve_model_id_passthrough(tmp_path):
    cfg = load_config(path=tmp_path / "missing.yml")
    assert cfg.resolve_model_id("grok") == "xai/grok-4.3"
    # Unknown friendly name passes through as a raw id.
    assert cfg.resolve_model_id("openai/gpt-4o") == "openai/gpt-4o"


def test_malformed_config_ignored(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("not: [a, valid: mapping\n", encoding="utf-8")
    cfg = load_config(path=cfg_path)
    # Falls back to defaults rather than raising.
    assert cfg.models["grok"] == DEFAULT_MODELS["grok"]


# --------------------------------------------------------------------------- #
# Single-source-of-truth for provider metadata (issue #19)
# --------------------------------------------------------------------------- #


def test_provider_env_vars_derived_from_single_source():
    """PROVIDER_ENV_VARS is built from the OpenAI-compat + native source tables.

    There is no hand-maintained second copy that could drift: every env-var entry
    must trace back to exactly one source table.
    """
    derived = {
        **{prefix: list(meta.env_vars) for prefix, meta in OPENAI_COMPAT_PROVIDERS.items()},
        **{prefix: list(env_vars) for prefix, env_vars in NATIVE_PROVIDERS.items()},
    }
    assert PROVIDER_ENV_VARS == derived


def test_every_openai_compat_provider_has_env_vars():
    """An OpenAI-compatible provider can never be declared with a URL but no key."""
    for prefix, meta in OPENAI_COMPAT_PROVIDERS.items():
        assert meta.env_vars, f"{prefix} declares a URL but no env var"
        assert PROVIDER_ENV_VARS[prefix] == list(meta.env_vars)


def test_registry_and_adapter_url_tables_are_in_sync():
    """registry.OPENAI_COMPAT_PROVIDERS and adapters.OPENAI_COMPAT_URLS agree.

    This is the invariant whose violation used to escape as a runtime KeyError in
    _openai_compat_adapter (issue #19). The live tables must match exactly.
    """
    from conclave.adapters.openai_compat import OPENAI_COMPAT_URLS

    assert set(OPENAI_COMPAT_PROVIDERS) == set(OPENAI_COMPAT_URLS)
    for prefix, meta in OPENAI_COMPAT_PROVIDERS.items():
        assert meta.completions_url == OPENAI_COMPAT_URLS[prefix]


def test_consistency_check_passes_for_live_metadata():
    """The import-time guard is a no-op against the real, in-sync tables."""
    # Must not raise.
    _assert_metadata_consistent()


def test_consistency_check_detects_url_drift(monkeypatch):
    """A URL present in the adapter table but absent from registry is caught loudly.

    Simulates the exact drift scenario: someone adds a prefix to
    adapters.OPENAI_COMPAT_URLS without adding it to the registry source of truth.
    The former failure mode was a silent per-call KeyError; now it is a clear,
    immediate RegistryError.
    """
    import conclave.adapters.openai_compat as oc

    drifted = dict(oc.OPENAI_COMPAT_URLS)
    drifted["ghostprovider"] = "https://api.ghost.example/v1/chat/completions"
    monkeypatch.setattr(oc, "OPENAI_COMPAT_URLS", drifted)

    with pytest.raises(RegistryError, match="ghostprovider"):
        _assert_metadata_consistent()


def test_consistency_check_detects_missing_url(monkeypatch):
    """A registry provider with no matching adapter URL is caught loudly."""
    import conclave.adapters.openai_compat as oc

    drifted = dict(oc.OPENAI_COMPAT_URLS)
    drifted.pop("perplexity")
    monkeypatch.setattr(oc, "OPENAI_COMPAT_URLS", drifted)

    with pytest.raises(RegistryError, match="perplexity"):
        _assert_metadata_consistent()


def test_consistency_check_detects_url_mismatch(monkeypatch):
    """Same prefix, divergent URL between the two tables is caught loudly."""
    import conclave.adapters.openai_compat as oc

    drifted = dict(oc.OPENAI_COMPAT_URLS)
    drifted["openai"] = "https://api.openai.com/v2/chat/completions"
    monkeypatch.setattr(oc, "OPENAI_COMPAT_URLS", drifted)

    with pytest.raises(RegistryError, match="URL drift"):
        _assert_metadata_consistent()
