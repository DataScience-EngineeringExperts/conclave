"""Tests for the provider registry and config merge logic."""

from __future__ import annotations

from pathlib import Path

from conclave.config import load_config
from conclave.registry import (
    DEFAULT_MODELS,
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
