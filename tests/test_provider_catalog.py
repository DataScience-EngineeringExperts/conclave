"""Tests for the lean static provider capability catalog (CAC-03).

The catalog is PURE DATA: a checked-in, conservative snapshot of which
generation knobs each built-in model accepts. These tests pin three things:

1. **Coverage** — every model id in :data:`conclave.registry.DEFAULT_MODELS`
   has a :class:`ProviderCapabilities` entry, and lookups resolve correctly
   (exact match, known-provider fallback, unknown -> ``None``).
2. **Honesty** — the structured-output / json-mode flags match the
   per-provider truth documented in CAC-03 (verified vs. conservative-False).
3. **Secret-safety / offline** — no field anywhere in the catalog can hold a
   string (so no key value can hide in a capability), and the ``static``
   module reads neither env, disk, nor network.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

import conclave.provider_catalog.static as static_mod
from conclave.provider_catalog import (
    PROVIDER_FALLBACK,
    STATIC_CATALOG,
    ProviderCapabilities,
    capabilities_for,
)
from conclave.registry import DEFAULT_MODELS

# The 9 built-in model ids, derived from the single source of truth so this
# test can never drift from the registry.
MODEL_IDS = sorted(DEFAULT_MODELS.values())

# Models that document native structured output (json_schema / responseSchema
# / tool-use schema). The rest are conservative ``False``.
STRUCTURED_OUTPUT_TRUE = {
    "openai/gpt-4.1",
    "gemini/gemini-2.5-pro",
    "anthropic/claude-sonnet-4-6",
    "mistral/mistral-large-latest",
    "groq/llama-3.3-70b-versatile",
    "xai/grok-4.3",
}


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_every_default_model_has_catalog_entry(model_id):
    """Every model in DEFAULT_MODELS maps to a ProviderCapabilities entry."""
    assert model_id in STATIC_CATALOG
    assert isinstance(STATIC_CATALOG[model_id], ProviderCapabilities)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_capabilities_for_returns_exact_catalog_object(model_id):
    """A known model id resolves to the very STATIC_CATALOG object (identity)."""
    assert capabilities_for(model_id) is STATIC_CATALOG[model_id]


def test_capabilities_for_known_provider_unknown_model_uses_fallback():
    """A known provider with an unlisted model resolves to its provider fallback."""
    caps = capabilities_for("openai/gpt-4o-mini")
    assert caps is PROVIDER_FALLBACK["openai"]


def test_capabilities_for_unknown_provider_returns_none():
    """A fully unknown provider (with or without a slash) resolves to None."""
    assert capabilities_for("mystery/model") is None
    assert capabilities_for("nope") is None


def test_structured_output_true_for_schema_native_models():
    """The schema-native trio expose structured output."""
    assert STATIC_CATALOG["openai/gpt-4.1"].supports_structured_output is True
    assert STATIC_CATALOG["gemini/gemini-2.5-pro"].supports_structured_output is True
    assert STATIC_CATALOG["anthropic/claude-sonnet-4-6"].supports_structured_output is True


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_structured_output_flag_matches_honesty_table(model_id):
    """Each model's structured-output flag matches the CAC-03 honesty table."""
    expected = model_id in STRUCTURED_OUTPUT_TRUE
    assert STATIC_CATALOG[model_id].supports_structured_output is expected


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_streaming_and_system_prompt_true_for_all(model_id):
    """All 9 built-ins support streaming and a system prompt."""
    caps = STATIC_CATALOG[model_id]
    assert caps.supports_streaming is True
    assert caps.supports_system_prompt is True


def test_provider_capabilities_is_frozen():
    """ProviderCapabilities is immutable: attribute assignment raises."""
    caps = ProviderCapabilities()
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.supports_temperature = False  # type: ignore[misc]


def test_no_field_anywhere_holds_a_string():
    """Secret-safety: every field across the whole catalog is bool | int | None.

    A string field would be the one place a key value or other secret could
    hide inside a capability object; asserting the type closes that door.
    """
    all_caps = list(STATIC_CATALOG.values()) + list(PROVIDER_FALLBACK.values())
    for caps in all_caps:
        for field in dataclasses.fields(caps):
            value = getattr(caps, field.name)
            assert isinstance(value, (bool, int)) or value is None, (
                f"{field.name}={value!r} is not bool/int/None"
            )


def test_static_module_does_no_io():
    """Offline proof: the static module never reads env, disk, or network."""
    src = inspect.getsource(static_mod)
    assert "import os" not in src
    assert "httpx" not in src
    assert "requests" not in src
    assert "open(" not in src
    assert "environ" not in src
