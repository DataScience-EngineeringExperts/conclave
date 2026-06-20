"""STATIC, checked-in conservative capability defaults (dated 2026-06-20).

These tables are a hand-curated, *static* snapshot of provider capabilities.
There is deliberately NO cache, NO network call, NO disk read, and NO
env-var access in this module: it is pure data plus one lookup helper. A
later ticket (COC-05) will enrich these defaults via an explicit, opt-in
refresh/discovery step; until then this checked-in snapshot is the only source.

The catalog references provider and model IDENTIFIERS only (e.g. ``"openai"``,
``"openai/gpt-4.1"``). It never references, stores, or returns a key VALUE.

Honesty stance: a flag is ``True`` only where the provider documents the
capability for the model. Where support is per-model, version-dependent, or
otherwise unverified, the flag is a conservative ``False`` (annotated inline).
Volatile numbers (context window, output token limit) are left ``None`` rather
than guessed.
"""

from __future__ import annotations

from conclave.provider_catalog import ProviderCapabilities

# Per-model capabilities for the 9 built-in models. Keys are full model ids
# (provider-prefixed), matching conclave.registry.DEFAULT_MODELS.values().
STATIC_CATALOG: dict[str, ProviderCapabilities] = {
    # OpenAI gpt-4.1: native json_schema response_format. Not a reasoning model.
    "openai/gpt-4.1": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=True,
    ),
    # xAI grok-4.3: xAI documents structured outputs via json_schema.
    "xai/grok-4.3": ProviderCapabilities(
        supports_structured_output=True,  # verified: xAI structured outputs response_format
        supports_json_mode=True,
    ),
    # Gemini 2.5 Pro: native responseSchema; not the OpenAI json_object flag.
    "gemini/gemini-2.5-pro": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=False,
    ),
    # Claude Sonnet 4.6: structured output via tool-use schema; no json_object flag.
    "anthropic/claude-sonnet-4-6": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=False,
    ),
    # Perplexity sonar-pro: search-oriented, no schema/json_object guarantee.
    "perplexity/sonar-pro": ProviderCapabilities(
        supports_structured_output=False,  # conservative: perplexity sonar is search-oriented
        supports_json_mode=False,
    ),
    # Groq llama-3.3-70b: Groq exposes json_schema response_format.
    "groq/llama-3.3-70b-versatile": ProviderCapabilities(
        supports_structured_output=True,  # verified: Groq structured outputs (json_schema response_format)
        supports_json_mode=True,
    ),
    # DeepSeek chat: json_object mode yes; strict json_schema support uncertain.
    "deepseek/deepseek-chat": ProviderCapabilities(
        supports_structured_output=False,  # unverified: deepseek json_schema strict support uncertain — conservative False
        supports_json_mode=True,
    ),
    # Mistral large: documents json_schema response_format.
    "mistral/mistral-large-latest": ProviderCapabilities(
        supports_structured_output=True,  # verified: Mistral docs response_format json_schema
        supports_json_mode=True,
    ),
    # Together Llama-3.3-70B-Instruct-Turbo: json_object yes; json_schema per-model.
    "together/meta-llama/Llama-3.3-70B-Instruct-Turbo": ProviderCapabilities(
        supports_structured_output=False,  # unverified: Together json_schema support is per-model — conservative False
        supports_json_mode=True,
    ),
}

# Per-provider fallback keyed by provider PREFIX, so a known provider's
# non-default model still resolves to sane caps. Same honesty stance as the
# per-model table above. All limits left None; reasoning default False;
# streaming and system_prompt True everywhere.
PROVIDER_FALLBACK: dict[str, ProviderCapabilities] = {
    "openai": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=True,
    ),
    "xai": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=True,
    ),
    "gemini": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=False,
    ),
    "anthropic": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=False,
    ),
    "perplexity": ProviderCapabilities(
        supports_structured_output=False,
        supports_json_mode=False,
    ),
    "groq": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=True,
    ),
    "deepseek": ProviderCapabilities(
        supports_structured_output=False,
        supports_json_mode=True,
    ),
    "mistral": ProviderCapabilities(
        supports_structured_output=True,
        supports_json_mode=True,
    ),
    "together": ProviderCapabilities(
        supports_structured_output=False,
        supports_json_mode=True,
    ),
}


def capabilities_for(model_id: str) -> ProviderCapabilities | None:
    """Resolve the capabilities for a model id, exact match before fallback.

    Lookup order:

    1. Exact match in :data:`STATIC_CATALOG` -> return that object.
    2. Otherwise derive the provider prefix (the segment before the first
       ``/``) and return :data:`PROVIDER_FALLBACK` for it, if present.
    3. Unknown provider -> ``None``.

    The prefix is split locally (rather than importing the registry) to keep
    this pure-data module free of cross-module coupling.

    Args:
        model_id: A provider-prefixed model id, e.g. ``"openai/gpt-4.1"``. A
            bare id with no ``/`` is treated as its own prefix.

    Returns:
        The matching :class:`ProviderCapabilities`, or ``None`` when neither the
        exact model nor its provider prefix is known.
    """
    exact = STATIC_CATALOG.get(model_id)
    if exact is not None:
        return exact
    prefix = model_id.split("/", 1)[0] if "/" in model_id else model_id
    return PROVIDER_FALLBACK.get(prefix)
