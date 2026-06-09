"""Provider registry: the single source of truth for built-in provider metadata.

This module owns, in ONE place, everything that defines a built-in provider:

* its provider prefix,
* the env var(s) that satisfy its key,
* (for OpenAI-compatible providers) its full ``/chat/completions`` URL,
* the friendly-name -> default model id mapping.

Previously the env-var names lived here while the OpenAI-compatible URLs lived in
``adapters.openai_compat.OPENAI_COMPAT_URLS``. The two tables could silently
drift: a URL present without a matching env var (or vice versa) surfaced only as a
runtime ``KeyError`` deep in the call path. Now the per-provider metadata is
declared once (see :data:`OPENAI_COMPAT_PROVIDERS` and :data:`NATIVE_PROVIDERS`)
and the derived tables (:data:`PROVIDER_ENV_VARS`) are built from it, so adding or
removing a provider can never desync. A module-level consistency check
(:func:`_assert_metadata_consistent`) runs at import and fails loudly if the
adapter layer's URL table ever drifts from this source of truth -- turning a
silent per-call KeyError into an immediate, diagnosable startup error.

It NEVER reads or returns a key value -- only whether the relevant variable is
set and non-empty. That keeps secrets out of every code path and out of any
serialized output.

### Adding a built-in provider (single edit step)

* **OpenAI-compatible provider** -> add one entry to
  :data:`OPENAI_COMPAT_PROVIDERS` (prefix -> URL + env vars) **and** the matching
  URL to ``adapters.openai_compat.OPENAI_COMPAT_URLS``. The import-time check
  guarantees you cannot forget one half.
* **Native (non-compatible) provider** -> add its env var(s) to
  :data:`NATIVE_PROVIDERS` and register its adapter in
  ``adapters._NATIVE_BUILDERS``.
* **Zero-code custom provider** -> declare it under ``endpoints:`` in
  ``~/.conclave/config.yml``; no registry edit needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class RegistryError(RuntimeError):
    """Raised at import time when provider metadata is internally inconsistent.

    A loud, immediate failure here replaces the former silent drift between the
    env-var table and the OpenAI-compatible URL table, which used to escape as a
    runtime ``KeyError`` deep inside the per-call adapter resolution.
    """


@dataclass(frozen=True)
class OpenAICompatProvider:
    """Authoritative metadata for one built-in OpenAI-compatible provider.

    Attributes:
        completions_url: Full POST URL of the provider's ``/chat/completions``
            endpoint.
        env_vars: Candidate env var NAMES (never values); the first present one is
            the active key. Order matters for fallbacks.
    """

    completions_url: str
    env_vars: tuple[str, ...]


# Friendly name -> default provider-prefixed model id. Overridable via ~/.conclave/config.yml.
# Every entry is a DIRECT vendor key to a DIRECT vendor endpoint (no aggregator/router) --
# the no-middleman positioning in PRODUCT_DESIGN_DOCUMENT.md §11 is load-bearing.
DEFAULT_MODELS: dict[str, str] = {
    "grok": "xai/grok-4.3",
    "gemini": "gemini/gemini-2.5-pro",
    "claude": "anthropic/claude-sonnet-4-6",
    "perplexity": "perplexity/sonar-pro",
    "openai": "openai/gpt-4.1",
    "groq": "groq/llama-3.3-70b-versatile",
    "deepseek": "deepseek/deepseek-chat",
    "mistral": "mistral/mistral-large-latest",
    "together": "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

# SINGLE SOURCE OF TRUTH for built-in OpenAI-compatible providers: prefix ->
# (URL + env vars) in one place. The adapter layer's OPENAI_COMPAT_URLS is kept in
# lockstep with this table by the import-time consistency check below.
OPENAI_COMPAT_PROVIDERS: dict[str, OpenAICompatProvider] = {
    "openai": OpenAICompatProvider(
        completions_url="https://api.openai.com/v1/chat/completions",
        env_vars=("OPENAI_API_KEY",),
    ),
    "xai": OpenAICompatProvider(
        completions_url="https://api.x.ai/v1/chat/completions",
        env_vars=("XAI_API_KEY",),
    ),
    # Perplexity has NO /v1 segment; that detail lives here, once.
    "perplexity": OpenAICompatProvider(
        completions_url="https://api.perplexity.ai/chat/completions",
        env_vars=("PERPLEXITY_API_KEY",),
    ),
    # Groq's OpenAI-compatible surface lives under an /openai/v1 path prefix
    # (https://console.groq.com/docs/openai). Direct vendor key, direct endpoint.
    "groq": OpenAICompatProvider(
        completions_url="https://api.groq.com/openai/v1/chat/completions",
        env_vars=("GROQ_API_KEY",),
    ),
    # DeepSeek's /v1 segment is OpenAI-SDK compatibility sugar (no version meaning)
    # and is accepted (https://api-docs.deepseek.com/).
    "deepseek": OpenAICompatProvider(
        completions_url="https://api.deepseek.com/v1/chat/completions",
        env_vars=("DEEPSEEK_API_KEY",),
    ),
    "mistral": OpenAICompatProvider(
        completions_url="https://api.mistral.ai/v1/chat/completions",
        env_vars=("MISTRAL_API_KEY",),
    ),
    # Together's canonical REST host is api.together.xyz (https://docs.together.ai).
    "together": OpenAICompatProvider(
        completions_url="https://api.together.xyz/v1/chat/completions",
        env_vars=("TOGETHER_API_KEY",),
    ),
}

# Native (non OpenAI-compatible) providers: prefix -> candidate env var names.
# These have bespoke adapters in adapters._NATIVE_BUILDERS rather than a URL here.
NATIVE_PROVIDERS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

# Derived: provider prefix -> the env var(s) that satisfy it. Built from
# the single-source tables above so it can never drift from them. The first
# present var in the list is the active key. Order matters for fallbacks.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    **{prefix: list(meta.env_vars) for prefix, meta in OPENAI_COMPAT_PROVIDERS.items()},
    **{prefix: list(env_vars) for prefix, env_vars in NATIVE_PROVIDERS.items()},
}

DEFAULT_SYNTHESIZER = "claude"


def _assert_metadata_consistent() -> None:
    """Fail loudly at import if provider metadata has drifted.

    Two invariants are enforced:

    1. The adapter layer's ``OPENAI_COMPAT_URLS`` must have exactly the same
       prefixes as :data:`OPENAI_COMPAT_PROVIDERS`, with identical URLs. This is
       the drift that used to escape as a runtime ``KeyError`` in
       ``adapters._openai_compat_adapter`` (issue #19).
    2. Every prefix in :data:`OPENAI_COMPAT_PROVIDERS` must declare at least one
       env var, so an OpenAI-compatible provider can never be half-declared.

    Raises:
        RegistryError: with a message naming the drifting prefixes / URLs.
    """
    # Imported lazily inside the function to avoid any import-order coupling:
    # openai_compat does not import registry, so this edge is safe and acyclic.
    from .adapters.openai_compat import OPENAI_COMPAT_URLS

    source_prefixes = set(OPENAI_COMPAT_PROVIDERS)
    adapter_prefixes = set(OPENAI_COMPAT_URLS)

    missing_in_adapter = source_prefixes - adapter_prefixes
    missing_in_source = adapter_prefixes - source_prefixes
    if missing_in_adapter or missing_in_source:
        raise RegistryError(
            "OpenAI-compatible provider tables have drifted: "
            f"prefixes only in registry.OPENAI_COMPAT_PROVIDERS={sorted(missing_in_adapter)}, "
            f"prefixes only in adapters.OPENAI_COMPAT_URLS={sorted(missing_in_source)}. "
            "Add the missing entry to both (registry is the source of truth)."
        )

    mismatched_urls = {
        prefix: (OPENAI_COMPAT_PROVIDERS[prefix].completions_url, OPENAI_COMPAT_URLS[prefix])
        for prefix in source_prefixes
        if OPENAI_COMPAT_PROVIDERS[prefix].completions_url != OPENAI_COMPAT_URLS[prefix]
    }
    if mismatched_urls:
        raise RegistryError(
            "OpenAI-compatible URL drift between registry and adapters: "
            f"{mismatched_urls} (registry is the source of truth)."
        )

    half_declared = [
        prefix for prefix, meta in OPENAI_COMPAT_PROVIDERS.items() if not meta.env_vars
    ]
    if half_declared:
        raise RegistryError(
            f"OpenAI-compatible providers declare a URL but no env var: {half_declared}. "
            "Every provider needs at least one env var name."
        )


def provider_prefix(model_id: str) -> str:
    """Extract the provider prefix from a model id.

    Args:
        model_id: e.g. ``"xai/grok-4.3"``.

    Returns:
        The provider prefix (``"xai"``). If the id has no ``/`` we treat the
        whole string as the prefix (the bare-name convention for unprefixed ids).
    """
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


def required_env_vars(model_id: str) -> list[str]:
    """Return the candidate env var names that can satisfy this model.

    Unknown providers return an empty list, meaning "we can't statically prove a
    key is needed"; the call is still attempted and any auth error is caught.
    """
    return PROVIDER_ENV_VARS.get(provider_prefix(model_id), [])


def key_present(model_id: str) -> bool:
    """True if at least one satisfying env var is set and non-empty.

    Never returns or logs the value. Unknown providers return True so we don't
    pre-emptively skip a model we can't reason about; the live call decides.
    """
    candidates = required_env_vars(model_id)
    if not candidates:
        return True
    return any(os.environ.get(var, "").strip() for var in candidates)


def key_source(model_id: str) -> str | None:
    """Return the NAME of the env var providing the key, or None if absent.

    Only the variable name is returned -- never the value.
    """
    for var in required_env_vars(model_id):
        if os.environ.get(var, "").strip():
            return var
    return None


# Enforce the single-source-of-truth invariant at import time. Any drift between
# this registry and the adapter layer's URL table is a programming error in a
# provider definition; surfacing it here (loudly, once) is strictly better than
# letting it escape as a per-call KeyError deep in the call path (issue #19).
_assert_metadata_consistent()
