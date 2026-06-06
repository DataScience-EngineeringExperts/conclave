"""Provider registry: friendly names, LiteLLM model ids, and key presence.

This module is the single source of truth for which environment variable a
given provider needs. It NEVER reads or returns a key value -- only whether the
relevant variable is set and non-empty. That keeps secrets out of every code
path and out of any serialized output.
"""

from __future__ import annotations

import os

# Friendly name -> default LiteLLM model id. Overridable via ~/.conclave/config.yml.
DEFAULT_MODELS: dict[str, str] = {
    "grok": "xai/grok-4.3",
    "gemini": "gemini/gemini-2.5-pro",
    "claude": "anthropic/claude-sonnet-4-6",
    "perplexity": "perplexity/sonar-pro",
    "openai": "openai/gpt-4.1",
}

# LiteLLM provider prefix -> the env var(s) that satisfy it. The first present
# var in the list is considered the active key. Order matters for fallbacks.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "xai": ["XAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "perplexity": ["PERPLEXITY_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
}

DEFAULT_SYNTHESIZER = "claude"


def provider_prefix(model_id: str) -> str:
    """Extract the LiteLLM provider prefix from a model id.

    Args:
        model_id: e.g. ``"xai/grok-4.3"``.

    Returns:
        The provider prefix (``"xai"``). If the id has no ``/`` we treat the
        whole string as the prefix, which mirrors LiteLLM's bare-name handling.
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
