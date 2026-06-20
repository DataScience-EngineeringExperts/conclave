"""Adapter registry: the single extension seam for adding providers.

:func:`resolve_adapter` maps a model id to the concrete adapter that speaks its
wire format. Adding a provider is a small, well-defined change:

* **OpenAI-compatible provider** -> add an entry to
  :data:`conclave.adapters.openai_compat.OPENAI_COMPAT_URLS` and its env var to
  :data:`conclave.registry.PROVIDER_ENV_VARS` (or declare it in ``config.yml``
  under ``endpoints:`` for a zero-code addition).
* **Native (non-compatible) provider** -> write an adapter satisfying
  :class:`conclave.adapters.base.ProviderAdapter` and register it in
  :data:`_NATIVE_BUILDERS` below.

The registry is config-aware: an unknown prefix that matches a declared custom
OpenAI-compatible endpoint is served by a generic
:class:`OpenAICompatAdapter`; an unknown prefix with no declaration raises a
clear :class:`ProviderError` that the call path turns into a helpful
``ModelAnswer.error``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..registry import PROVIDER_ENV_VARS, provider_prefix
from .anthropic import AnthropicAdapter
from .base import OutputContract, ProviderAdapter, ProviderError, redact
from .gemini import GeminiAdapter
from .openai_compat import OPENAI_COMPAT_URLS, OpenAICompatAdapter

if TYPE_CHECKING:  # avoid a config import cycle at runtime
    from ..config import ConclaveConfig

__all__ = [
    "ProviderAdapter",
    "ProviderError",
    "OutputContract",
    "redact",
    "resolve_adapter",
    "OpenAICompatAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
]

# Native (non OpenAI-compatible) providers -> zero-arg adapter builders. Each
# adapter sources its env var names from PROVIDER_ENV_VARS so the mapping is DRY.
_NATIVE_BUILDERS: dict[str, Callable[[], ProviderAdapter]] = {
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def _openai_compat_adapter(prefix: str) -> OpenAICompatAdapter:
    """Build the built-in OpenAI-compatible adapter for a known prefix."""
    return OpenAICompatAdapter(
        prefix=prefix,
        completions_url=OPENAI_COMPAT_URLS[prefix],
        env_vars=tuple(PROVIDER_ENV_VARS[prefix]),
    )


def resolve_adapter(model_id: str, config: ConclaveConfig | None = None) -> ProviderAdapter:
    """Resolve a model id to the adapter that speaks its provider's wire format.

    Args:
        model_id: A provider-prefixed id (e.g. ``"xai/grok-4.3"``). The prefix is
            extracted with :func:`conclave.registry.provider_prefix`.
        config: Optional config carrying user-declared custom OpenAI-compatible
            ``endpoints``. When provided, an unknown prefix matching a declared
            endpoint is served by a generic OpenAI-compatible adapter.

    Returns:
        A concrete :class:`ProviderAdapter` instance for ``model_id``.

    Raises:
        ProviderError: When the prefix is unknown and no custom endpoint declares
            it. The message names the prefix and the remedy.
    """
    prefix = provider_prefix(model_id)

    if prefix in OPENAI_COMPAT_URLS:
        return _openai_compat_adapter(prefix)
    if prefix in _NATIVE_BUILDERS:
        return _NATIVE_BUILDERS[prefix]()

    # Unknown prefix: serve it only if the user declared a custom endpoint for it.
    if config is not None and prefix in config.endpoints:
        spec = config.endpoints[prefix]
        return OpenAICompatAdapter(
            prefix=prefix,
            completions_url=spec.completions_url,
            env_vars=(spec.env_var,),
        )

    raise ProviderError(
        f"unknown provider '{prefix}' for model '{model_id}': no built-in adapter "
        "and no custom OpenAI-compatible endpoint declared in config "
        "(add it under 'endpoints:' in ~/.conclave/config.yml)"
    )
