"""Lean static provider capability catalog (CAC-03).

This package answers one question, offline and from checked-in data only: for a
given provider-prefixed model id, which generation knobs does the provider
accept? It exists so callers can pick safe request parameters without a live
probe and without coupling to any vendor SDK.

The catalog is intentionally *lean*: it records boolean capability flags plus a
pair of optional integer limits, and nothing else. It never holds a key value,
never reads the environment, and never touches the network or disk. A later
ticket (COC-05) will add explicit, opt-in refresh/discovery to enrich these
conservative defaults; until then the checked-in snapshot is the only source.

Public surface:

* :class:`ProviderCapabilities` — the frozen capability record.
* :data:`STATIC_CATALOG` — model id -> capabilities for every built-in model.
* :data:`PROVIDER_FALLBACK` — provider prefix -> capabilities, so a known
  provider's non-default model still resolves to sane caps.
* :func:`capabilities_for` — exact-match-then-prefix-fallback lookup.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    """Static, conservative capability flags for one model (or provider).

    Every value here is a *static, conservative default*: a checked-in snapshot
    chosen to never overclaim. A ``True`` means the capability is documented by
    the provider for this model; a ``False`` means either the provider does not
    support it or support is unverified and we decline to claim it. Limits that
    we cannot pin to a stable, non-volatile number are left ``None`` rather than
    guessed.

    Attributes:
        supports_temperature: Provider accepts a ``temperature`` sampling knob.
        supports_top_p: Provider accepts a ``top_p`` nucleus-sampling knob.
        supports_max_output_tokens: Provider accepts a max-output-tokens cap.
        supports_reasoning_effort: Provider accepts a reasoning-effort control
            (e.g. an explicit reasoning budget). Default ``False`` — only
            reasoning models expose this and none of the built-ins are claimed.
        supports_streaming: Provider can stream tokens incrementally.
        supports_system_prompt: Provider honors a distinct system message.
        supports_tool_calls: Provider supports function/tool calling. Default
            ``False`` — not load-bearing for the council, claimed conservatively.
        supports_structured_output: Provider supports schema-constrained output
            (json_schema response_format, responseSchema, or tool-use schema).
        supports_json_mode: Provider supports an OpenAI-style ``json_object``
            response mode (free-form JSON, not a strict schema).
        context_window: Total context window in tokens, or ``None`` when we
            decline to pin a volatile number.
        output_token_limit: Max output tokens in a single response, or ``None``
            when we decline to pin a volatile number.
    """

    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_max_output_tokens: bool = True
    supports_reasoning_effort: bool = False
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_tool_calls: bool = False
    supports_structured_output: bool = False
    supports_json_mode: bool = False
    context_window: int | None = None
    output_token_limit: int | None = None


# Imported AFTER ProviderCapabilities is defined: ``.static`` imports the class
# from this package, so the class must exist first or the import would crash.
from .static import (  # noqa: E402 -- intentional: break the import cycle
    PROVIDER_FALLBACK,
    STATIC_CATALOG,
    capabilities_for,
)

__all__ = [
    "ProviderCapabilities",
    "STATIC_CATALOG",
    "capabilities_for",
    "PROVIDER_FALLBACK",
]
