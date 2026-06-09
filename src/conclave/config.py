"""Configuration loading and merging for conclave.

Loads ``~/.conclave/config.yml`` (if present) and merges it over the built-in
defaults. Config references providers by friendly NAME only and never contains
key values. A typical config looks like::

    models:
      grok: xai/grok-4.3
      claude: anthropic/claude-sonnet-4-6
    councils:
      default: [grok, claude, perplexity]
      fast: [grok, perplexity]
    synthesizer: claude
    endpoints:                 # optional: custom OpenAI-compatible providers
      together:
        completions_url: https://api.together.xyz/v1/chat/completions
        env_var: TOGETHER_API_KEY

A declared endpoint makes its prefix usable in a model id (``together/<model>``)
with no code change: the adapter registry serves it via the generic
OpenAI-compatible adapter. ``env_var`` still names a variable only -- never a value.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .logging import get_logger
from .registry import DEFAULT_MODELS, DEFAULT_SYNTHESIZER

logger = get_logger("config")

DEFAULT_CONFIG_PATH = Path.home() / ".conclave" / "config.yml"


class CustomEndpoint(BaseModel):
    """A user-declared OpenAI-compatible provider.

    Lets a user add a provider without touching code: declare its prefix, full
    chat-completions URL, and the env var that supplies its key. The adapter
    registry resolves matching model ids through the generic OpenAI-compat path.

    Attributes:
        completions_url: Full POST URL of the ``/chat/completions`` endpoint.
        env_var: NAME of the env var holding the key (never the value).
    """

    completions_url: str
    env_var: str


class ConclaveConfig(BaseModel):
    """Resolved configuration after merging file over defaults.

    Attributes:
        models: friendly name -> provider model id.
        councils: named lists of friendly names.
        synthesizer: friendly name of the default synthesizer model.
        endpoints: prefix -> custom OpenAI-compatible endpoint declaration.
        cache: opt-in result cache (off by default). When ``True`` an identical
            repeat run is served from the on-disk cache (see :mod:`conclave.cache`)
            instead of re-calling the providers. The cache never persists keys.
            A ``--cache/--no-cache`` CLI flag overrides this per invocation.
    """

    models: dict[str, str] = Field(default_factory=dict)
    councils: dict[str, list[str]] = Field(default_factory=dict)
    synthesizer: str = DEFAULT_SYNTHESIZER
    endpoints: dict[str, CustomEndpoint] = Field(default_factory=dict)
    cache: bool = False

    def resolve_model_id(self, name: str) -> str:
        """Map a friendly name to a provider-prefixed model id.

        If ``name`` is unknown it is passed through verbatim, so a user can name
        a council member by a raw provider-prefixed id (e.g. ``"openai/gpt-4o"``).
        """
        return self.models.get(name, name)

    def resolve_council(self, name_or_csv: str) -> list[str]:
        """Resolve a council selector into a list of friendly names.

        ``name_or_csv`` may be a named council from config (e.g. ``"default"``)
        or a comma-separated list of friendly names (``"grok,claude"``).
        """
        if name_or_csv in self.councils:
            return list(self.councils[name_or_csv])
        return [part.strip() for part in name_or_csv.split(",") if part.strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, returning {} on absence or parse error."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            logger.warning("config at %s is not a mapping; ignoring", path)
            return {}
        return data
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("failed to read config %s: %s", path, exc)
        return {}


def _resolve_path(path: Path | None) -> Path:
    """Resolve the effective config path from an explicit arg or the environment."""
    if path is not None:
        return path
    env_path = os.environ.get("CONCLAVE_CONFIG")
    return Path(env_path) if env_path else DEFAULT_CONFIG_PATH


def _cache_key(path: Path) -> tuple[str, float]:
    """A (path, mtime) cache key. mtime is 0.0 when the file is absent.

    Keying on mtime means an edited or newly created config is picked up on the
    next call, while an unchanged file is served from the in-process cache --
    eliminating the repeated disk read + YAML parse that ``call_model`` used to
    incur on every single model call (issue #15), without ever serving stale data.
    """
    try:
        return (str(path), path.stat().st_mtime)
    except OSError:
        return (str(path), 0.0)


# In-process memo: (path, mtime) -> merged config. Cleared automatically when the
# underlying file changes (mtime moves) and explicitly via ``clear_config_cache``.
_CONFIG_CACHE: dict[tuple[str, float], ConclaveConfig] = {}


def clear_config_cache() -> None:
    """Drop the memoized config. Intended for tests and long-lived processes."""
    _CONFIG_CACHE.clear()


def load_config(path: Path | None = None) -> ConclaveConfig:
    """Load and merge configuration, memoized by (path, mtime).

    The result is cached in-process keyed on the resolved path and its
    modification time, so repeated calls within a run (e.g. one per model call)
    do not re-read disk or re-parse YAML. The cache self-invalidates when the
    file's mtime changes, preserving the previous always-fresh behavior across
    edits while removing the redundant hot-path reads (issue #15).

    Args:
        path: Optional override path. Defaults to ``~/.conclave/config.yml`` or
            the ``CONCLAVE_CONFIG`` env var if set.

    Returns:
        A fully merged ``ConclaveConfig``. Built-in model defaults are always
        present; file entries override or extend them.
    """
    resolved = _resolve_path(path)
    key = _cache_key(resolved)
    cached = _CONFIG_CACHE.get(key)
    if cached is not None:
        return cached

    config = _load_config_uncached(resolved)
    _CONFIG_CACHE[key] = config
    return config


def _load_config_uncached(path: Path) -> ConclaveConfig:
    """Read + merge config from ``path`` with no caching (the real disk work)."""
    raw = _read_yaml(path)

    merged_models = dict(DEFAULT_MODELS)
    merged_models.update(raw.get("models", {}) or {})

    councils = {name: list(members) for name, members in (raw.get("councils", {}) or {}).items()}
    # Always provide a "default" council if none defined: all known providers.
    councils.setdefault("default", list(DEFAULT_MODELS.keys()))

    synthesizer = raw.get("synthesizer", DEFAULT_SYNTHESIZER)

    endpoints = {
        prefix: CustomEndpoint(**spec)
        for prefix, spec in (raw.get("endpoints", {}) or {}).items()
        if isinstance(spec, dict)
    }

    # Off by default; only an explicit truthy ``cache: true`` in the file enables it.
    cache = bool(raw.get("cache", False))

    return ConclaveConfig(
        models=merged_models,
        councils=councils,
        synthesizer=synthesizer,
        endpoints=endpoints,
        cache=cache,
    )
