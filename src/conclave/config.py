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


class ConclaveConfig(BaseModel):
    """Resolved configuration after merging file over defaults.

    Attributes:
        models: friendly name -> LiteLLM model id.
        councils: named lists of friendly names.
        synthesizer: friendly name of the default synthesizer model.
    """

    models: dict[str, str] = Field(default_factory=dict)
    councils: dict[str, list[str]] = Field(default_factory=dict)
    synthesizer: str = DEFAULT_SYNTHESIZER

    def resolve_model_id(self, name: str) -> str:
        """Map a friendly name to a LiteLLM model id.

        If ``name`` is unknown it is passed through verbatim, so a user can name
        a council member by a raw LiteLLM id (e.g. ``"openai/gpt-4o"``).
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


def load_config(path: Path | None = None) -> ConclaveConfig:
    """Load and merge configuration.

    Args:
        path: Optional override path. Defaults to ``~/.conclave/config.yml`` or
            the ``CONCLAVE_CONFIG`` env var if set.

    Returns:
        A fully merged ``ConclaveConfig``. Built-in model defaults are always
        present; file entries override or extend them.
    """
    if path is None:
        env_path = os.environ.get("CONCLAVE_CONFIG")
        path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH

    raw = _read_yaml(path)

    merged_models = dict(DEFAULT_MODELS)
    merged_models.update(raw.get("models", {}) or {})

    councils = {
        name: list(members)
        for name, members in (raw.get("councils", {}) or {}).items()
    }
    # Always provide a "default" council if none defined: all known providers.
    councils.setdefault("default", list(DEFAULT_MODELS.keys()))

    synthesizer = raw.get("synthesizer", DEFAULT_SYNTHESIZER)

    return ConclaveConfig(
        models=merged_models, councils=councils, synthesizer=synthesizer
    )
