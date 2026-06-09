"""Optional on-disk result cache for council runs (off by default).

This is the §9 #4 roadmap item: an opt-in cache keyed on
``(prompt, council, mode, model ids)`` so repeated or eval runs are cheap. It is
**off by default** and **never persists key material** -- the cache key and the
stored payload are derived solely from the normalized prompt, the ordered council
member friendly-names + resolved model ids, the run mode, the synthesizer/judge
identity, and the mode parameters that affect output. No environment variable is
read here; no key value reaches the key string or the on-disk artifact.

Storage
=======
Entries live one-per-file under ``$XDG_CACHE_HOME/conclave`` (falling back to
``~/.cache/conclave``). Each file is named ``<sha256-hex>.json`` and holds the
JSON serialization of a :class:`conclave.models.CouncilResult` (via
``model_dump(mode="json")``), which by construction carries no secrets.

Graceful degradation
====================
A corrupt, unreadable, or schema-incompatible cache entry is treated as a **miss**
(logged at warning level), never an error: a bad cache file can never crash a run.
Writes that fail (e.g. a read-only cache dir) are likewise logged and swallowed --
caching is a best-effort optimization, never a correctness dependency.

Key-ordering choice
===================
Member order is **preserved** (not sorted) in the cache key. For ``synthesize`` /
``raw`` the member order does not change the output, but for ``debate`` and
``adversarial`` it does: the adversarial proposer defaults to the first member and
debate assigns stable letter labels by member position. Preserving order is
therefore the conservative, always-correct choice -- two runs collide only when
they would genuinely produce equivalent results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from pydantic import ValidationError

from .logging import get_logger
from .models import CouncilResult

logger = get_logger("cache")

# Bumped if the cache-key composition or stored schema changes incompatibly, so
# old entries simply miss instead of being mis-served against new code.
_CACHE_VERSION = 1

_WHITESPACE = re.compile(r"\s+")


def cache_dir() -> Path:
    """Return the conclave cache directory, honoring ``XDG_CACHE_HOME``.

    Falls back to ``~/.cache/conclave`` when ``XDG_CACHE_HOME`` is unset or empty.
    The directory is not created here; :func:`store` creates it lazily on write.
    """
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "conclave"


def _normalize_prompt(prompt: str) -> str:
    """Collapse runs of whitespace and strip ends for a stable prompt key.

    Two prompts that differ only in incidental whitespace should hit the same
    cache entry; semantic content is otherwise preserved verbatim.
    """
    return _WHITESPACE.sub(" ", prompt).strip()


def make_key(
    *,
    prompt: str,
    mode: str,
    members: list[tuple[str, str]],
    synthesizer: str | None,
    synthesizer_model_id: str | None,
    temperature: float,
    rounds: int | None = None,
    proposer: str | None = None,
) -> str:
    """Build the stable cache key (sha256 hex) for a council run.

    The key is a SHA-256 over a canonical JSON document of only output-affecting,
    secret-free identity:

    * normalized prompt,
    * run mode,
    * ordered ``(friendly_name, resolved_model_id)`` member pairs (order matters --
      see module docstring),
    * synthesizer/judge friendly name + resolved model id,
    * temperature, and mode params (``rounds`` for debate, ``proposer`` for
      adversarial) when they apply.

    Args:
        prompt: The raw user prompt (normalized internally).
        mode: ``"synthesize" | "raw" | "debate" | "adversarial"``.
        members: Ordered ``(friendly_name, resolved_model_id)`` pairs actually run.
        synthesizer: Synthesizer/judge friendly name (``None`` when not applicable).
        synthesizer_model_id: Resolved synthesizer/judge model id.
        temperature: Sampling temperature (affects output).
        rounds: Debate round count (included only for ``debate``).
        proposer: Adversarial proposer friendly name (included only for
            ``adversarial``).

    Returns:
        A 64-char lowercase hex SHA-256 digest. Contains zero key material.
    """
    payload: dict[str, object] = {
        "v": _CACHE_VERSION,
        "prompt": _normalize_prompt(prompt),
        "mode": mode,
        # Pairs as lists so JSON round-trips; order preserved deliberately.
        "members": [[name, model_id] for name, model_id in members],
        "synthesizer": synthesizer,
        "synthesizer_model_id": synthesizer_model_id,
        "temperature": temperature,
    }
    if mode == "debate":
        payload["rounds"] = rounds
    if mode == "adversarial":
        payload["proposer"] = proposer

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entry_path(key: str) -> Path:
    """Map a cache key to its on-disk entry path."""
    return cache_dir() / f"{key}.json"


def load(key: str) -> CouncilResult | None:
    """Return the cached :class:`CouncilResult` for ``key``, or ``None`` on miss.

    A missing file is a normal miss (silent). A present-but-unreadable or
    schema-incompatible file is a degraded miss: it is logged at warning level and
    treated as absent so a corrupt entry can never crash a run.

    Args:
        key: The cache key from :func:`make_key`.

    Returns:
        The deserialized result with ``cached=True`` set, or ``None``.
    """
    try:
        path = _entry_path(key)
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("cache read failed for key %s: %s; treating as miss", key[:12], exc)
        return None

    try:
        data = json.loads(raw)
        result = CouncilResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.warning("corrupt cache entry %s: %s; treating as miss", path, exc)
        return None

    # Mark as cache-served so consumers can distinguish a hit from a live run.
    result.cached = True
    return result


def store(key: str, result: CouncilResult) -> None:
    """Persist ``result`` under ``key``, best-effort (failures are swallowed).

    The cache directory is created lazily. Any write failure (read-only dir, disk
    full, serialization error) is logged at warning level and ignored -- caching
    must never turn a successful run into a failure. The stored payload is
    ``result.model_dump(mode="json")``, which carries no secrets.

    The ``cached`` flag is normalized to ``False`` before writing so a stored
    entry reflects how it was produced (live), not how it will later be served.

    Args:
        key: The cache key from :func:`make_key`.
        result: The live :class:`CouncilResult` to persist.
    """
    try:
        path = _entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        payload["cached"] = False
        # Atomic-ish write: write to a temp sibling then replace, so a crash mid
        # write never leaves a half-written (corrupt) entry behind.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "cache write failed for key %s: %s; continuing without caching", key[:12], exc
        )
