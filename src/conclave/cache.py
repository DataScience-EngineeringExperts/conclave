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
``~/.cache/conclave``). Each file is named ``<sha256-hex>.json`` and holds a
versioned envelope around the JSON serialization of a
:class:`conclave.models.CouncilResult`. Unversioned/old-format envelopes are
misses, never migrated or replayed against current protocol semantics.

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
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import ValidationError

from .logging import get_logger
from .models import ELITE_PROTOCOL_VERSION, CouncilResult
from .prompts import ELITE_PROMPT_VERSION, SYNTHESIS_PROMPT_VERSION
from .verdict import (
    VERDICT_EXTRACTION_PROMPT_VERSION,
    VERDICT_SCHEMA_VERSION,
)

logger = get_logger("cache")

# Bumped if the cache-key composition or stored schema changes incompatibly, so
# old entries simply miss instead of being mis-served against new code.
CACHE_FORMAT_VERSION = "2"

_WHITESPACE = re.compile(r"\s+")
_SECRET_QUERY_PARTS = (
    "authorization",
    "auth",
    "credential",
    "passwd",
    "password",
    "secret",
    "signature",
    "token",
)
_SECRET_QUERY_KEYS = {
    "access_key",
    "api_key",
    "apikey",
    "code",
    "key",
    "sig",
}
_SECRET_VALUE_MARKERS = (
    "akia",
    "authorization:",
    "bearer ",
    "ghp_",
    "sk-",
    "sk_",
    "x-api-key",
)


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


def _digest(value: str) -> str:
    """Return a one-way identity fingerprint without retaining ``value``."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _endpoint_fingerprint(raw_url: str) -> str:
    """Fingerprint output-affecting endpoint routing without credential material.

    Userinfo, fragments, and secret-like query parameters are deliberately
    excluded. Safe query parameters (for example an API version) remain part of
    the fingerprint because they can change model behavior. Only the digest is
    returned; the normalized URL never enters an identity document or diagnostic.
    """
    parsed = urlsplit(raw_url)
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        # A malformed configured port will fail at request time, but cache
        # identity construction must remain safe and deterministic first.
        port = None
        hostname = f"{hostname}:invalid-port"
    if port is not None:
        hostname = f"{hostname}:{port}"
    safe_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        value_lower = value.lower()
        if key_lower in _SECRET_QUERY_KEYS or any(
            part in key_lower for part in _SECRET_QUERY_PARTS
        ):
            continue
        if any(marker in value_lower for marker in _SECRET_VALUE_MARKERS):
            continue
        safe_query.append((key, value))
    safe_query.sort()
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            hostname,
            parsed.path.rstrip("/") or "/",
            urlencode(safe_query),
            "",
        )
    )
    return _digest(normalized)


def build_identity(
    *,
    prompt: str,
    mode: str,
    members: list[tuple[str, str]],
    synthesizer: str | None,
    synthesizer_model_id: str | None,
    temperature: float,
    timeout: float = 120.0,
    rounds: int | None = None,
    proposer: str | None = None,
    converge_threshold: float | None = None,
    choices: list[str] | None = None,
    extract_verdict: bool = True,
    endpoint_urls: Mapping[str, str] | None = None,
    source_bundle_digest: str | None = None,
    cache_format_version: str = CACHE_FORMAT_VERSION,
    protocol_version: str = ELITE_PROTOCOL_VERSION,
    synthesis_prompt_version: str = SYNTHESIS_PROMPT_VERSION,
    elite_prompt_version: str = ELITE_PROMPT_VERSION,
    verdict_schema_version: str = VERDICT_SCHEMA_VERSION,
    verdict_prompt_version: str = VERDICT_EXTRACTION_PROMPT_VERSION,
) -> dict[str, object]:
    """Build the canonical, secret-free identity document for a council run.

    Raw endpoint URLs and source bundle values never enter the returned document;
    only sanitized one-way fingerprints do. API keys are not accepted and no
    environment value is read here.
    """
    payload: dict[str, object] = {
        "versions": {
            "cache_format": cache_format_version,
            "elite_protocol": protocol_version,
            "synthesis_prompt": synthesis_prompt_version,
            "elite_prompt": elite_prompt_version,
            "verdict_schema": verdict_schema_version,
            "verdict_extraction_prompt": verdict_prompt_version,
        },
        "prompt_fingerprint": _digest(_normalize_prompt(prompt)),
        "mode": mode,
        # Pairs as lists so JSON round-trips; order preserved deliberately.
        "members": [[name, model_id] for name, model_id in members],
        "synthesizer": [synthesizer, synthesizer_model_id],
        "generation": {"temperature": temperature, "timeout": timeout},
        "extract_verdict": extract_verdict,
        "endpoint_fingerprints": {
            prefix: _endpoint_fingerprint(url)
            for prefix, url in sorted((endpoint_urls or {}).items())
        },
        # Re-hash the caller-supplied digest. This keeps even a malformed caller
        # value out of the inspectable identity while preserving invalidation.
        "source_bundle_fingerprint": (
            _digest(source_bundle_digest) if source_bundle_digest is not None else None
        ),
        "mode_params": {},
    }
    mode_params = payload["mode_params"]
    assert isinstance(mode_params, dict)
    if mode == "debate":
        mode_params["rounds"] = rounds
        mode_params["converge_threshold"] = converge_threshold
    if mode == "adversarial":
        mode_params["proposer"] = proposer
    if mode == "vote":
        mode_params["choices"] = choices or []
    return payload


def make_key(
    *,
    prompt: str,
    mode: str,
    members: list[tuple[str, str]],
    synthesizer: str | None,
    synthesizer_model_id: str | None,
    temperature: float,
    timeout: float = 120.0,
    rounds: int | None = None,
    proposer: str | None = None,
    converge_threshold: float | None = None,
    choices: list[str] | None = None,
    extract_verdict: bool = True,
    endpoint_urls: Mapping[str, str] | None = None,
    source_bundle_digest: str | None = None,
    cache_format_version: str = CACHE_FORMAT_VERSION,
    protocol_version: str = ELITE_PROTOCOL_VERSION,
    synthesis_prompt_version: str = SYNTHESIS_PROMPT_VERSION,
    elite_prompt_version: str = ELITE_PROMPT_VERSION,
    verdict_schema_version: str = VERDICT_SCHEMA_VERSION,
    verdict_prompt_version: str = VERDICT_EXTRACTION_PROMPT_VERSION,
) -> str:
    """Build the stable SHA-256 cache key from canonical secret-free identity.

    Identity covers:

    * normalized prompt,
    * run mode,
    * ordered ``(friendly_name, resolved_model_id)`` member pairs (order matters --
      see module docstring),
    * synthesizer/judge friendly name + resolved model id,
    * generation settings and mode parameters,
    * protocol/prompt/schema/cache-format versions,
    * verdict extraction behavior, custom endpoint routing, and an optional
      future source-bundle digest.

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
        converge_threshold: Debate early-stop threshold (included only for
            ``debate``). A converged run and a fixed-rounds run over otherwise
            identical inputs must not collide, so this is part of the key.

    Returns:
        A 64-char lowercase hex SHA-256 digest. Contains zero key material.
    """
    payload = build_identity(
        prompt=prompt,
        mode=mode,
        members=members,
        synthesizer=synthesizer,
        synthesizer_model_id=synthesizer_model_id,
        temperature=temperature,
        timeout=timeout,
        rounds=rounds,
        proposer=proposer,
        converge_threshold=converge_threshold,
        choices=choices,
        extract_verdict=extract_verdict,
        endpoint_urls=endpoint_urls,
        source_bundle_digest=source_bundle_digest,
        cache_format_version=cache_format_version,
        protocol_version=protocol_version,
        synthesis_prompt_version=synthesis_prompt_version,
        elite_prompt_version=elite_prompt_version,
        verdict_schema_version=verdict_schema_version,
        verdict_prompt_version=verdict_prompt_version,
    )
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
        if not isinstance(data, dict) or data.get("cache_format_version") != CACHE_FORMAT_VERSION:
            return None
        result = CouncilResult.model_validate(data.get("result"))
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
        # KEY-LEAK INVARIANT (audit vector 1): the cache only ever persists a
        # CouncilResult that has ALREADY passed through redaction upstream. Every
        # error string on the result (ModelAnswer.error, synthesis_error,
        # verdict_error) is scrubbed by redact() at the point of capture in
        # conclave.providers, BEFORE it is placed on the result and therefore long
        # before it reaches this write. Member/synthesis answer TEXT is provider
        # content, never key material. The cache KEY (make_key) is composed solely
        # of canonical secret-free identity and then SHA-256 hashed. Custom
        # endpoint URLs are sanitized and fingerprinted; no env var or key value
        # is read here. Net: no raw key or credential-bearing URL can reach a
        # cache file or filename. Do not move any un-redacted capture into the
        # result after this contract -- it would persist a secret to disk.
        result_payload = result.model_dump(mode="json")
        result_payload["cached"] = False
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "result": result_payload,
        }
        # Atomic-ish write: write to a temp sibling then replace, so a crash mid
        # write never leaves a half-written (corrupt) entry behind.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "cache write failed for key %s: %s; continuing without caching", key[:12], exc
        )
