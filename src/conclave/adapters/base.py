"""Adapter contract: the per-provider request/response translation layer.

An adapter is the *only* place that knows a provider's wire format. It builds the
HTTP request (URL, headers, JSON body) from conclave's OpenAI-style message list
and parses the provider's response back into ``(text, TokenUsage | None)``.

Adapters never perform I/O themselves -- they hand the built request to
:func:`conclave.transport.post_json`. This keeps the network boundary single and
keeps adapters trivially unit-testable (``build_request`` / ``parse_response``
are pure functions of their inputs).

Two cross-cutting concerns live here:

* :class:`ProviderError` -- a normalized error type for non-2xx responses or
  malformed payloads. Its message is ALREADY scrubbed via :func:`redact` so it
  is safe to surface in ``ModelAnswer.error``.
* :func:`redact` -- key-leak hardening. Strips bearer tokens, ``sk-`` style
  keys, ``x-api-key`` echoes, and any value of the env vars we hold names for --
  built-in providers AND custom-endpoint ``api_key_env`` names declared in
  config -- before an error string can ever escape the call path.
"""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

from ..models import TokenUsage
from ..registry import PROVIDER_ENV_VARS

# Matches "Bearer sk-abc123" / "Bearer xai-..." auth headers echoed into errors.
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)
# Matches standalone provider-style keys: sk-..., xai-..., pplx-..., AIza... etc.
_KEY_LIKE_RE = re.compile(r"\b(?:sk|xai|pplx|AIza)[A-Za-z0-9._\-]{8,}\b")
# Matches an x-api-key / x-goog-api-key header echoed with its value.
_HEADER_KEY_RE = re.compile(r"(x-(?:goog-)?api-key)\s*[:=]\s*[A-Za-z0-9._\-]+", re.IGNORECASE)

_REDACTED = "[REDACTED]"

# Upper bound on the error-detail substring extracted from a provider body. A
# provider can return an arbitrarily large error payload (a multi-KB
# ``error.message`` has been observed); without a cap that whole blob lands in
# ``ModelAnswer.error``, logs, and ``--json`` output, and amplifies any leak the
# redactor then has to scrub. Bounded here so error strings stay readable.
_DETAIL_CAP = 500


def status_error(
    prefix: str,
    status: int,
    payload: object,
    *,
    secondary_keys: tuple[str, ...] = (),
) -> str:
    """Build a concise, redact-safe ``"{prefix}: HTTP {status}[: detail]"`` string.

    Shared by every concrete adapter so the non-2xx error format -- and the
    detail length cap -- live in exactly one place. The extracted ``detail`` is
    always truncated to :data:`_DETAIL_CAP` characters regardless of whether it
    came from a dict ``error.message``, a secondary key, a string ``error``, a
    top-level ``message``, or a raw string body. The returned message is NOT
    redacted here; callers wrap it in :class:`ProviderError`, which redacts on
    construction.

    Args:
        prefix: Provider label that opens the message (e.g. ``"anthropic"``).
        status: HTTP status code returned by the transport.
        payload: Decoded JSON object (dict), a raw string body, or anything else.
        secondary_keys: Fallback keys read from a dict ``error`` object when
            ``error.message`` is absent -- e.g. ``("type",)`` for Anthropic/OpenAI
            or ``("status",)`` for Gemini. Tried in order.

    Returns:
        A bounded, single-line error string safe to pass to ``ProviderError``.
    """
    detail = ""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or "")
            if not detail:
                for key in secondary_keys:
                    value = err.get(key)
                    if value:
                        detail = str(value)
                        break
        elif isinstance(err, str):
            detail = err
        # Some OpenAI-compatible providers put the message at the top level.
        if not detail and "message" in payload:
            detail = str(payload["message"])
    elif isinstance(payload, str):
        detail = payload

    detail = detail[:_DETAIL_CAP]
    suffix = f": {detail}" if detail else ""
    return f"{prefix}: HTTP {status}{suffix}"


def _custom_endpoint_env_vars() -> list[str]:
    """Return the env-var NAMES declared by custom endpoints, or [] on any error.

    Custom OpenAI-compatible endpoints (``config.endpoints[*].env_var``) name a
    key var that is NOT in :data:`PROVIDER_ENV_VARS`, so :func:`redact` would not
    otherwise know to mask its value -- the BYO-keys leak class. We load config
    here (lazily, to avoid an import cycle) purely to learn those names; failures
    must never break redaction, so any error yields an empty list and we fall
    back to pattern-based scrubbing only.
    """
    try:
        from ..config import load_config

        return [ep.env_var for ep in load_config().endpoints.values() if ep.env_var]
    except Exception:  # noqa: BLE001 -- redaction must never raise
        return []


def redact(text: str) -> str:
    """Scrub anything key-shaped from a string before it can be surfaced.

    Removes, in order: any live value of an env var we know a name for --
    including custom-endpoint ``api_key_env`` names declared in config --
    ``Bearer <token>`` auth headers, ``x-api-key``/``x-goog-api-key`` header
    echoes, and standalone provider-style key tokens (``sk-``, ``xai-``,
    ``pplx-``, ``AIza...``). Idempotent and safe on already-clean text.

    Args:
        text: An error or diagnostic string that may have captured a secret.

    Returns:
        The same text with every recognizable secret replaced by ``[REDACTED]``.
    """
    if not text:
        return text
    cleaned = text
    # 1) Redact concrete env-var values first (most authoritative). This covers
    # the built-in providers AND any custom-endpoint key var declared in config,
    # so a BYO custom key with an unrecognized shape is still masked. We only
    # read values here to mask them; the masked result never contains the value.
    builtin_names = [name for names in PROVIDER_ENV_VARS.values() for name in names]
    for name in builtin_names + _custom_endpoint_env_vars():
        value = os.environ.get(name, "").strip()
        if value:
            cleaned = cleaned.replace(value, _REDACTED)
    # 2) Header-shaped echoes.
    cleaned = _HEADER_KEY_RE.sub(rf"\1: {_REDACTED}", cleaned)
    # 3) Bearer auth headers.
    cleaned = _BEARER_RE.sub(f"Bearer {_REDACTED}", cleaned)
    # 4) Standalone key-like tokens.
    cleaned = _KEY_LIKE_RE.sub(_REDACTED, cleaned)
    return cleaned


class ProviderError(Exception):
    """A provider-side failure: non-2xx status or a malformed/empty payload.

    The message passed in is redacted on construction, so the stored message is
    always safe to place in ``ModelAnswer.error`` and to log.
    """

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))


@runtime_checkable
class ProviderAdapter(Protocol):
    """The contract every concrete provider adapter satisfies.

    Identity attributes let the registry map a model id to an adapter and let the
    provider call path locate the right env var without re-deriving the mapping:

    * ``prefix`` -- matches :func:`conclave.registry.provider_prefix(model_id)`.
    * ``env_vars`` -- candidate env var names (first present is the active key).
    * ``completions_url`` -- the endpoint the request is POSTed to (may embed the
      model name, e.g. Gemini).
    """

    prefix: str
    env_vars: tuple[str, ...]
    completions_url: str

    def build_request(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        timeout: float,
        api_key: str,
    ) -> tuple[str, dict[str, str], dict]:
        """Build ``(url, headers, json_body)`` for this provider.

        Args:
            model_id: Friendly-resolved model id (e.g. ``"xai/grok-4.3"``).
            messages: OpenAI-style message list (roles system/user/assistant).
            temperature: Sampling temperature, or ``None`` to omit the parameter
                entirely so the provider applies its own default (some models
                reject an explicit ``temperature``).
            timeout: Per-call timeout in seconds (informational for body params).
            api_key: The resolved key VALUE, read at call time and never stored.

        Returns:
            A ``(url, headers, json_body)`` tuple ready for ``post_json``.
        """
        ...

    def parse_response(self, status: int, payload: object) -> tuple[str, TokenUsage | None]:
        """Parse a provider response into ``(text, usage)``.

        Args:
            status: HTTP status code returned by the transport.
            payload: Decoded JSON object (or raw text on non-JSON responses).

        Returns:
            A ``(text, usage)`` tuple on success.

        Raises:
            ProviderError: On non-2xx status or a malformed/empty payload, with a
                message already scrubbed of secrets.
        """
        ...
