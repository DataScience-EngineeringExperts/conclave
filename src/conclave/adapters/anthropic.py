"""Anthropic Messages API adapter (native, not OpenAI-compatible).

Anthropic's ``/v1/messages`` differs from the OpenAI shape in three ways that
this adapter handles:

* **Auth header** is ``x-api-key`` (plus a required ``anthropic-version``), not
  ``Authorization: Bearer``.
* **System prompt is top-level.** Any OpenAI-style ``{"role": "system"}`` message
  is hoisted out of the array into the body's ``system`` field; only user/
  assistant turns remain in ``messages``.
* **``max_tokens`` is required.** It defaults to 4096 and is configurable.

Response text is the concatenation of every ``content[*].text`` block whose
``type == "text"``; usage maps ``input_tokens``/``output_tokens``.
"""

from __future__ import annotations

from ..models import TokenUsage
from ..registry import PROVIDER_ENV_VARS
from .base import ProviderError

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


class AnthropicAdapter:
    """Adapter for Anthropic's native Messages API.

    Args:
        max_tokens: Required-by-API generation cap. Defaults to 4096.
    """

    prefix = "anthropic"
    completions_url = ANTHROPIC_URL

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.max_tokens = max_tokens
        self.env_vars = tuple(PROVIDER_ENV_VARS["anthropic"])

    def _bare_model(self, model_id: str) -> str:
        """Strip the ``anthropic/`` prefix to the bare Anthropic model name."""
        return model_id.split("/", 1)[1] if "/" in model_id else model_id

    def build_request(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        timeout: float,
        api_key: str,
    ) -> tuple[str, dict[str, str], dict]:
        """Build the Messages POST, hoisting system out of the message array.

        See :meth:`ProviderAdapter.build_request`.
        """
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        system_parts: list[str] = []
        turns: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if content:
                    system_parts.append(content)
            elif role in ("user", "assistant"):
                turns.append({"role": role, "content": content})
            else:  # unknown role -> treat as user content so nothing is dropped
                turns.append({"role": "user", "content": content})

        body: dict = {
            "model": self._bare_model(model_id),
            "max_tokens": self.max_tokens,
            "messages": turns,
            "temperature": temperature,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        return self.completions_url, headers, body

    def parse_response(self, status: int, payload: object) -> tuple[str, TokenUsage | None]:
        """Concatenate ``content[*].text`` and map usage.

        See :meth:`ProviderAdapter.parse_response`.
        """
        if status < 200 or status >= 300:
            raise ProviderError(_status_error(status, payload))
        if not isinstance(payload, dict):
            raise ProviderError(f"anthropic: non-JSON response body (status {status})")

        content = payload.get("content")
        if not isinstance(content, list):
            raise ProviderError("anthropic: malformed response, missing content array")
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise ProviderError("anthropic: empty response (no text content)")

        usage = _parse_usage(payload.get("usage"))
        return text, usage


def _parse_usage(raw: object) -> TokenUsage | None:
    """Map Anthropic ``input_tokens``/``output_tokens`` to :class:`TokenUsage`."""
    if not isinstance(raw, dict):
        return None
    prompt = int(raw.get("input_tokens", 0) or 0)
    completion = int(raw.get("output_tokens", 0) or 0)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _status_error(status: int, payload: object) -> str:
    """Build a concise, redact-safe error message for a non-2xx status."""
    detail = ""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("type") or "")
        elif isinstance(err, str):
            detail = err
    elif isinstance(payload, str):
        detail = payload[:200]
    suffix = f": {detail}" if detail else ""
    return f"anthropic: HTTP {status}{suffix}"
