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

import json

from ..models import TokenUsage
from ..registry import PROVIDER_ENV_VARS
from .base import OutputContract, ProviderError, SSEDelta, status_error

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
    supports_streaming = True

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
        temperature: float | None,
        timeout: float,
        api_key: str,
        output_contract: OutputContract | None = None,
    ) -> tuple[str, dict[str, str], dict]:
        """Build the Messages POST, hoisting system out of the message array.

        ``temperature`` is included only when not ``None``; passing ``None``
        omits it so the provider applies its own default (some models reject an
        explicit ``temperature``). See :meth:`ProviderAdapter.build_request`.
        """
        # output_contract: accepted; provider-native translation deferred to
        # CAC-02-ANT (Anthropic tool ``input_schema``). No-op today.
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
        }
        if temperature is not None:
            body["temperature"] = temperature
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        return self.completions_url, headers, body

    def parse_response(self, status: int, payload: object) -> tuple[str, TokenUsage | None]:
        """Concatenate ``content[*].text`` and map usage.

        See :meth:`ProviderAdapter.parse_response`.
        """
        if status < 200 or status >= 300:
            raise ProviderError(
                status_error("anthropic", status, payload, secondary_keys=("type",))
            )
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

    def stream_request(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        timeout: float,
        api_key: str,
        output_contract: OutputContract | None = None,
    ) -> tuple[str, dict[str, str], dict]:
        """Build the streaming POST: ``build_request`` + ``stream: true``.

        Anthropic streams the same ``/v1/messages`` endpoint with ``stream:
        true``; no other body change is needed. See
        :meth:`ProviderAdapter.stream_request`.
        """
        # output_contract: accepted; passed through to build_request (no-op
        # today; provider-native translation deferred to CAC-02-ANT).
        url, headers, body = self.build_request(
            model_id, messages, temperature, timeout, api_key, output_contract
        )
        body["stream"] = True
        return url, headers, body

    def parse_sse_event(self, event: str, data: str) -> SSEDelta:
        """Parse one Anthropic SSE frame by its named ``event`` type.

        Event flow handled (verified against the Anthropic Messages streaming
        reference):

        * ``message_start`` -- carries ``message.usage.input_tokens`` (the
          prompt accounting) -> a usage frame with ``prompt_tokens`` set.
        * ``content_block_delta`` with ``delta.type == "text_delta"`` ->
          ``delta.text`` is a text delta. (``input_json_delta`` /
          ``thinking_delta`` blocks carry no answer text and are skipped.)
        * ``message_delta`` -- carries the *cumulative* ``usage.output_tokens``
          -> a usage frame with ``completion_tokens`` set (last wins).
        * ``message_stop`` -> ``done=True``.
        * ``error`` -- a structured stream error -> :class:`ProviderError`.

        Other events (``content_block_start``/``stop``, ``ping``) yield an empty
        :class:`SSEDelta`. See :meth:`ProviderAdapter.parse_sse_event`.
        """
        if event == "message_stop":
            return SSEDelta(done=True)
        try:
            frame = json.loads(data)
        except (ValueError, TypeError) as exc:
            raise ProviderError(
                f"anthropic: malformed stream frame ({type(exc).__name__})"
            ) from exc
        if not isinstance(frame, dict):
            raise ProviderError("anthropic: malformed stream frame (non-object)")

        frame_type = frame.get("type")
        if frame_type == "error" or event == "error":
            raise ProviderError(status_error("anthropic", 200, frame, secondary_keys=("type",)))

        if frame_type == "content_block_delta":
            delta = frame.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str):
                    return SSEDelta(text=text)
            return SSEDelta()

        if frame_type == "message_start":
            message = frame.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                prompt = int(usage.get("input_tokens", 0) or 0)
                if prompt:
                    # Leave total at 0 so _merge_usage recomputes it from the
                    # merged prompt + completion (Anthropic never sends a sum).
                    return SSEDelta(usage=TokenUsage(prompt_tokens=prompt))
            return SSEDelta()

        if frame_type == "message_delta":
            usage = frame.get("usage")
            if isinstance(usage, dict):
                completion = int(usage.get("output_tokens", 0) or 0)
                if completion:
                    return SSEDelta(usage=TokenUsage(completion_tokens=completion))
            return SSEDelta()

        return SSEDelta()


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
