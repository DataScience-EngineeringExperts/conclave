"""Google Gemini ``generateContent`` adapter (native v1beta).

Gemini's wire format diverges from OpenAI in several ways this adapter handles:

* **Model in the URL.** The bare model name (``gemini/`` prefix stripped) goes
  into the path: ``/v1beta/models/{model}:generateContent``.
* **Auth header** is ``x-goog-api-key`` (no ``Bearer``).
* **Roles** map ``assistant`` -> ``model`` and ``user`` -> ``user``; each turn
  becomes ``{"role", "parts": [{"text": ...}]}``.
* **System prompt is top-level** ``systemInstruction``, hoisted out of the array.
* **Generation params** live under ``generationConfig`` as ``temperature`` and
  ``maxOutputTokens`` (default 4096, configurable).

Response text is the concatenation of ``candidates[0].content.parts[*].text``;
usage maps ``usageMetadata.promptTokenCount``/``candidatesTokenCount``/
``totalTokenCount``.
"""

from __future__ import annotations

import json

from ..models import TokenUsage
from ..registry import PROVIDER_ENV_VARS
from .base import OutputContract, ProviderError, SSEDelta, status_error

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# OpenAI role -> Gemini role. system is handled separately (hoisted).
_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiAdapter:
    """Adapter for Google's Gemini ``generateContent`` endpoint.

    Args:
        max_output_tokens: ``generationConfig.maxOutputTokens``. Defaults to 4096.
    """

    prefix = "gemini"
    # The concrete URL embeds the model and is built per-request; this base is
    # exposed for parity with the protocol's ``completions_url`` attribute.
    completions_url = GEMINI_BASE
    supports_streaming = True

    def __init__(self, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> None:
        self.max_output_tokens = max_output_tokens
        self.env_vars = tuple(PROVIDER_ENV_VARS["gemini"])

    def _bare_model(self, model_id: str) -> str:
        """Strip the ``gemini/`` prefix to the bare model name for the URL path."""
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
        """Build the generateContent POST.

        ``temperature`` is added to ``generationConfig`` only when not ``None``;
        passing ``None`` omits it so the model applies its own default. See
        :meth:`ProviderAdapter.build_request`.
        """
        # output_contract: accepted; provider-native translation deferred to
        # CAC-02-GEM (Gemini ``generationConfig.responseSchema``). No-op today.
        model = self._bare_model(model_id)
        url = f"{GEMINI_BASE}/{model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        system_parts: list[str] = []
        contents: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            gemini_role = _ROLE_MAP.get(role, "user")
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

        generation_config: dict = {"maxOutputTokens": self.max_output_tokens}
        if temperature is not None:
            generation_config["temperature"] = temperature
        body: dict = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return url, headers, body

    def parse_response(self, status: int, payload: object) -> tuple[str, TokenUsage | None]:
        """Concatenate the first candidate's text parts. See base protocol."""
        if status < 200 or status >= 300:
            raise ProviderError(status_error("gemini", status, payload, secondary_keys=("status",)))
        if not isinstance(payload, dict):
            raise ProviderError(f"gemini: non-JSON response body (status {status})")

        try:
            candidate = payload["candidates"][0]
            parts = candidate["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"gemini: malformed response, missing "
                f"candidates[0].content.parts ({type(exc).__name__})"
            ) from exc

        text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part
        )
        if not text:
            raise ProviderError("gemini: empty response (no text parts)")

        usage = _parse_usage(payload.get("usageMetadata"))
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
        """Build the streaming POST against ``streamGenerateContent?alt=sse``.

        Same body as :meth:`build_request`, but the URL targets the streaming
        method with ``?alt=sse`` so Gemini emits standard SSE frames (without
        ``alt=sse`` it returns a single JSON array, not a stream -- verified
        against the Gemini API streaming reference). See
        :meth:`ProviderAdapter.stream_request`.
        """
        # output_contract: accepted; passed through to build_request (no-op
        # today; provider-native translation deferred to CAC-02-GEM).
        _url, headers, body = self.build_request(
            model_id, messages, temperature, timeout, api_key, output_contract
        )
        model = self._bare_model(model_id)
        url = f"{GEMINI_BASE}/{model}:streamGenerateContent?alt=sse"
        return url, headers, body

    def parse_sse_event(self, event: str, data: str) -> SSEDelta:
        """Parse one Gemini SSE frame (a partial ``GenerateContentResponse``).

        Each frame carries ``candidates[0].content.parts[*].text`` (a text
        delta) and may carry a *cumulative* ``usageMetadata`` accounting (last
        wins). Gemini has no ``[DONE]`` sentinel -- the stream simply ends -- so
        no frame sets ``done``; the transport's end-of-iteration terminates the
        loop. A frame whose JSON is malformed raises :class:`ProviderError`; a
        frame carrying a structured ``error`` likewise raises. A safety-blocked
        or otherwise text-less candidate yields a usage-only / empty delta. See
        :meth:`ProviderAdapter.parse_sse_event`.
        """
        try:
            frame = json.loads(data)
        except (ValueError, TypeError) as exc:
            raise ProviderError(f"gemini: malformed stream frame ({type(exc).__name__})") from exc
        if not isinstance(frame, dict):
            raise ProviderError("gemini: malformed stream frame (non-object)")

        if isinstance(frame.get("error"), (dict, str)):
            raise ProviderError(status_error("gemini", 200, frame, secondary_keys=("status",)))

        text = ""
        candidates = frame.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            content = candidate.get("content") if isinstance(candidate, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list):
                text = "".join(
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict) and "text" in part
                )

        usage = _parse_usage(frame.get("usageMetadata"))
        return SSEDelta(text=text, usage=usage)


def _parse_usage(raw: object) -> TokenUsage | None:
    """Map Gemini ``usageMetadata`` counts to :class:`TokenUsage`."""
    if not isinstance(raw, dict):
        return None
    return TokenUsage(
        prompt_tokens=int(raw.get("promptTokenCount", 0) or 0),
        completion_tokens=int(raw.get("candidatesTokenCount", 0) or 0),
        total_tokens=int(raw.get("totalTokenCount", 0) or 0),
    )
