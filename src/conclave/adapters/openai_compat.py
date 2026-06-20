"""OpenAI-compatible chat-completions adapter.

The widest-reach adapter: any provider exposing the OpenAI ``/chat/completions``
shape (``{model, messages, temperature}`` in, ``choices[0].message.content`` out)
is served by a single :class:`OpenAICompatAdapter` instance, parameterized by its
full completions URL and env var name(s). conclave ships instances for **openai**,
**xai**, **perplexity**, **groq**, **deepseek**, **mistral**, and **together**
(all direct vendor key -> direct vendor endpoint); the same class powers any
user-supplied OpenAI-compatible endpoint declared in config.

Per-provider full URLs live in :data:`OPENAI_COMPAT_URLS` so the verified
endpoints sit in one place. Env-var names are sourced from
:data:`conclave.registry.PROVIDER_ENV_VARS` -- never duplicated here.
"""

from __future__ import annotations

import json

from ..models import TokenUsage
from .base import OutputContract, ProviderError, SSEDelta, status_error

# Verified per-provider full completions URLs. Note Perplexity has NO ``/v1``
# segment while xAI/OpenAI do, and Groq nests its OpenAI surface under
# ``/openai/v1``. These mirror :data:`conclave.registry.OPENAI_COMPAT_PROVIDERS`
# (the source of truth) -- the import-time drift guard fails loudly if they
# desync. Every entry is a direct vendor endpoint (no aggregator/router).
OPENAI_COMPAT_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "perplexity": "https://api.perplexity.ai/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
}


class OpenAICompatAdapter:
    """Adapter for OpenAI-style ``/chat/completions`` endpoints.

    Args:
        prefix: Provider prefix this instance serves (e.g. ``"xai"``); matches
            :func:`conclave.registry.provider_prefix`.
        completions_url: Full POST URL for the chat-completions endpoint.
        env_vars: Candidate env var names; the first present is the active key.
        max_tokens: Optional ``max_tokens`` cap. When ``None`` (default) the
            parameter is omitted so the provider applies its own default.
    """

    # Every OpenAI-compatible vendor conclave ships speaks the standard
    # streaming protocol (``stream: true`` -> SSE deltas -> ``[DONE]``).
    supports_streaming = True

    def __init__(
        self,
        prefix: str,
        completions_url: str,
        env_vars: tuple[str, ...],
        max_tokens: int | None = None,
    ) -> None:
        self.prefix = prefix
        self.completions_url = completions_url
        self.env_vars = env_vars
        self.max_tokens = max_tokens

    def _bare_model(self, model_id: str) -> str:
        """Strip the provider prefix to get the id the API expects.

        OpenAI-compatible providers want the bare model name (``"grok-4.3"``),
        not the conclave-internal ``"xai/grok-4.3"`` form.
        """
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
        """Build the OpenAI-style POST.

        ``temperature`` is included only when not ``None``; passing ``None``
        omits it so the provider applies its own default (some reasoning models
        reject an explicit ``temperature`` with a 400). See
        :meth:`ProviderAdapter.build_request`.
        """
        # output_contract: accepted; provider-native translation deferred to
        # CAC-02-OAI (OpenAI ``response_format``). No-op today.
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: dict = {
            "model": self._bare_model(model_id),
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        return self.completions_url, headers, body

    def parse_response(self, status: int, payload: object) -> tuple[str, TokenUsage | None]:
        """Parse ``choices[0].message.content`` + ``usage``.

        See :meth:`ProviderAdapter.parse_response`.
        """
        if status < 200 or status >= 300:
            raise ProviderError(
                status_error(self.prefix, status, payload, secondary_keys=("type",))
            )
        if not isinstance(payload, dict):
            raise ProviderError(f"{self.prefix}: non-JSON response body (status {status})")

        try:
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.prefix}: malformed response, missing "
                f"choices[0].message.content ({type(exc).__name__})"
            ) from exc

        if not content:
            raise ProviderError(f"{self.prefix}: empty response (no message content)")

        usage = _parse_usage(payload.get("usage"))
        return content, usage

    def stream_request(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        timeout: float,
        api_key: str,
        output_contract: OutputContract | None = None,
    ) -> tuple[str, dict[str, str], dict]:
        """Build the streaming POST: ``build_request`` + ``stream`` flags.

        Sets ``stream: true`` and ``stream_options.include_usage: true`` so the
        provider emits incremental ``choices[0].delta.content`` chunks followed
        by a final chunk with empty ``choices`` and a top-level ``usage`` object
        (verified against the OpenAI chat-completions streaming reference). See
        :meth:`ProviderAdapter.stream_request`.
        """
        # output_contract: accepted; passed through to build_request (no-op
        # today; provider-native translation deferred to CAC-02-OAI).
        url, headers, body = self.build_request(
            model_id, messages, temperature, timeout, api_key, output_contract
        )
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        return url, headers, body

    def parse_sse_event(self, event: str, data: str) -> SSEDelta:
        """Parse one OpenAI-style SSE frame.

        Frame shapes handled (verified against the OpenAI chat-completions
        streaming reference):

        * ``[DONE]`` -- the terminating sentinel -> ``done=True``.
        * a chunk with ``choices[0].delta.content`` -> a text delta.
        * the final ``include_usage`` chunk: ``choices == []`` and a top-level
          ``usage`` object -> a usage frame.

        A frame whose JSON is malformed raises :class:`ProviderError`; a frame
        that simply carries no content (role-only delta, ``finish_reason`` only)
        yields an empty :class:`SSEDelta`. See
        :meth:`ProviderAdapter.parse_sse_event`.
        """
        if data == "[DONE]":
            return SSEDelta(done=True)
        try:
            chunk = json.loads(data)
        except (ValueError, TypeError) as exc:
            raise ProviderError(
                f"{self.prefix}: malformed stream frame ({type(exc).__name__})"
            ) from exc
        if not isinstance(chunk, dict):
            raise ProviderError(f"{self.prefix}: malformed stream frame (non-object)")

        # A structured error can arrive mid-stream as a normal data frame.
        if isinstance(chunk.get("error"), (dict, str)):
            raise ProviderError(status_error(self.prefix, 200, chunk, secondary_keys=("type",)))

        text = ""
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    text = content

        usage = _parse_usage(chunk.get("usage"))
        return SSEDelta(text=text, usage=usage)


def _parse_usage(raw: object) -> TokenUsage | None:
    """Map an OpenAI-style ``usage`` block to :class:`TokenUsage`, or ``None``."""
    if not isinstance(raw, dict):
        return None
    return TokenUsage(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        total_tokens=int(raw.get("total_tokens", 0) or 0),
    )
