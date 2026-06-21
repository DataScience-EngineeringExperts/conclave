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

Structured output (CAC-02-OAI): when an :class:`OutputContract` is supplied, the
adapter translates it into the OpenAI ``response_format`` surface, capability-gated
via :func:`conclave.provider_catalog.capabilities_for`. The 7 OpenAI-compatible
providers are NOT identical (some do strict ``json_schema``, some only
``json_object``, some neither), so the adapter never hardcodes per-provider
branches -- it reads the capability flags and degrades to free prose (never raises)
for models that support neither. See :meth:`OpenAICompatAdapter._apply_output_contract`.
"""

from __future__ import annotations

import json

from ..logging import get_logger
from ..models import TokenUsage
from ..provider_catalog import capabilities_for
from .base import OutputContract, ProviderError, SSEDelta, status_error

logger = get_logger(__name__)

# Default name stamped into ``response_format.json_schema.name`` when an
# ``OutputContract`` carries no explicit ``schema_name``. OpenAI requires a
# non-empty name for a ``json_schema`` response format; "verdict" matches the
# council's primary structured-output use (the verdict/member schema).
_DEFAULT_SCHEMA_NAME = "verdict"

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

    def _apply_output_contract(
        self,
        body: dict,
        model_id: str,
        output_contract: OutputContract | None,
    ) -> None:
        """Inject ``response_format`` into ``body`` per the contract + capabilities.

        Capability-gated, in-place request shaping shared by ``build_request`` and
        ``stream_request`` (``response_format`` is documented as compatible with
        ``stream: true``, so the gating is identical for both paths). The body is
        mutated only when a contract is supplied *and* the model's capabilities
        permit a structured response; otherwise the body is left byte-for-byte
        unchanged so non-contract requests keep their exact pre-CAC-02 shape.

        The lookup uses the **full, provider-prefixed** ``model_id`` (e.g.
        ``"openai/gpt-4.1"``) because :func:`conclave.provider_catalog.capabilities_for`
        is keyed on the full id — this is why the contract is applied before
        :meth:`_bare_model` strips the prefix for the ``model`` body field.

        Capability tiers (most → least capable):

        * ``supports_structured_output`` → strict schema:
          ``{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}``.
        * else ``supports_json_mode`` → free-form JSON object:
          ``{"type": "json_object"}`` (a warning notes the schema is NOT enforced).
        * else / unknown caps → no injection (a non-fatal warning is logged).

        Never raises: an unsupported model degrades to the current free-prose
        behavior rather than aborting the council (Scope Plan §5).

        Args:
            body: The request body being assembled; mutated in place.
            model_id: The FULL provider-prefixed model id used for the capability
                lookup (NOT the bare wire id).
            output_contract: The structured-output contract, or ``None`` to skip
                injection entirely.
        """
        if output_contract is None:
            return

        caps = capabilities_for(model_id)
        if caps is None:
            # Unknown / custom endpoint: we cannot assert support, so we do not
            # shape the request. Free-prose answer; caller (CAC-05) validates.
            logger.warning(
                "%s: no capability record for %s; structured output not requested "
                "(free-prose answer)",
                self.prefix,
                model_id,
            )
            return

        if caps.supports_structured_output:
            name = output_contract.schema_name or _DEFAULT_SCHEMA_NAME
            json_schema: dict = {"name": name, "strict": output_contract.strict}
            # Omit a None schema rather than send ``"schema": null`` — a contract
            # may set strict/name without a concrete schema dict.
            if output_contract.schema is not None:
                json_schema["schema"] = output_contract.schema
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        elif caps.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
            logger.warning(
                "%s: %s supports JSON mode only, not strict json_schema; "
                "requesting json_object (schema NOT enforced by the provider)",
                self.prefix,
                model_id,
            )
        else:
            logger.warning(
                "%s: %s supports neither structured output nor JSON mode; "
                "structured output not requested (free-prose answer)",
                self.prefix,
                model_id,
            )

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
        # Capability-gated ``response_format`` injection. Passes the FULL
        # provider-prefixed ``model_id`` (NOT the bare wire id already stored in
        # body["model"]) because the catalog is keyed on the full id. No-op when
        # ``output_contract is None`` -> body stays byte-for-byte unchanged.
        self._apply_output_contract(body, model_id, output_contract)
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
        # output_contract is passed through to build_request, which applies the
        # capability-gated ``response_format`` shaping (compatible with
        # stream:true). Stream flags are layered on top of the shaped body.
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
