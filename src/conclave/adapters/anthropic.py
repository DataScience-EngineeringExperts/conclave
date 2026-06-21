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

**Structured output (CAC-02-ANT).** Anthropic has no OpenAI-style
``response_format``; the documented way to constrain output to a JSON Schema is
*forced tool use* — declare a single tool whose ``input_schema`` is the contract
and pin ``tool_choice`` to it. The model then returns the structured object as
the ``input`` of a ``content[*]`` block whose ``type == "tool_use"`` (NOT as
``text``). This adapter injects that tool only when an :class:`OutputContract`
is present AND the static catalog reports ``supports_structured_output`` for the
model; otherwise it logs a non-fatal warning and falls back to free prose. It
never raises on an unsupported model. See :meth:`AnthropicAdapter.build_request`
and :meth:`AnthropicAdapter.parse_response`.
"""

from __future__ import annotations

import json

from ..logging import get_logger
from ..models import TokenUsage
from ..provider_catalog import capabilities_for
from ..registry import PROVIDER_ENV_VARS
from .base import OutputContract, ProviderError, SSEDelta, status_error

logger = get_logger("adapters.anthropic")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
# Tool name used when an OutputContract supplies no schema_name. The council's
# structured member/verdict output is the only forced-tool use today.
DEFAULT_TOOL_NAME = "verdict"


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
        # Name of the tool forced by the most recent structured-output
        # build_request, or None when no tool was forced. parse_response reads
        # this to know whether to extract a tool_use block instead of text.
        # Mirrors the existing per-instance state pattern (self.max_tokens); the
        # adapter is resolved per call so this never leaks across councils.
        self._forced_tool_name: str | None = None

    def _bare_model(self, model_id: str) -> str:
        """Strip the ``anthropic/`` prefix to the bare Anthropic model name."""
        return model_id.split("/", 1)[1] if "/" in model_id else model_id

    def _tool_for_contract(
        self, model_id: str, output_contract: OutputContract | None
    ) -> dict | None:
        """Translate an :class:`OutputContract` into a forced-tool spec, or None.

        Anthropic constrains output to a JSON Schema via forced tool use, not a
        ``response_format``. We inject the tool ONLY when a contract is present,
        it carries a ``schema``, and the static catalog reports
        ``supports_structured_output`` for this model. Any other case (no
        contract, no schema, unsupported/unknown capability) returns ``None`` and
        — for the supported-but-declined cases — logs a non-fatal warning. This
        method never raises: an unparseable capability lookup degrades to free
        prose rather than aborting the member call.

        Returns:
            A ``{"name", "description", "input_schema"}`` tool dict to inject, or
            ``None`` to leave the request as free-prose.
        """
        if output_contract is None:
            return None
        if not isinstance(output_contract.schema, dict):
            # A contract with no schema cannot constrain anything; nothing to do.
            return None

        caps = capabilities_for(model_id)
        if caps is None or not caps.supports_structured_output:
            # Capability unknown or explicitly unsupported -> do NOT inject the
            # tool; degrade to free prose. Non-fatal: the council still runs.
            logger.warning(
                "anthropic: structured output requested for %s but capability is "
                "unsupported/unknown; sending free-prose request",
                model_id,
            )
            return None

        name = output_contract.schema_name or DEFAULT_TOOL_NAME
        return {
            "name": name,
            "description": "Return the result as structured data.",
            "input_schema": output_contract.schema,
        }

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

        **Structured output.** When ``output_contract`` is present and the model
        is catalog-capable, a single tool whose ``input_schema`` is the contract
        schema is appended and ``tool_choice`` is pinned to it, forcing the model
        to return the schema-shaped object as a ``tool_use`` block.
        :meth:`parse_response` then extracts that block. The system-hoist and
        required ``max_tokens`` are unchanged in both paths. With no contract (or
        an unsupported model) the body is byte-for-byte identical to the legacy
        free-prose request.
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
        }
        if temperature is not None:
            body["temperature"] = temperature
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        # Forced tool use is the Anthropic-native structured-output surface. Only
        # injected when a contract is present and the model is catalog-capable;
        # otherwise the body above is the unchanged free-prose request.
        tool = self._tool_for_contract(model_id, output_contract)
        if tool is not None:
            body["tools"] = [tool]
            body["tool_choice"] = {"type": "tool", "name": tool["name"]}
            self._forced_tool_name = tool["name"]
        else:
            # Clear any stale flag from a prior build on this instance so a later
            # free-prose call never mistakenly parses a tool_use block.
            self._forced_tool_name = None
        return self.completions_url, headers, body

    def parse_response(self, status: int, payload: object) -> tuple[str, TokenUsage | None]:
        """Parse a Messages response into ``(text, usage)``.

        Two shapes, selected by whether the most recent :meth:`build_request`
        forced a tool (``self._forced_tool_name``):

        * **Structured (tool forced).** A forced tool returns the schema-shaped
          object as the ``input`` of a ``content[*]`` block whose
          ``type == "tool_use"`` — NOT as ``text``. The matching block's
          ``input`` is serialized to a JSON string so the existing
          ``ModelAnswer.answer: str`` contract holds. The block whose ``name``
          equals the forced tool is preferred; the first ``tool_use`` block is
          the fallback (a forced tool yields exactly one).
        * **Free prose (no tool).** Unchanged: concatenate every ``text`` block.

        Usage parsing (``input_tokens``/``output_tokens``) is identical in both
        paths. See :meth:`ProviderAdapter.parse_response`.
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

        usage = _parse_usage(payload.get("usage"))

        if self._forced_tool_name is not None:
            text = self._extract_tool_input(content, self._forced_tool_name)
            return text, usage

        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise ProviderError("anthropic: empty response (no text content)")
        return text, usage

    def _extract_tool_input(self, content: list, tool_name: str) -> str:
        """Serialize the forced ``tool_use`` block's ``input`` to a JSON string.

        Scans ``content`` for ``tool_use`` blocks, preferring the one whose
        ``name`` matches the forced tool and falling back to the first tool_use
        block found. The block's ``input`` (always an object per the Messages
        spec) is dumped with sorted keys so the structured answer is stable.

        Raises:
            ProviderError: When no ``tool_use`` block is present (the model
                ignored the forced tool) or its ``input`` is not a JSON object.
        """
        chosen: dict | None = None
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") == tool_name:
                chosen = block
                break
            if chosen is None:  # first tool_use as fallback
                chosen = block

        if chosen is None:
            raise ProviderError("anthropic: empty response (no tool_use content)")

        tool_input = chosen.get("input")
        if not isinstance(tool_input, dict):
            raise ProviderError("anthropic: malformed tool_use block (input is not an object)")
        return json.dumps(tool_input, sort_keys=True)

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
        true``; no other body change is needed. ``output_contract`` is passed
        through to :meth:`build_request`, so a forced tool is injected here too;
        :meth:`parse_sse_event` accumulates the resulting ``input_json_delta``
        fragments. See :meth:`ProviderAdapter.stream_request`.
        """
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
          ``delta.text`` is a text delta.
        * ``content_block_delta`` with ``delta.type == "input_json_delta"`` ->
          ``delta.partial_json`` is a fragment of a forced tool's ``input``
          object; it is surfaced as a text delta so the buffered stream
          consumer reconstructs the full JSON string (the same string
          :meth:`parse_response` would return for the structured case). These
          frames only arrive when a tool was forced, so the no-contract stream
          is unaffected. (``thinking_delta`` blocks carry no answer text and are
          skipped.)
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
            if isinstance(delta, dict):
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str):
                        return SSEDelta(text=text)
                elif delta_type == "input_json_delta":
                    # Forced-tool structured output: accumulate the partial JSON
                    # of the tool's input as text so the buffered consumer
                    # reassembles the full object. Only emitted when a tool was
                    # forced, so the no-contract path is untouched.
                    partial = delta.get("partial_json")
                    if isinstance(partial, str) and partial:
                        return SSEDelta(text=partial)
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
