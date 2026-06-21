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

**Structured output (CAC-02-GEM).** When an :class:`OutputContract` is supplied
*and* the static capability catalog marks the model
``supports_structured_output``, this adapter sets
``generationConfig.responseMimeType = "application/json"`` and
``generationConfig.responseSchema = <transformed schema>``. Gemini's
``responseSchema`` is an OpenAPI-3.0 *subset* (the ``v1beta`` ``Schema`` proto),
NOT full JSON Schema: it rejects ``additionalProperties`` (which the CAC-01
schema sets ``false`` on every object), ``title``/``$schema``/``$defs``/``$ref``,
and JSON-Schema union ``type`` arrays. :func:`_transform_schema_for_gemini`
recursively strips the unsupported keywords, uppercases ``type`` to the OpenAPI
enum (``OBJECT``/``ARRAY``/``STRING``/...), and maps a ``["string", "null"]``
nullable-union to a single ``type`` plus ``nullable: true``. A construct that
genuinely cannot be represented (a multi-type union or a composition keyword)
degrades to ``responseMimeType`` only (JSON without a strict schema) with a
non-fatal warning — an invalid schema is never sent, and the council never
aborts.
"""

from __future__ import annotations

import json
import warnings

from ..models import TokenUsage
from ..provider_catalog import capabilities_for
from ..registry import PROVIDER_ENV_VARS
from .base import OutputContract, ProviderError, SSEDelta, status_error

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# OpenAI role -> Gemini role. system is handled separately (hoisted).
_ROLE_MAP = {"user": "user", "assistant": "model"}

# JSON-Schema keywords the Gemini ``responseSchema`` (OpenAPI-3.0 subset)
# rejects; stripped at every node. ``additionalProperties`` is the load-bearing
# one (CAC-01 sets it ``false`` on every object); the rest are stripped
# defensively even though the CAC-01 schema should not emit them.
_GEMINI_STRIP_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "title",
        "$schema",
        "$id",
        "$comment",
        "$defs",
        "$ref",
        "definitions",
        "default",
        "examples",
        "patternProperties",
        "const",
    }
)

# Composition keywords with no representation in the OpenAPI-3.0 subset. Their
# presence forces the mimeType-only fallback (we never emit an invalid schema).
_GEMINI_UNSUPPORTED_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "not", "if", "then", "else"})

# JSON-Schema ``type`` string -> Gemini OpenAPI ``Type`` enum (uppercase).
_GEMINI_TYPE_MAP = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
}

# JSON-Schema keys whose VALUE is itself a schema and must be recursed into.
_SCHEMA_VALUED_KEYS = frozenset({"items", "additionalItems", "contains"})


class _UnrepresentableSchema(ValueError):
    """A schema construct that the Gemini OpenAPI subset cannot express.

    Raised internally by :func:`_transform_schema_for_gemini`; the adapter
    catches it and degrades to ``responseMimeType``-only JSON with a warning
    rather than sending an invalid ``responseSchema``.
    """


def _map_type(type_value: object) -> tuple[str, bool]:
    """Map a JSON-Schema ``type`` to a ``(gemini_type, nullable)`` pair.

    Handles the scalar form (``"string"``) and the CAC-01 nullable-union form
    (``["string", "null"]`` -> ``("STRING", True)``). A union with more than one
    non-null member has no single-``type`` OpenAPI representation and raises
    :class:`_UnrepresentableSchema`.

    Args:
        type_value: The raw ``type`` value: a string or a list of strings.

    Returns:
        ``(gemini_type, nullable)`` where ``gemini_type`` is the uppercase
        OpenAPI enum and ``nullable`` is ``True`` when ``"null"`` was in a union.

    Raises:
        _UnrepresentableSchema: On an unknown type name or a multi-type union.
    """
    if isinstance(type_value, str):
        mapped = _GEMINI_TYPE_MAP.get(type_value)
        if mapped is None:
            raise _UnrepresentableSchema(f"unsupported type {type_value!r}")
        return mapped, False
    if isinstance(type_value, list):
        non_null = [t for t in type_value if t != "null"]
        nullable = len(non_null) != len(type_value)
        if len(non_null) != 1:
            raise _UnrepresentableSchema(f"non-representable type union {type_value!r}")
        mapped = _GEMINI_TYPE_MAP.get(non_null[0])
        if mapped is None:
            raise _UnrepresentableSchema(f"unsupported type {non_null[0]!r}")
        return mapped, nullable
    raise _UnrepresentableSchema(f"unsupported type value {type_value!r}")


def _transform_schema_for_gemini(schema: dict) -> dict:
    """Transform a draft-style JSON Schema into a Gemini ``responseSchema`` dict.

    Recursively rebuilds ``schema`` (never mutating the input) into the
    OpenAPI-3.0 subset Gemini's ``generationConfig.responseSchema`` accepts:

    * **Strips** every keyword in :data:`_GEMINI_STRIP_KEYWORDS` — chiefly
      ``additionalProperties`` (CAC-01 sets it ``false`` everywhere), plus the
      ``title``/``$schema``/``$defs``/``$ref`` family.
    * **Maps** ``type`` to the uppercase OpenAPI ``Type`` enum and collapses a
      ``["string", "null"]`` union into a single ``type`` + ``nullable: true``.
    * **Recurses** into ``properties`` values, ``items`` (dict or list), and the
      other schema-valued keys.
    * **Carries through** ``enum``, ``required``, ``nullable``, ``format``,
      ``description``, ``minItems``/``maxItems``, and other scalar OpenAPI fields
      verbatim.

    Args:
        schema: A draft-style JSON Schema ``dict`` (e.g. the CAC-01
            :func:`conclave.verdict.verdict_json_schema` output).

    Returns:
        A fresh ``dict`` safe to place in ``generationConfig.responseSchema``.

    Raises:
        _UnrepresentableSchema: When the schema contains a construct with no
            OpenAPI-subset representation (a composition keyword such as
            ``anyOf``/``oneOf``, or a multi-type ``type`` union). The adapter
            catches this and falls back to mimeType-only JSON.
    """
    if not isinstance(schema, dict):
        raise _UnrepresentableSchema(f"schema node is not an object: {type(schema).__name__}")

    out: dict = {}
    union_nullable = False

    for key, value in schema.items():
        if key in _GEMINI_STRIP_KEYWORDS:
            continue
        if key in _GEMINI_UNSUPPORTED_KEYWORDS:
            raise _UnrepresentableSchema(f"unsupported composition keyword {key!r}")
        if key == "type":
            gemini_type, union_nullable = _map_type(value)
            out["type"] = gemini_type
        elif key == "properties":
            if not isinstance(value, dict):
                raise _UnrepresentableSchema("properties must be an object")
            out["properties"] = {
                prop: _transform_schema_for_gemini(sub) for prop, sub in value.items()
            }
        elif key in _SCHEMA_VALUED_KEYS:
            if isinstance(value, list):
                out[key] = [_transform_schema_for_gemini(sub) for sub in value]
            else:
                out[key] = _transform_schema_for_gemini(value)
        else:
            # enum / required / nullable / format / description / min*/max* and
            # other scalar OpenAPI-subset fields pass through unchanged.
            out[key] = value

    # A ``["string", "null"]`` union contributed nullability; merge it with any
    # explicit ``nullable`` already carried through (explicit True wins).
    if union_nullable:
        out["nullable"] = bool(out.get("nullable", False)) or True

    return out


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

    def _apply_output_contract(
        self,
        generation_config: dict,
        model_id: str,
        output_contract: OutputContract | None,
    ) -> None:
        """Inject structured-output keys into ``generation_config`` in place.

        Capability-gated and never-aborts. When ``output_contract`` is ``None``
        this is a no-op so the legacy request body is byte-for-byte unchanged.
        Otherwise, only when the static catalog marks the model
        ``supports_structured_output``, sets ``responseMimeType`` to
        ``"application/json"`` and ``responseSchema`` to the transformed schema.

        Degradation ladder (each step warns, none raises):

        * capability unknown / unsupported -> inject nothing (warn).
        * contract present but ``schema is None`` -> ``responseMimeType`` only
          (free-form JSON), no ``responseSchema`` (no warning — this is the
          caller asking for JSON mode without a schema).
        * schema present but unrepresentable in the OpenAPI subset ->
          ``responseMimeType`` only + warn (JSON without strict schema), never an
          invalid schema.

        Args:
            generation_config: The ``generationConfig`` dict to mutate in place.
            model_id: The provider-prefixed model id (capability lookup key).
            output_contract: The optional structured-output contract.
        """
        if output_contract is None:
            return

        caps = capabilities_for(model_id)
        if caps is None or not caps.supports_structured_output:
            warnings.warn(
                f"gemini: structured output requested for {model_id!r} but the "
                "capability catalog does not mark it supported; sending free prose.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Caller wants JSON; mimeType is safe even without a schema.
        generation_config["responseMimeType"] = "application/json"

        schema = output_contract.schema
        if schema is None:
            # JSON mode without a strict schema — intentional, not an error.
            return

        try:
            generation_config["responseSchema"] = _transform_schema_for_gemini(schema)
        except _UnrepresentableSchema as exc:
            # Never send an invalid schema; degrade to mimeType-only JSON.
            warnings.warn(
                f"gemini: output schema could not be mapped to responseSchema "
                f"({exc}); falling back to JSON without a strict schema.",
                UserWarning,
                stacklevel=2,
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
        """Build the generateContent POST.

        ``temperature`` is added to ``generationConfig`` only when not ``None``;
        passing ``None`` omits it so the model applies its own default. When an
        ``output_contract`` is supplied and the model is catalog-capable, the
        structured-output keys (``responseMimeType`` + transformed
        ``responseSchema``) are injected via :meth:`_apply_output_contract`; with
        ``output_contract is None`` the body is byte-for-byte the legacy shape.
        See :meth:`ProviderAdapter.build_request`.
        """
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
        # Conditional structured-output injection (no-op when contract is None,
        # so the legacy request body is preserved exactly).
        self._apply_output_contract(generation_config, model_id, output_contract)
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

        Same body as :meth:`build_request` (including any structured-output
        injection from an ``output_contract``), but the URL targets the streaming
        method with ``?alt=sse`` so Gemini emits standard SSE frames (without
        ``alt=sse`` it returns a single JSON array, not a stream -- verified
        against the Gemini API streaming reference). See
        :meth:`ProviderAdapter.stream_request`.
        """
        # output_contract flows into build_request, which performs the
        # capability-gated responseMimeType/responseSchema injection.
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
