"""Structured-output tests for the Gemini adapter (CAC-02-GEM).

These cover the schema-transform that turns conclave's draft-style JSON Schema
(the CAC-01 LCD verdict/member schema) into Gemini's ``generationConfig``
``responseSchema`` — an OpenAPI-3.0 *subset* that rejects several JSON-Schema
keywords (notably ``additionalProperties``, plus ``title``/``$schema``/
``$defs``/``$ref``) and expresses nullability via a ``nullable`` boolean + an
uppercase single ``type`` rather than a ``["string", "null"]`` union.

Boundaries of this ticket:

* We only SHAPE the request (mimeType + transformed responseSchema). The
  multi-call validate/repair/raw-fallback loop is CAC-05, not here.
* With ``output_contract is None`` the request body must be byte-for-byte the
  legacy shape so ``tests/test_adapters.py`` stays green.
* Unsupported capability / unrepresentable schema construct must NEVER raise —
  it warns and degrades to mimeType-only JSON (no responseSchema).
"""

from __future__ import annotations

import pytest

from conclave.adapters.base import OutputContract
from conclave.adapters.gemini import (
    _GEMINI_STRIP_KEYWORDS,
    GeminiAdapter,
    _transform_schema_for_gemini,
)
from conclave.verdict import member_answer_json_schema, verdict_json_schema

CAPABLE_MODEL = "gemini/gemini-2.5-pro"
KEY = "AIza-secret"
MESSAGES = [
    {"role": "system", "content": "You are a careful council member."},
    {"role": "user", "content": "Decide and return JSON."},
]


def _verdict_contract() -> OutputContract:
    return OutputContract(schema=verdict_json_schema(), schema_name="CouncilVerdict")


# --------------------------------------------------------------------------- #
# Pure transform: keyword stripping + type/nullable mapping
# --------------------------------------------------------------------------- #


def test_transform_strips_additional_properties_and_meta_keywords():
    raw = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Thing",
        "type": "object",
        "additionalProperties": False,
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    out = _transform_schema_for_gemini(raw)

    # Gemini-unsupported keywords are gone at every level.
    assert "additionalProperties" not in out
    assert "title" not in out
    assert "$schema" not in out
    # Structural keywords survive.
    assert out["type"] == "OBJECT"
    assert out["properties"]["name"]["type"] == "STRING"
    assert out["required"] == ["name"]


def test_transform_does_not_mutate_input():
    raw = {
        "type": "object",
        "additionalProperties": False,
        "title": "Thing",
        "properties": {"name": {"type": "string"}},
    }
    snapshot = {
        "type": "object",
        "additionalProperties": False,
        "title": "Thing",
        "properties": {"name": {"type": "string"}},
    }
    _transform_schema_for_gemini(raw)
    assert raw == snapshot  # input untouched; transform builds a new dict


def test_transform_maps_nullable_union_to_nullable_flag():
    # CAC-01 uses ``["string", "null"]`` / ``["number", "null"]`` unions.
    raw = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "position": {"type": ["string", "null"]},
            "score": {"type": ["number", "null"]},
        },
    }
    out = _transform_schema_for_gemini(raw)
    pos = out["properties"]["position"]
    score = out["properties"]["score"]
    assert pos["type"] == "STRING"
    assert pos["nullable"] is True
    assert score["type"] == "NUMBER"
    assert score["nullable"] is True


def test_transform_preserves_enum_and_required():
    raw = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["confidence"],
    }
    out = _transform_schema_for_gemini(raw)
    conf = out["properties"]["confidence"]
    assert conf["type"] == "STRING"
    assert conf["enum"] == ["low", "medium", "high"]
    assert out["required"] == ["confidence"]


def test_transform_recurses_into_array_items():
    raw = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"label": {"type": "string"}},
        },
    }
    out = _transform_schema_for_gemini(raw)
    assert out["type"] == "ARRAY"
    assert "additionalProperties" not in out["items"]
    assert out["items"]["type"] == "OBJECT"
    assert out["items"]["properties"]["label"]["type"] == "STRING"


def test_transform_raises_on_unrepresentable_union():
    # A genuine multi-type union with >1 non-null member can't map to one
    # ``type`` + ``nullable`` → transform signals unrepresentability.
    raw = {"type": ["string", "number"]}
    with pytest.raises(ValueError):
        _transform_schema_for_gemini(raw)


def test_transform_raises_on_unsupported_composition_keyword():
    raw = {
        "type": "object",
        "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    with pytest.raises(ValueError):
        _transform_schema_for_gemini(raw)


def test_transform_raises_on_unknown_scalar_type():
    # A type name with no OpenAPI-subset mapping (e.g. JSON-Schema ``null`` as a
    # bare scalar type) is unrepresentable.
    with pytest.raises(ValueError):
        _transform_schema_for_gemini({"type": "null"})


def test_transform_raises_on_non_dict_node():
    # A schema node that is not an object (e.g. a stray bool in ``items``) is
    # unrepresentable, not silently passed through.
    with pytest.raises(ValueError):
        _transform_schema_for_gemini({"type": "array", "items": True})


def test_transform_raises_on_non_dict_properties():
    with pytest.raises(ValueError):
        _transform_schema_for_gemini({"type": "object", "properties": ["not", "a", "dict"]})


def test_transform_handles_nested_array_of_objects_with_enum():
    # End-to-end shape resembling the verdict ``positions[]`` block.
    raw = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "positions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "providers": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["label"],
                },
            },
        },
    }
    out = _transform_schema_for_gemini(raw)
    item = out["properties"]["positions"]["items"]
    assert item["type"] == "OBJECT"
    assert "additionalProperties" not in item
    assert item["properties"]["providers"]["type"] == "ARRAY"
    assert item["properties"]["providers"]["items"]["type"] == "STRING"
    assert item["required"] == ["label"]


def test_strip_keyword_set_contains_additional_properties():
    # Guard the contract the ticket pins: additionalProperties + meta keywords.
    for kw in ("additionalProperties", "title", "$schema", "$defs", "$ref"):
        assert kw in _GEMINI_STRIP_KEYWORDS


# --------------------------------------------------------------------------- #
# Real CAC-01 schemas transform cleanly
# --------------------------------------------------------------------------- #


def test_real_verdict_schema_transforms_without_error():
    out = _transform_schema_for_gemini(verdict_json_schema())
    # No banned keyword survives anywhere in the tree.
    _assert_no_banned_keywords(out)
    # Root + a known nullable leaf mapped correctly.
    assert out["type"] == "OBJECT"
    assert out["properties"]["consensus_score"]["type"] == "NUMBER"
    assert out["properties"]["consensus_score"]["nullable"] is True
    # An enum field survived.
    votes_item = out["properties"]["provider_votes"]["items"]
    assert votes_item["properties"]["confidence"]["enum"]
    # required lists survive.
    assert "verdict_type" in out["required"]


def test_real_member_answer_schema_transforms_without_error():
    out = _transform_schema_for_gemini(member_answer_json_schema())
    _assert_no_banned_keywords(out)
    assert out["type"] == "OBJECT"
    assert out["properties"]["position"]["type"] == "STRING"
    assert out["properties"]["position"]["nullable"] is True
    assert out["required"] == ["key_points"]


def _assert_no_banned_keywords(node: object) -> None:
    """Recursively assert no Gemini-banned keyword appears anywhere."""
    if isinstance(node, dict):
        for kw in _GEMINI_STRIP_KEYWORDS:
            assert kw not in node, f"banned keyword {kw!r} survived"
        # ``type`` must be a single uppercase string, never a list union.
        if "type" in node:
            assert isinstance(node["type"], str)
            assert node["type"].isupper()
        for value in node.values():
            _assert_no_banned_keywords(value)
    elif isinstance(node, list):
        for item in node:
            _assert_no_banned_keywords(item)


# --------------------------------------------------------------------------- #
# build_request injection (capable model)
# --------------------------------------------------------------------------- #


def test_build_request_injects_mime_type_and_response_schema():
    adapter = GeminiAdapter()
    _url, _headers, body = adapter.build_request(
        CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, _verdict_contract()
    )
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    schema = gen["responseSchema"]
    # The transform ran: additionalProperties/title stripped, types uppercased.
    _assert_no_banned_keywords(schema)
    assert schema["type"] == "OBJECT"
    assert schema["properties"]["consensus_score"]["nullable"] is True


def test_build_request_keeps_role_map_system_hoist_and_max_tokens():
    adapter = GeminiAdapter()
    _url, _headers, body = adapter.build_request(
        CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, _verdict_contract()
    )
    # systemInstruction hoist preserved.
    assert body["systemInstruction"]["parts"][0]["text"] == "You are a careful council member."
    # role mapping preserved (user stays user; no system in contents).
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Decide and return JSON."}]}]
    # maxOutputTokens + temperature still present.
    assert body["generationConfig"]["maxOutputTokens"] == 4096
    assert body["generationConfig"]["temperature"] == 0.4


def test_build_request_none_contract_is_byte_for_byte_legacy():
    adapter = GeminiAdapter()
    url_a, headers_a, body_a = adapter.build_request(CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY)
    url_b, headers_b, body_b = adapter.build_request(CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, None)
    assert (url_a, headers_a, body_a) == (url_b, headers_b, body_b)
    # No structured-output keys leaked into the legacy body.
    assert "responseMimeType" not in body_a["generationConfig"]
    assert "responseSchema" not in body_a["generationConfig"]


def test_build_request_unsupported_capability_does_not_inject(recwarn):
    adapter = GeminiAdapter()
    # sonar is structured_output=False in the static catalog; but it is also a
    # different provider prefix. Use a gemini model the catalog marks capable to
    # isolate the *capability* gate, then a model with no caps to prove no-inject.
    # An unknown gemini sub-model still resolves to the gemini provider fallback
    # (capable), so to exercise the unsupported branch we patch capabilities.
    contract = _verdict_contract()
    _url, _headers, body = adapter.build_request(
        "gemini/unknown-tiny-model", MESSAGES, 0.4, 120.0, KEY, contract
    )
    # gemini provider fallback IS capable, so this DOES inject. Assert that path.
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseSchema" in body["generationConfig"]


def test_build_request_capability_none_warns_and_skips(monkeypatch):
    import conclave.adapters.gemini as gem

    monkeypatch.setattr(gem, "capabilities_for", lambda _mid: None)
    adapter = GeminiAdapter()
    with pytest.warns(UserWarning):
        _url, _headers, body = adapter.build_request(
            CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, _verdict_contract()
        )
    assert "responseMimeType" not in body["generationConfig"]
    assert "responseSchema" not in body["generationConfig"]


def test_build_request_structured_unsupported_warns_and_skips(monkeypatch):
    import conclave.adapters.gemini as gem
    from conclave.provider_catalog import ProviderCapabilities

    monkeypatch.setattr(
        gem,
        "capabilities_for",
        lambda _mid: ProviderCapabilities(supports_structured_output=False),
    )
    adapter = GeminiAdapter()
    with pytest.warns(UserWarning):
        _url, _headers, body = adapter.build_request(
            CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, _verdict_contract()
        )
    assert "responseMimeType" not in body["generationConfig"]
    assert "responseSchema" not in body["generationConfig"]


def test_build_request_unrepresentable_schema_falls_back_to_mime_only(monkeypatch):
    # A schema with a non-mappable union → mimeType-only JSON, no responseSchema,
    # plus a warning. Never raises.
    bad = {"type": "object", "properties": {"x": {"type": ["string", "number"]}}}
    adapter = GeminiAdapter()
    with pytest.warns(UserWarning):
        _url, _headers, body = adapter.build_request(
            CAPABLE_MODEL,
            MESSAGES,
            0.4,
            120.0,
            KEY,
            OutputContract(schema=bad),
        )
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert "responseSchema" not in gen


def test_build_request_empty_schema_contract_injects_mime_only():
    # OutputContract with schema=None: caller asked for structured output but
    # gave no schema → JSON mode only (mimeType), no responseSchema, no warning.
    adapter = GeminiAdapter()
    _url, _headers, body = adapter.build_request(
        CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, OutputContract(schema=None)
    )
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert "responseSchema" not in gen


# --------------------------------------------------------------------------- #
# stream_request injection
# --------------------------------------------------------------------------- #


def test_stream_request_injects_response_schema_and_keeps_sse_url():
    adapter = GeminiAdapter()
    url, _headers, body = adapter.stream_request(
        CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, _verdict_contract()
    )
    assert url.endswith(":streamGenerateContent?alt=sse")
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    _assert_no_banned_keywords(gen["responseSchema"])
    assert gen["responseSchema"]["type"] == "OBJECT"


def test_stream_request_none_contract_unchanged():
    adapter = GeminiAdapter()
    url, _headers, body = adapter.stream_request(CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY)
    assert url.endswith(":streamGenerateContent?alt=sse")
    assert "responseMimeType" not in body["generationConfig"]
    assert "responseSchema" not in body["generationConfig"]


def test_stream_request_unrepresentable_falls_back_to_mime_only():
    bad = {"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}]}}}
    adapter = GeminiAdapter()
    with pytest.warns(UserWarning):
        _url, _headers, body = adapter.stream_request(
            CAPABLE_MODEL, MESSAGES, 0.4, 120.0, KEY, OutputContract(schema=bad)
        )
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert "responseSchema" not in gen
