"""Structured-output tests for the Anthropic adapter (CAC-02-ANT).

Anthropic has no OpenAI-style ``response_format``; structured output is forced
*tool use*. These tests pin the request shape and the response/stream parsing
for the structured path, and assert the no-contract path is unchanged:

* ``build_request`` injects a single tool (``input_schema`` = contract schema)
  and pins ``tool_choice`` when a contract is present AND the model is
  catalog-capable; injects nothing (and never raises) otherwise.
* ``parse_response`` extracts the forced ``tool_use`` block's ``input`` and
  serializes it to JSON (round-trip), while the free-prose path still
  concatenates ``text`` blocks and still ignores stray ``tool_use`` blocks.
* token usage is parsed identically in both paths.
* streaming accumulates ``input_json_delta`` fragments into the buffered text.

The free-prose request/response/stream behavior is covered by
``tests/test_adapters.py``; here we assert it is *unchanged* by the new code.
"""

from __future__ import annotations

import json

import pytest

from conclave.adapters import ProviderError
from conclave.adapters.anthropic import DEFAULT_TOOL_NAME, AnthropicAdapter
from conclave.adapters.base import OutputContract

# A capable Anthropic model per the static catalog
# (``supports_structured_output=True``).
CAPABLE_MODEL = "anthropic/claude-sonnet-4-6"

# A small JSON Schema standing in for the council verdict/member schema.
SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

_MESSAGES = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "decide"},
]


# --------------------------------------------------------------------------- #
# build_request — tool + tool_choice injection
# --------------------------------------------------------------------------- #


def test_build_request_injects_tool_and_tool_choice_when_capable():
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA, schema_name="member_answer")
    _url, _headers, body = adapter.build_request(
        CAPABLE_MODEL, _MESSAGES, 0.2, 120.0, "sk-ant-secret", contract
    )
    assert body["tools"] == [
        {
            "name": "member_answer",
            "description": "Return the result as structured data.",
            "input_schema": SCHEMA,
        }
    ]
    assert body["tool_choice"] == {"type": "tool", "name": "member_answer"}


def test_build_request_tool_name_defaults_to_verdict_without_schema_name():
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA)  # no schema_name
    _url, _headers, body = adapter.build_request(
        CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", contract
    )
    assert body["tools"][0]["name"] == DEFAULT_TOOL_NAME == "verdict"
    assert body["tool_choice"] == {"type": "tool", "name": "verdict"}


def test_build_request_keeps_system_hoist_and_max_tokens_with_tool():
    adapter = AnthropicAdapter(max_tokens=1024)
    contract = OutputContract(schema=SCHEMA)
    _url, _headers, body = adapter.build_request(
        CAPABLE_MODEL, _MESSAGES, 0.5, 120.0, "k", contract
    )
    # System hoist + required max_tokens are intact alongside the tool.
    assert body["system"] == "be terse"
    assert body["max_tokens"] == 1024
    assert body["messages"] == [{"role": "user", "content": "decide"}]
    assert body["temperature"] == 0.5
    assert "tools" in body and "tool_choice" in body


# --------------------------------------------------------------------------- #
# build_request — capability gating (never raises)
# --------------------------------------------------------------------------- #


def test_build_request_no_tool_when_contract_is_none():
    """output_contract=None -> body byte-for-byte the free-prose request."""
    adapter = AnthropicAdapter()
    _url, _headers, body = adapter.build_request(CAPABLE_MODEL, _MESSAGES, 0.2, 120.0, "k", None)
    assert "tools" not in body
    assert "tool_choice" not in body
    assert body == {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "decide"}],
        "temperature": 0.2,
        "system": "be terse",
    }


def test_build_request_no_tool_for_unsupported_model_and_warns(caplog):
    """An incapable model degrades to free prose with a non-fatal warning."""
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA)
    # perplexity/sonar-pro: supports_structured_output=False in the catalog.
    with caplog.at_level("WARNING", logger="conclave"):
        _url, _headers, body = adapter.build_request(
            "perplexity/sonar-pro", _MESSAGES, 0.2, 120.0, "k", contract
        )
    assert "tools" not in body
    assert "tool_choice" not in body
    assert any("unsupported/unknown" in r.getMessage() for r in caplog.records)


def test_build_request_no_tool_for_unknown_provider_and_warns(caplog):
    """An unknown provider (capabilities_for -> None) degrades, never raises."""
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA)
    with caplog.at_level("WARNING", logger="conclave"):
        _url, _headers, body = adapter.build_request(
            "anthropic/some-unknown-future-model",
            _MESSAGES,
            0.2,
            120.0,
            "k",
            contract,
        )
    # anthropic/ prefix falls back to a capable record, so the tool IS injected
    # here -- prove the *unknown-provider* branch with a non-anthropic prefix.
    assert "tools" in body  # sanity: anthropic prefix is catalog-capable

    adapter2 = AnthropicAdapter()
    with caplog.at_level("WARNING", logger="conclave"):
        _u, _h, body2 = adapter2.build_request(
            "nosuchprovider/model", _MESSAGES, 0.2, 120.0, "k", contract
        )
    assert "tools" not in body2
    assert any("unsupported/unknown" in r.getMessage() for r in caplog.records)


def test_build_request_no_tool_when_contract_has_no_schema():
    """A contract with schema=None cannot constrain output -> no tool, no warn."""
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=None, schema_name="verdict")
    _url, _headers, body = adapter.build_request(
        CAPABLE_MODEL, _MESSAGES, 0.2, 120.0, "k", contract
    )
    assert "tools" not in body


# --------------------------------------------------------------------------- #
# parse_response — structured (tool_use) extraction
# --------------------------------------------------------------------------- #


def test_parse_response_extracts_tool_use_input_round_trip():
    """After a forced tool, parse_response returns the tool_use input as JSON."""
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA, schema_name="member_answer")
    # build_request sets the forced-tool flag parse_response reads.
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, 0.2, 120.0, "k", contract)

    obj = {"answer": "ship it", "confidence": 0.9}
    payload = {
        "content": [
            {"type": "tool_use", "id": "tu_1", "name": "member_answer", "input": obj},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }
    text, usage = adapter.parse_response(200, payload)
    # The answer string round-trips back to the original object.
    assert json.loads(text) == obj
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        12,
        5,
        17,
    )


def test_parse_response_prefers_named_tool_block():
    """When several tool_use blocks exist, the one matching the name wins."""
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA, schema_name="verdict")
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", contract)

    payload = {
        "content": [
            {"type": "tool_use", "name": "other", "input": {"answer": "no"}},
            {"type": "tool_use", "name": "verdict", "input": {"answer": "yes"}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    text, _usage = adapter.parse_response(200, payload)
    assert json.loads(text) == {"answer": "yes"}


def test_parse_response_falls_back_to_first_tool_use_block():
    """If no block matches by name, the first tool_use block is used."""
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA, schema_name="verdict")
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", contract)

    payload = {
        "content": [
            {"type": "text", "text": "preamble"},
            {"type": "tool_use", "name": "unexpected", "input": {"answer": "only"}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    text, _usage = adapter.parse_response(200, payload)
    assert json.loads(text) == {"answer": "only"}


def test_parse_response_raises_when_tool_forced_but_absent():
    """Model ignored the forced tool -> a redacted ProviderError, not text."""
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA, schema_name="verdict")
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", contract)

    payload = {"content": [{"type": "text", "text": "no tool here"}]}
    with pytest.raises(ProviderError, match="no tool_use content"):
        adapter.parse_response(200, payload)


def test_parse_response_raises_when_tool_input_not_object():
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA, schema_name="verdict")
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", contract)

    payload = {"content": [{"type": "tool_use", "name": "verdict", "input": "oops"}]}
    with pytest.raises(ProviderError, match="input is not an object"):
        adapter.parse_response(200, payload)


# --------------------------------------------------------------------------- #
# parse_response — free-prose path is UNCHANGED
# --------------------------------------------------------------------------- #


def test_parse_response_text_path_unchanged_without_contract():
    """No forced tool -> text concatenation, and stray tool_use is ignored."""
    adapter = AnthropicAdapter()
    # No build_request call (or a None-contract one) -> _forced_tool_name is None.
    payload = {
        "content": [
            {"type": "text", "text": "hello "},
            {"type": "tool_use", "id": "x", "input": {"k": "v"}},  # ignored
            {"type": "text", "text": "world"},
        ],
        "usage": {"input_tokens": 9, "output_tokens": 3},
    }
    text, usage = adapter.parse_response(200, payload)
    assert text == "hello world"
    assert usage is not None
    assert usage.total_tokens == 12


def test_none_contract_build_then_parse_uses_text_path():
    """A None-contract build clears the flag; parse stays on the text path."""
    adapter = AnthropicAdapter()
    # First a structured build sets the flag...
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", OutputContract(schema=SCHEMA))
    assert adapter._forced_tool_name == "verdict"
    # ...then a None-contract build must clear it.
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", None)
    assert adapter._forced_tool_name is None

    payload = {
        "content": [{"type": "text", "text": "plain"}],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    text, _usage = adapter.parse_response(200, payload)
    assert text == "plain"


def test_parse_response_error_status_still_raises_in_structured_mode():
    adapter = AnthropicAdapter()
    adapter.build_request(CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", OutputContract(schema=SCHEMA))
    payload = {"error": {"type": "overloaded_error", "message": "overloaded"}}
    with pytest.raises(ProviderError, match="HTTP 529"):
        adapter.parse_response(529, payload)


# --------------------------------------------------------------------------- #
# Streaming — accumulate input_json_delta into buffered text
# --------------------------------------------------------------------------- #


def test_stream_request_injects_tool_when_capable():
    adapter = AnthropicAdapter()
    contract = OutputContract(schema=SCHEMA, schema_name="verdict")
    _url, _headers, body = adapter.stream_request(
        CAPABLE_MODEL, _MESSAGES, None, 120.0, "k", contract
    )
    assert body["stream"] is True
    assert body["tool_choice"] == {"type": "tool", "name": "verdict"}
    assert body["tools"][0]["input_schema"] == SCHEMA


def test_parse_sse_accumulates_input_json_delta_into_full_object():
    """The partial_json fragments concatenate into the full tool input JSON."""
    adapter = AnthropicAdapter()
    fragments = ['{"answer":', ' "ship', ' it", "confidence"', ": 0.9}"]
    accumulated = ""
    for frag in fragments:
        frame = json.dumps(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": frag},
            }
        )
        delta = adapter.parse_sse_event("content_block_delta", frame)
        accumulated += delta.text
    assert json.loads(accumulated) == {"answer": "ship it", "confidence": 0.9}


def test_parse_sse_input_json_delta_empty_fragment_yields_no_text():
    """The leading empty partial_json fragment contributes nothing."""
    adapter = AnthropicAdapter()
    frame = json.dumps(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ""},
        }
    )
    delta = adapter.parse_sse_event("content_block_delta", frame)
    assert delta.text == ""


def test_parse_sse_text_delta_unchanged():
    """The existing text_delta path is untouched by the input_json_delta branch."""
    adapter = AnthropicAdapter()
    frame = json.dumps(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "tok"},
        }
    )
    delta = adapter.parse_sse_event("content_block_delta", frame)
    assert delta.text == "tok"


def test_parse_sse_usage_and_done_unchanged():
    """message_delta (usage) and message_stop (done) behave as before."""
    adapter = AnthropicAdapter()
    usage_frame = json.dumps({"type": "message_delta", "usage": {"output_tokens": 7}})
    usage_delta = adapter.parse_sse_event("message_delta", usage_frame)
    assert usage_delta.usage is not None
    assert usage_delta.usage.completion_tokens == 7

    done = adapter.parse_sse_event("message_stop", "{}")
    assert done.done is True
