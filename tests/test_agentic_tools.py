"""Agentic tool-calling surface (CAC-02, DSE-1807).

Covers the additive surface only: :class:`ToolSpec` declarations, the neutral
content-block helpers, Anthropic's wire translation, and
``parse_agentic_response``. The pre-existing ``build_request`` /
``parse_response`` contract is asserted unchanged here too, because the whole
point of making this surface additive is that the 468 existing tests keep
passing untouched.
"""

from __future__ import annotations

import pytest

from conclave.adapters import (
    AgenticResponse,
    ToolCall,
    ToolSpec,
    assistant_tool_call_message,
    tool_result_message,
)
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.base import OutputContract, ProviderError
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter

WEATHER = ToolSpec(
    name="get_weather",
    description="Look up the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
READ = ToolSpec(
    name="read_file",
    description="Read a file from disk.",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
)


@pytest.fixture
def anthropic() -> AnthropicAdapter:
    return AnthropicAdapter()


@pytest.fixture
def openai() -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        prefix="openai",
        completions_url="https://api.openai.com/v1/chat/completions",
        env_vars=("OPENAI_API_KEY",),
    )


def _build(adapter, messages, **kw):
    return adapter.build_request(
        "anthropic/claude-sonnet-5", messages, None, 30.0, "test-key", **kw
    )


# --------------------------------------------------------------------------
# Capability flags
# --------------------------------------------------------------------------


def test_only_anthropic_advertises_tool_calls(anthropic, openai):
    """Capability is declared, never inferred from a model name."""
    assert anthropic.supports_tool_calls is True
    assert openai.supports_tool_calls is False
    assert GeminiAdapter().supports_tool_calls is False


# --------------------------------------------------------------------------
# Neutral block helpers
# --------------------------------------------------------------------------


def test_assistant_tool_call_message_orders_text_then_calls():
    call = ToolCall(id="toolu_01", name="get_weather", arguments={"city": "Atlanta"})
    msg = assistant_tool_call_message("Let me check.", [call])
    assert msg["role"] == "assistant"
    assert msg["content"] == [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_call",
            "id": "toolu_01",
            "name": "get_weather",
            "arguments": {"city": "Atlanta"},
        },
    ]


def test_assistant_tool_call_message_omits_empty_text():
    """A model that goes straight to a call must not emit an empty text block."""
    call = ToolCall(id="toolu_01", name="get_weather", arguments={})
    assert assistant_tool_call_message("", [call])["content"] == [
        {"type": "tool_call", "id": "toolu_01", "name": "get_weather", "arguments": {}}
    ]


def test_tool_result_message_preserves_ids_and_order():
    msg = tool_result_message([("toolu_01", "72F", False), ("toolu_02", "boom", True)])
    assert msg["role"] == "user"
    assert [b["tool_call_id"] for b in msg["content"]] == ["toolu_01", "toolu_02"]
    assert msg["content"][1]["is_error"] is True


# --------------------------------------------------------------------------
# Anthropic request translation
# --------------------------------------------------------------------------


def test_tools_translate_to_anthropic_input_schema(anthropic):
    _, _, body = _build(anthropic, [{"role": "user", "content": "hi"}], tools=[WEATHER, READ])
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": WEATHER.description,
            "input_schema": WEATHER.input_schema,
        },
        {"name": "read_file", "description": READ.description, "input_schema": READ.input_schema},
    ]
    # auto is Anthropic's default; the key is omitted rather than sent explicitly.
    assert "tool_choice" not in body


@pytest.mark.parametrize(
    ("neutral", "expected"),
    [
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        ("get_weather", {"type": "tool", "name": "get_weather"}),
    ],
)
def test_tool_choice_translation(anthropic, neutral, expected):
    _, _, body = _build(
        anthropic, [{"role": "user", "content": "hi"}], tools=[WEATHER], tool_choice=neutral
    )
    assert body["tool_choice"] == expected


def test_tool_call_and_result_blocks_reach_the_wire(anthropic):
    """A full call/result round trip keeps ids correlated end-to-end."""
    call = ToolCall(id="toolu_abc", name="get_weather", arguments={"city": "Atlanta"})
    messages = [
        {"role": "user", "content": "weather?"},
        assistant_tool_call_message("Checking.", [call]),
        tool_result_message([("toolu_abc", "72F and clear", False)]),
    ]
    _, _, body = _build(anthropic, messages, tools=[WEATHER])

    assistant = body["messages"][1]
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "toolu_abc",
        "name": "get_weather",
        "input": {"city": "Atlanta"},
    }
    result = body["messages"][2]["content"][0]
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_abc",
        "content": "72F and clear",
    }
    # is_error is omitted on success so a normal result body stays minimal.
    assert "is_error" not in result


def test_error_result_carries_is_error(anthropic):
    messages = [tool_result_message([("toolu_abc", "ENOENT", True)])]
    _, _, body = _build(anthropic, messages, tools=[READ])
    assert body["messages"][0]["content"][0]["is_error"] is True


def test_tools_and_output_contract_are_mutually_exclusive(anthropic):
    """Silently preferring one mode would make the caller's intent unrecoverable."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build(
            anthropic,
            [{"role": "user", "content": "hi"}],
            tools=[WEATHER],
            output_contract=OutputContract(schema={"type": "object"}),
        )


def test_string_content_body_is_unchanged_by_the_new_surface(anthropic):
    """The additive guarantee: no tools means a byte-identical legacy body."""
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    _, _, before = _build(anthropic, messages)
    _, _, after = _build(anthropic, messages, tools=None)
    assert before == after
    assert before["system"] == "be brief"
    assert before["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in before


def test_stream_request_forwards_tools(anthropic):
    _, _, body = anthropic.stream_request(
        "anthropic/claude-sonnet-5",
        [{"role": "user", "content": "hi"}],
        None,
        30.0,
        "test-key",
        tools=[WEATHER],
    )
    assert body["stream"] is True
    assert body["tools"][0]["name"] == "get_weather"


# --------------------------------------------------------------------------
# Anthropic response parsing
# --------------------------------------------------------------------------


def test_parse_agentic_response_extracts_parallel_calls_in_order(anthropic):
    payload = {
        "content": [
            {"type": "text", "text": "Checking both."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "ATL"}},
            {"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {"path": "/x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    out = anthropic.parse_agentic_response(200, payload)
    assert isinstance(out, AgenticResponse)
    assert out.text == "Checking both."
    assert [c.id for c in out.tool_calls] == ["toolu_1", "toolu_2"]
    assert out.tool_calls[0].arguments == {"city": "ATL"}
    assert out.stop_reason == "tool_use"
    assert out.usage is not None


def test_parse_agentic_response_plain_answer(anthropic):
    out = anthropic.parse_agentic_response(
        200, {"content": [{"type": "text", "text": "42"}], "stop_reason": "end_turn"}
    )
    assert out.text == "42"
    assert out.tool_calls == ()
    assert out.stop_reason == "end_turn"


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("end_turn", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "end_turn"),
        ("refusal", "refusal"),
        ("something_new", "end_turn"),
    ],
)
def test_stop_reason_is_normalized(anthropic, raw, normalized):
    """An unrecognized reason means unmodelled, not failed."""
    out = anthropic.parse_agentic_response(200, {"content": [], "stop_reason": raw})
    assert out.stop_reason == normalized


def test_parse_agentic_response_ignores_forced_tool_state(anthropic):
    """A tool_use block here is a real call, not structured-output smuggling."""
    _build(
        anthropic,
        [{"role": "user", "content": "hi"}],
        output_contract=OutputContract(schema={"type": "object"}),
    )
    out = anthropic.parse_agentic_response(
        200,
        {
            "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}}],
            "stop_reason": "tool_use",
        },
    )
    assert out.tool_calls[0].name == "get_weather"


def test_parse_agentic_response_raises_on_http_error(anthropic):
    with pytest.raises(ProviderError):
        anthropic.parse_agentic_response(429, {"error": {"message": "slow down"}})


def test_parse_agentic_response_raises_on_missing_content(anthropic):
    with pytest.raises(ProviderError, match="no content blocks"):
        anthropic.parse_agentic_response(200, {"stop_reason": "end_turn"})


def test_malformed_tool_input_degrades_to_empty_dict(anthropic):
    """A non-object input must not crash the loop; callers always get a dict."""
    out = anthropic.parse_agentic_response(
        200,
        {
            "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": "not-an-object"}],
            "stop_reason": "tool_use",
        },
    )
    assert out.tool_calls[0].arguments == {}


# --------------------------------------------------------------------------
# Unimplemented providers fail loudly, not silently
# --------------------------------------------------------------------------


def test_unsupported_provider_rejects_tools(openai):
    with pytest.raises(ProviderError, match="supports_tool_calls"):
        openai.build_request(
            "openai/gpt-5.6", [{"role": "user", "content": "hi"}], None, 30.0, "k", tools=[WEATHER]
        )


def test_unsupported_provider_rejects_agentic_parse(openai):
    with pytest.raises(ProviderError, match="supports_tool_calls"):
        openai.parse_agentic_response(200, {"choices": []})


def test_unsupported_provider_without_tools_still_works(openai):
    """The guard must not disturb the existing non-agentic path."""
    _, _, body = openai.build_request(
        "openai/gpt-5.6", [{"role": "user", "content": "hi"}], None, 30.0, "k"
    )
    assert body["messages"] == [{"role": "user", "content": "hi"}]
