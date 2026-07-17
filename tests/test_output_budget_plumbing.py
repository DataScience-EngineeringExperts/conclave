"""Output-token ceilings are optional, provider-native, and end-to-end."""

from __future__ import annotations

from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter
from conclave.models import ModelAnswer
from conclave.providers import call_model, call_model_stream

MESSAGES = [{"role": "user", "content": "hi"}]


def test_adapter_caps_override_native_defaults_without_changing_default_bodies():
    openai = OpenAICompatAdapter("openai", "https://example.test/v1/chat", ("OPENAI_API_KEY",))
    anthropic = AnthropicAdapter()
    gemini = GeminiAdapter()

    _, _, openai_default = openai.build_request("openai/gpt-4.1", MESSAGES, 0.2, 30, "key")
    _, _, anthropic_default = anthropic.build_request(
        "anthropic/claude-sonnet-4-20250514", MESSAGES, 0.2, 30, "key"
    )
    _, _, gemini_default = gemini.build_request("gemini/gemini-2.5-pro", MESSAGES, 0.2, 30, "key")

    _, _, openai_capped = openai.build_request(
        "openai/gpt-4.1", MESSAGES, 0.2, 30, "key", max_output_tokens=321
    )
    _, _, anthropic_capped = anthropic.build_request(
        "anthropic/claude-sonnet-4-20250514",
        MESSAGES,
        0.2,
        30,
        "key",
        max_output_tokens=321,
    )
    _, _, gemini_capped = gemini.build_request(
        "gemini/gemini-2.5-pro", MESSAGES, 0.2, 30, "key", max_output_tokens=321
    )

    assert "max_tokens" not in openai_default
    assert anthropic_default["max_tokens"] == anthropic.max_tokens
    assert gemini_default["generationConfig"]["maxOutputTokens"] == gemini.max_output_tokens
    assert openai_capped["max_tokens"] == 321
    assert anthropic_capped["max_tokens"] == 321
    assert gemini_capped["generationConfig"]["maxOutputTokens"] == 321


def test_stream_requests_receive_the_same_optional_cap():
    adapters_and_models = (
        (
            OpenAICompatAdapter("openai", "https://example.test/v1/chat", ("OPENAI_API_KEY",)),
            "openai/gpt-4.1",
            lambda body: body["max_tokens"],
        ),
        (AnthropicAdapter(), "anthropic/claude-sonnet-4-20250514", lambda body: body["max_tokens"]),
        (
            GeminiAdapter(),
            "gemini/gemini-2.5-pro",
            lambda body: body["generationConfig"]["maxOutputTokens"],
        ),
    )
    for adapter, model_id, extract in adapters_and_models:
        _, _, body = adapter.stream_request(
            model_id, MESSAGES, 0.2, 30, "key", max_output_tokens=654
        )
        assert extract(body) == 654


async def test_call_model_threads_max_output_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")
    captured = {}

    async def fake_post_json(url, headers, json_body, timeout):
        captured["body"] = json_body
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("conclave.transport.post_json", fake_post_json)
    answer = await call_model(
        "openai",
        "openai/gpt-4.1",
        MESSAGES,
        max_output_tokens=777,
    )
    assert answer.ok
    assert captured["body"]["max_tokens"] == 777


async def test_call_model_stream_threads_max_output_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")
    captured = {}

    async def fake_stream_sse(url, headers, json_body, timeout):
        captured["body"] = json_body
        yield "", '{"choices":[{"delta":{"content":"ok"}}]}'
        yield "", "[DONE]"

    monkeypatch.setattr("conclave.transport.stream_sse", fake_stream_sse)
    items = [
        item
        async for item in call_model_stream(
            "openai",
            "openai/gpt-4.1",
            MESSAGES,
            max_output_tokens=888,
        )
    ]
    assert isinstance(items[-1], ModelAnswer)
    assert items[-1].ok
    assert captured["body"]["max_tokens"] == 888
