"""Per-adapter tests: ``build_request`` and ``parse_response`` for each provider.

These exercise the request-shaping and response-parsing of the three concrete
adapters in the owned provider highway:

* ``build_request`` — URL, exact auth header, and body shape (incl. the Anthropic
  top-level ``system`` hoist + required ``max_tokens`` and the Gemini role mapping
  + ``systemInstruction``).
* ``parse_response`` — over realistic recorded payloads: a success, a
  malformed/empty body, and a non-2xx status (-> ``ProviderError``).

Registry resolution, end-to-end ``call_model``, and ``redact`` live in
``test_providers.py``.
"""

from __future__ import annotations

import pytest

from conclave.adapters import ProviderError
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter


# --------------------------------------------------------------------------- #
# OpenAI-compatible adapter (openai / xai / perplexity)
# --------------------------------------------------------------------------- #


def _openai_adapter() -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        prefix="openai",
        completions_url="https://api.openai.com/v1/chat/completions",
        env_vars=("OPENAI_API_KEY",),
    )


def test_openai_compat_build_request():
    adapter = _openai_adapter()
    messages = [{"role": "user", "content": "hi"}]
    url, headers, body = adapter.build_request(
        "openai/gpt-4.1", messages, 0.7, 120.0, "sk-secret"
    )
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-secret"
    assert headers["Content-Type"] == "application/json"
    # Bare model id (prefix stripped), messages passed through, temperature set.
    assert body["model"] == "gpt-4.1"
    assert body["messages"] == messages
    assert body["temperature"] == 0.7
    # No max_tokens unless configured.
    assert "max_tokens" not in body


def test_openai_compat_max_tokens_included_when_set():
    adapter = OpenAICompatAdapter(
        prefix="xai",
        completions_url="https://api.x.ai/v1/chat/completions",
        env_vars=("XAI_API_KEY",),
        max_tokens=256,
    )
    _url, _headers, body = adapter.build_request(
        "xai/grok-4.3", [{"role": "user", "content": "q"}], 0.5, 30.0, "xai-key"
    )
    assert body["model"] == "grok-4.3"
    assert body["max_tokens"] == 256


def test_openai_compat_parse_success():
    adapter = _openai_adapter()
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "the answer"}}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "total_tokens": 15,
        },
    }
    text, usage = adapter.parse_response(200, payload)
    assert text == "the answer"
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        11,
        4,
        15,
    )


def test_openai_compat_parse_empty_content_raises():
    adapter = _openai_adapter()
    payload = {"choices": [{"message": {"content": ""}}]}
    with pytest.raises(ProviderError, match="empty response"):
        adapter.parse_response(200, payload)


def test_openai_compat_parse_malformed_raises():
    adapter = _openai_adapter()
    with pytest.raises(ProviderError, match="malformed response"):
        adapter.parse_response(200, {"unexpected": "shape"})


def test_openai_compat_parse_error_status_raises():
    adapter = _openai_adapter()
    payload = {"error": {"message": "invalid api key", "type": "auth_error"}}
    with pytest.raises(ProviderError, match="HTTP 401"):
        adapter.parse_response(401, payload)


# --------------------------------------------------------------------------- #
# Anthropic adapter
# --------------------------------------------------------------------------- #


def test_anthropic_build_request_hoists_system_and_requires_max_tokens():
    adapter = AnthropicAdapter()
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "prior"},
    ]
    url, headers, body = adapter.build_request(
        "anthropic/claude-sonnet-4-6", messages, 0.3, 120.0, "sk-ant-secret"
    )
    assert url == "https://api.anthropic.com/v1/messages"
    # Auth header is x-api-key (NOT Authorization), plus the version header.
    assert headers["x-api-key"] == "sk-ant-secret"
    assert "Authorization" not in headers
    assert headers["anthropic-version"] == "2023-06-01"
    # Bare model name and the REQUIRED max_tokens (default 4096).
    assert body["model"] == "claude-sonnet-4-6"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.3
    # System hoisted to a TOP-LEVEL string; only user/assistant turns remain.
    assert body["system"] == "be terse"
    assert body["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "prior"},
    ]


def test_anthropic_build_request_configurable_max_tokens_and_no_system():
    adapter = AnthropicAdapter(max_tokens=1024)
    _url, _headers, body = adapter.build_request(
        "anthropic/claude-sonnet-4-6",
        [{"role": "user", "content": "q"}],
        0.7,
        120.0,
        "key",
    )
    assert body["max_tokens"] == 1024
    # No system message -> no system key.
    assert "system" not in body


def test_anthropic_parse_success_concatenates_text_blocks():
    adapter = AnthropicAdapter()
    payload = {
        "content": [
            {"type": "text", "text": "hello "},
            {"type": "tool_use", "id": "x"},  # non-text block ignored
            {"type": "text", "text": "world"},
        ],
        "usage": {"input_tokens": 9, "output_tokens": 3},
    }
    text, usage = adapter.parse_response(200, payload)
    assert text == "hello world"
    assert usage is not None
    # total = input + output.
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        9,
        3,
        12,
    )


def test_anthropic_parse_empty_raises():
    adapter = AnthropicAdapter()
    with pytest.raises(ProviderError, match="empty response"):
        adapter.parse_response(200, {"content": []})


def test_anthropic_parse_malformed_raises():
    adapter = AnthropicAdapter()
    with pytest.raises(ProviderError, match="missing content array"):
        adapter.parse_response(200, {"no": "content"})


def test_anthropic_parse_error_status_raises():
    adapter = AnthropicAdapter()
    payload = {"error": {"type": "overloaded_error", "message": "overloaded"}}
    with pytest.raises(ProviderError, match="HTTP 529"):
        adapter.parse_response(529, payload)


# --------------------------------------------------------------------------- #
# Gemini adapter
# --------------------------------------------------------------------------- #


def test_gemini_build_request_role_mapping_and_system_instruction():
    adapter = GeminiAdapter()
    messages = [
        {"role": "system", "content": "stay factual"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "prior turn"},
    ]
    url, headers, body = adapter.build_request(
        "gemini/gemini-2.5-pro", messages, 0.4, 120.0, "AIza-secret"
    )
    # Model embedded in the URL path; gemini/ prefix stripped.
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-pro:generateContent"
    )
    # Auth header is x-goog-api-key (no Bearer).
    assert headers["x-goog-api-key"] == "AIza-secret"
    assert "Authorization" not in headers
    # Role mapping: user->user, assistant->model; system hoisted out.
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "prior turn"}]},
    ]
    assert body["systemInstruction"] == {"parts": [{"text": "stay factual"}]}
    # Generation config.
    assert body["generationConfig"]["temperature"] == 0.4
    assert body["generationConfig"]["maxOutputTokens"] == 4096


def test_gemini_build_request_configurable_max_output_tokens_no_system():
    adapter = GeminiAdapter(max_output_tokens=512)
    _url, _headers, body = adapter.build_request(
        "gemini/gemini-2.5-pro",
        [{"role": "user", "content": "q"}],
        0.7,
        120.0,
        "k",
    )
    assert body["generationConfig"]["maxOutputTokens"] == 512
    assert "systemInstruction" not in body


def test_gemini_parse_success():
    adapter = GeminiAdapter()
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "part one "}, {"text": "part two"}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": 8,
            "candidatesTokenCount": 6,
            "totalTokenCount": 14,
        },
    }
    text, usage = adapter.parse_response(200, payload)
    assert text == "part one part two"
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        8,
        6,
        14,
    )


def test_gemini_parse_empty_raises():
    adapter = GeminiAdapter()
    payload = {"candidates": [{"content": {"parts": [{"text": ""}]}}]}
    with pytest.raises(ProviderError, match="empty response"):
        adapter.parse_response(200, payload)


def test_gemini_parse_malformed_raises():
    adapter = GeminiAdapter()
    with pytest.raises(ProviderError, match="malformed response"):
        adapter.parse_response(200, {"candidates": []})


def test_gemini_parse_error_status_raises():
    adapter = GeminiAdapter()
    payload = {"error": {"status": "PERMISSION_DENIED", "message": "no access"}}
    with pytest.raises(ProviderError, match="HTTP 403"):
        adapter.parse_response(403, payload)
