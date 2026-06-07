"""Tests for the provider highway: adapters, registry, transport, and redaction.

These tests exercise the owned httpx-based provider layer that replaced LiteLLM:

* Per-adapter ``build_request`` (URL, exact auth header, body shape incl. the
  Anthropic top-level ``system`` hoist + required ``max_tokens`` and the Gemini
  role mapping + ``systemInstruction``).
* Per-adapter ``parse_response`` over realistic recorded payloads: a success, a
  malformed/empty body, and a non-2xx status (-> ``ProviderError``).
* ``resolve_adapter`` mapping incl. custom OpenAI-compatible endpoints and the
  unknown-prefix error.
* End-to-end ``call_model`` with ``conclave.transport.post_json`` patched, proving
  text + usage extraction and that a transport error becomes a non-raising
  ``ModelAnswer.error``.
* ``redact`` scrubbing a bearer/sk-token out of an error string.
"""

from __future__ import annotations

import pytest

from conclave.adapters import ProviderError, resolve_adapter
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.base import redact
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter
from conclave.config import ConclaveConfig, CustomEndpoint
from conclave.providers import call_model


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


# --------------------------------------------------------------------------- #
# Adapter registry
# --------------------------------------------------------------------------- #


def test_resolve_adapter_built_in_prefixes():
    assert isinstance(resolve_adapter("openai/gpt-4.1"), OpenAICompatAdapter)
    assert isinstance(resolve_adapter("xai/grok-4.3"), OpenAICompatAdapter)
    assert isinstance(resolve_adapter("perplexity/sonar-pro"), OpenAICompatAdapter)
    assert isinstance(
        resolve_adapter("anthropic/claude-sonnet-4-6"), AnthropicAdapter
    )
    assert isinstance(resolve_adapter("gemini/gemini-2.5-pro"), GeminiAdapter)


def test_resolve_adapter_per_provider_urls():
    assert (
        resolve_adapter("xai/grok-4.3").completions_url
        == "https://api.x.ai/v1/chat/completions"
    )
    # Perplexity has NO /v1 segment.
    assert (
        resolve_adapter("perplexity/sonar-pro").completions_url
        == "https://api.perplexity.ai/chat/completions"
    )


def test_resolve_adapter_custom_endpoint_from_config():
    config = ConclaveConfig(
        endpoints={
            "together": CustomEndpoint(
                completions_url="https://api.together.xyz/v1/chat/completions",
                env_var="TOGETHER_API_KEY",
            )
        }
    )
    adapter = resolve_adapter("together/some-model", config)
    assert isinstance(adapter, OpenAICompatAdapter)
    assert adapter.completions_url == "https://api.together.xyz/v1/chat/completions"
    assert adapter.env_vars == ("TOGETHER_API_KEY",)


def test_resolve_adapter_unknown_prefix_raises():
    with pytest.raises(ProviderError, match="unknown provider 'mystery'"):
        resolve_adapter("mystery/model")


# --------------------------------------------------------------------------- #
# call_model end-to-end with transport patched
# --------------------------------------------------------------------------- #


async def test_call_model_success_via_patched_transport(monkeypatch):
    """A provider-shaped payload yields the right text + usage on ModelAnswer."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}

    async def fake_post_json(url, headers, json_body, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json_body
        return 200, {
            "choices": [{"message": {"content": "hello from openai"}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr("conclave.transport.post_json", fake_post_json)

    answer = await call_model(
        "openai",
        "openai/gpt-4.1",
        [{"role": "user", "content": "hi"}],
    )
    assert answer.ok
    assert answer.answer == "hello from openai"
    assert answer.usage is not None
    assert answer.usage.total_tokens == 5
    assert answer.error is None
    # The real adapter built the request that reached the transport.
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


async def test_call_model_transport_error_becomes_model_answer_error(monkeypatch):
    """A raised transport error is captured as a non-raising ModelAnswer.error."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    from conclave.transport import TransportError

    async def boom(url, headers, json_body, timeout):
        raise TransportError("request timed out after 120s")

    monkeypatch.setattr("conclave.transport.post_json", boom)

    answer = await call_model(
        "openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}]
    )
    assert not answer.ok
    assert answer.answer is None
    assert "timed out" in answer.error


async def test_call_model_missing_key_is_error(monkeypatch):
    """No key in env -> a clean ModelAnswer.error naming the env var, never raises."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    answer = await call_model(
        "openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}]
    )
    assert not answer.ok
    assert "OPENAI_API_KEY" in answer.error


async def test_call_model_unknown_provider_is_error(monkeypatch):
    """An unknown provider prefix surfaces as a helpful, non-raising error."""
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    answer = await call_model(
        "mystery", "mystery/model", [{"role": "user", "content": "hi"}]
    )
    assert not answer.ok
    assert "unknown provider 'mystery'" in answer.error


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_redact_scrubs_bearer_and_sk_token():
    leaked = "auth failed for Authorization: Bearer sk-abc123DEF456ghi789"
    cleaned = redact(leaked)
    assert "sk-abc123DEF456ghi789" not in cleaned
    assert "[REDACTED]" in cleaned


def test_redact_scrubs_env_var_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "supersecretvalue123")
    leaked = "request to openai with key supersecretvalue123 was rejected"
    cleaned = redact(leaked)
    assert "supersecretvalue123" not in cleaned
    assert "[REDACTED]" in cleaned


def test_redact_scrubs_x_api_key_header_echo():
    leaked = "headers were x-api-key: sk-ant-aabbccddeeff and version 2023-06-01"
    cleaned = redact(leaked)
    assert "sk-ant-aabbccddeeff" not in cleaned
    assert "[REDACTED]" in cleaned


def test_provider_error_message_is_pre_redacted():
    err = ProviderError("openai: HTTP 401: Bearer sk-leakedTOKEN12345")
    assert "sk-leakedTOKEN12345" not in str(err)
    assert "[REDACTED]" in str(err)
