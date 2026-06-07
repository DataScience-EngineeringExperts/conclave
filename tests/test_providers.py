"""Tests for the provider highway: registry, end-to-end call_model, redaction.

* ``resolve_adapter`` mapping incl. per-provider URLs, custom OpenAI-compatible
  endpoints, and the unknown-prefix error.
* End-to-end ``call_model`` with ``conclave.transport.post_json`` patched, proving
  text + usage extraction and that a transport error / missing key / unknown
  provider each become a non-raising ``ModelAnswer.error``.
* ``redact`` scrubbing a bearer/sk-token out of an error string.

Per-adapter ``build_request`` / ``parse_response`` tests live in
``test_adapters.py``.
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
