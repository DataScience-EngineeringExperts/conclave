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

import conclave.config as config_mod
import conclave.providers as providers_mod
from conclave.adapters import ProviderError, resolve_adapter
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.base import redact
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter
from conclave.config import ConclaveConfig, CustomEndpoint, clear_config_cache
from conclave.providers import call_model


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Each test starts and ends with an empty config memo for isolation."""
    clear_config_cache()
    yield
    clear_config_cache()


# --------------------------------------------------------------------------- #
# Adapter registry
# --------------------------------------------------------------------------- #


def test_resolve_adapter_built_in_prefixes():
    assert isinstance(resolve_adapter("openai/gpt-4.1"), OpenAICompatAdapter)
    assert isinstance(resolve_adapter("xai/grok-4.3"), OpenAICompatAdapter)
    assert isinstance(resolve_adapter("perplexity/sonar-pro"), OpenAICompatAdapter)
    assert isinstance(resolve_adapter("anthropic/claude-sonnet-4-6"), AnthropicAdapter)
    assert isinstance(resolve_adapter("gemini/gemini-2.5-pro"), GeminiAdapter)


def test_resolve_adapter_per_provider_urls():
    assert resolve_adapter("xai/grok-4.3").completions_url == "https://api.x.ai/v1/chat/completions"
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

    answer = await call_model("openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}])
    assert not answer.ok
    assert answer.answer is None
    assert "timed out" in answer.error


async def test_call_model_missing_key_is_error(monkeypatch):
    """No key in env -> a clean ModelAnswer.error naming the env var, never raises."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    answer = await call_model("openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}])
    assert not answer.ok
    assert "OPENAI_API_KEY" in answer.error


async def test_call_model_unknown_provider_is_error(monkeypatch):
    """An unknown provider prefix surfaces as a helpful, non-raising error."""
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    answer = await call_model("mystery", "mystery/model", [{"role": "user", "content": "hi"}])
    assert not answer.ok
    assert "unknown provider 'mystery'" in answer.error


async def test_call_model_custom_endpoint_key_not_leaked_in_error(monkeypatch, tmp_path):
    """A custom-endpoint key value echoed in a provider error is scrubbed (issue #14).

    Repro: declare a custom OpenAI-compatible endpoint whose api_key_env is NOT
    in PROVIDER_ENV_VARS and whose value has no recognized prefix, then have the
    mocked transport return a 401 whose error message echoes that key. The
    resulting ModelAnswer.error must not contain the key value anywhere.
    """
    # Obviously-synthetic, unprefixed fake key -- no sk-/xai-/pplx-/AIza shape,
    # so pattern-based scrubbing alone would miss it; only name-based scrubbing
    # via the custom endpoint's env var saves it.
    fake_key = "togetherFAKEsecret_unprefixed_0123456789"
    monkeypatch.setenv("TOGETHER_API_KEY", fake_key)

    config_file = tmp_path / "conclave.yml"
    config_file.write_text(
        "endpoints:\n"
        "  together:\n"
        "    completions_url: https://api.together.xyz/v1/chat/completions\n"
        "    env_var: TOGETHER_API_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONCLAVE_CONFIG", str(config_file))

    async def echoing_401(url, headers, json_body, timeout):
        # Simulate a gateway that echoes the submitted credential on auth failure.
        return 401, {"error": {"message": f"invalid api key: {fake_key}"}}

    monkeypatch.setattr("conclave.transport.post_json", echoing_401)

    answer = await call_model(
        "together",
        "together/some-model",
        [{"role": "user", "content": "hi"}],
    )
    assert not answer.ok
    assert answer.error is not None
    assert fake_key not in answer.error
    assert "[REDACTED]" in answer.error


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


def test_redact_scrubs_custom_endpoint_env_var_value(monkeypatch, tmp_path):
    """An unprefixed custom-endpoint key value is scrubbed via config (issue #14).

    The key has no recognized provider prefix, so only name-based scrubbing
    sourced from config.endpoints[*].env_var can catch it.
    """
    fake_key = "togetherFAKEsecret_unprefixed_0123456789"
    monkeypatch.setenv("TOGETHER_API_KEY", fake_key)

    config_file = tmp_path / "conclave.yml"
    config_file.write_text(
        "endpoints:\n"
        "  together:\n"
        "    completions_url: https://api.together.xyz/v1/chat/completions\n"
        "    env_var: TOGETHER_API_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONCLAVE_CONFIG", str(config_file))

    cleaned = redact(f"auth failed: invalid api key: {fake_key}")
    assert fake_key not in cleaned
    assert "[REDACTED]" in cleaned


# --------------------------------------------------------------------------- #
# Config is resolved once / injectable, not re-read per call (issue #15)
# --------------------------------------------------------------------------- #


async def _ok_post(url, headers, json_body, timeout):
    """A minimal successful transport stub for hot-path tests."""
    return 200, {"choices": [{"message": {"content": "ok"}}]}


async def test_call_model_uses_injected_config_without_load(monkeypatch):
    """When a config is injected, call_model never calls load_config (issue #15)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("conclave.transport.post_json", _ok_post)

    calls = {"n": 0}

    def spy_load_config(*args, **kwargs):  # pragma: no cover - must not run
        calls["n"] += 1
        return ConclaveConfig()

    monkeypatch.setattr(providers_mod, "load_config", spy_load_config)

    injected = ConclaveConfig()
    answer = await call_model(
        "openai",
        "openai/gpt-4.1",
        [{"role": "user", "content": "hi"}],
        config=injected,
    )
    assert answer.ok
    assert calls["n"] == 0, "load_config must not be called when config is injected"


async def test_call_model_standalone_reads_disk_at_most_once(monkeypatch, tmp_path):
    """Repeated standalone calls re-read the config file at most once (issue #15).

    The memoized loader means the disk read + YAML parse happens once even across
    many call_model invocations for the same unchanged file -- the hot-path /
    caching blocker the issue describes.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("conclave.transport.post_json", _ok_post)

    config_file = tmp_path / "conclave.yml"
    config_file.write_text("synthesizer: openai\n", encoding="utf-8")
    monkeypatch.setenv("CONCLAVE_CONFIG", str(config_file))

    reads = {"n": 0}
    real_read_yaml = config_mod._read_yaml

    def counting_read_yaml(path):
        reads["n"] += 1
        return real_read_yaml(path)

    monkeypatch.setattr(config_mod, "_read_yaml", counting_read_yaml)

    # Simulate a 4-member x 3-round debate + synthesis worth of calls.
    for _ in range(13):
        answer = await call_model(
            "openai",
            "openai/gpt-4.1",
            [{"role": "user", "content": "hi"}],
        )
        assert answer.ok

    assert reads["n"] == 1, f"expected a single disk read, got {reads['n']}"


def test_config_cache_invalidates_on_file_change(tmp_path):
    """The memo self-invalidates when the config file's mtime changes."""
    import os
    import time

    from conclave.config import load_config

    clear_config_cache()
    config_file = tmp_path / "conclave.yml"
    config_file.write_text("synthesizer: openai\n", encoding="utf-8")

    first = load_config(path=config_file)
    assert first.synthesizer == "openai"

    # Rewrite with a new value and bump mtime so the key changes.
    config_file.write_text("synthesizer: gemini\n", encoding="utf-8")
    os.utime(config_file, (time.time() + 5, time.time() + 5))

    second = load_config(path=config_file)
    assert second.synthesizer == "gemini"
