"""Structured-output request shaping for the OpenAI-compatible adapter (CAC-02-OAI).

These tests pin the capability-gated translation of an :class:`OutputContract`
into the OpenAI ``response_format`` surface. They assert the exact body shape for
each capability tier and the backward-compatibility invariant (no contract ->
byte-for-byte unchanged body). Both ``build_request`` and ``stream_request`` are
covered because ``response_format`` is documented as compatible with
``stream: true`` and the two paths must agree.

Capability tiers exercised, driven by
:func:`conclave.provider_catalog.capabilities_for`:

* structured-capable model (openai/gpt-4.1, xai/grok-4.3) -> ``json_schema``.
* json_mode-only model (deepseek/deepseek-chat) -> ``json_object`` + warning.
* unsupported model (perplexity/sonar-pro) -> no injection + warning.
* unknown / custom endpoint (no capability record) -> no injection + warning.

Free-prose / parse behavior and the non-structured body shape live in
``test_adapters.py`` (which this module must not disturb).
"""

from __future__ import annotations

import logging

import pytest

from conclave.adapters.base import OutputContract
from conclave.adapters.openai_compat import OpenAICompatAdapter

# A representative schema; the adapter passes it through opaquely.
_SCHEMA: dict = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

_MESSAGES = [{"role": "user", "content": "decide"}]


def _adapter(prefix: str, url: str, env: str) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(prefix=prefix, completions_url=url, env_vars=(env,))


def _openai() -> OpenAICompatAdapter:
    return _adapter("openai", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY")


def _xai() -> OpenAICompatAdapter:
    return _adapter("xai", "https://api.x.ai/v1/chat/completions", "XAI_API_KEY")


def _deepseek() -> OpenAICompatAdapter:
    return _adapter("deepseek", "https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY")


def _perplexity() -> OpenAICompatAdapter:
    return _adapter(
        "perplexity", "https://api.perplexity.ai/chat/completions", "PERPLEXITY_API_KEY"
    )


def _build(adapter: OpenAICompatAdapter, model_id: str, contract: OutputContract | None) -> dict:
    _url, _headers, body = adapter.build_request(
        model_id, _MESSAGES, 0.5, 120.0, "sk-secret", contract
    )
    return body


def _stream(adapter: OpenAICompatAdapter, model_id: str, contract: OutputContract | None) -> dict:
    _url, _headers, body = adapter.stream_request(
        model_id, _MESSAGES, 0.5, 120.0, "sk-secret", contract
    )
    return body


# --------------------------------------------------------------------------- #
# Tier 1: structured-output capable -> json_schema response_format
# --------------------------------------------------------------------------- #


def test_json_schema_injected_for_structured_capable_openai():
    body = _build(
        _openai(), "openai/gpt-4.1", OutputContract(schema=_SCHEMA, schema_name="verdict")
    )
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "strict": False, "schema": _SCHEMA},
    }
    # The wire model id stays bare even though the catalog lookup used the full id.
    assert body["model"] == "gpt-4.1"


def test_json_schema_injected_for_structured_capable_xai():
    body = _build(_xai(), "xai/grok-4.3", OutputContract(schema=_SCHEMA, schema_name="member"))
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "member", "strict": False, "schema": _SCHEMA},
    }


def test_json_schema_default_name_when_schema_name_absent():
    body = _build(_openai(), "openai/gpt-4.1", OutputContract(schema=_SCHEMA))
    assert body["response_format"]["json_schema"]["name"] == "verdict"


def test_json_schema_strict_honored_true():
    body = _build(
        _openai(),
        "openai/gpt-4.1",
        OutputContract(schema=_SCHEMA, schema_name="verdict", strict=True),
    )
    assert body["response_format"]["json_schema"]["strict"] is True


def test_json_schema_strict_honored_false_default():
    body = _build(_openai(), "openai/gpt-4.1", OutputContract(schema=_SCHEMA))
    assert body["response_format"]["json_schema"]["strict"] is False


def test_json_schema_omits_schema_key_when_contract_schema_is_none():
    # A contract may carry name/strict without a concrete schema dict; we must
    # not emit ``"schema": null`` (OpenAI rejects a null schema).
    body = _build(_openai(), "openai/gpt-4.1", OutputContract(schema=None, strict=True))
    json_schema = body["response_format"]["json_schema"]
    assert "schema" not in json_schema
    assert json_schema == {"name": "verdict", "strict": True}


# --------------------------------------------------------------------------- #
# Tier 2: json_mode only -> json_object + warning
# --------------------------------------------------------------------------- #


def test_json_object_for_json_mode_only_model(caplog):
    # deepseek/deepseek-chat: json_mode True, structured_output False (catalog).
    with caplog.at_level(logging.WARNING, logger="conclave"):
        body = _build(_deepseek(), "deepseek/deepseek-chat", OutputContract(schema=_SCHEMA))
    assert body["response_format"] == {"type": "json_object"}
    assert any(
        "json mode only" in rec.getMessage().lower() or "json_object" in rec.getMessage().lower()
        for rec in caplog.records
    )


# --------------------------------------------------------------------------- #
# Tier 3: unsupported model -> no injection + warning
# --------------------------------------------------------------------------- #


def test_no_injection_for_unsupported_model(caplog):
    # perplexity/sonar-pro: both flags False in the catalog.
    with caplog.at_level(logging.WARNING, logger="conclave"):
        body = _build(_perplexity(), "perplexity/sonar-pro", OutputContract(schema=_SCHEMA))
    assert "response_format" not in body
    assert any("neither" in rec.getMessage().lower() for rec in caplog.records)


def test_no_injection_for_unknown_endpoint(caplog):
    # An id whose prefix is absent from the catalog -> capabilities_for() None.
    adapter = _adapter("custom", "https://llm.internal.example/v1/chat/completions", "CUSTOM_KEY")
    with caplog.at_level(logging.WARNING, logger="conclave"):
        body = _build(adapter, "custom/private-model", OutputContract(schema=_SCHEMA))
    assert "response_format" not in body
    assert any("no capability record" in rec.getMessage().lower() for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Backward compatibility: no contract -> body byte-for-byte unchanged
# --------------------------------------------------------------------------- #


def test_no_contract_leaves_build_body_unchanged():
    adapter = _openai()
    body = _build(adapter, "openai/gpt-4.1", None)
    assert body == {
        "model": "gpt-4.1",
        "messages": _MESSAGES,
        "temperature": 0.5,
    }
    assert "response_format" not in body


def test_no_contract_leaves_stream_body_unchanged():
    adapter = _openai()
    body = _stream(adapter, "openai/gpt-4.1", None)
    # Identical to the non-structured build body plus only the stream flags.
    assert body == {
        "model": "gpt-4.1",
        "messages": _MESSAGES,
        "temperature": 0.5,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert "response_format" not in body


# --------------------------------------------------------------------------- #
# stream_request applies the same gating as build_request
# --------------------------------------------------------------------------- #


def test_stream_request_injects_json_schema_for_structured_model():
    body = _stream(_openai(), "openai/gpt-4.1", OutputContract(schema=_SCHEMA, strict=True))
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "strict": True, "schema": _SCHEMA},
    }
    # Stream flags layered on top, not clobbered by the contract.
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_stream_request_json_object_for_json_mode_only_model():
    body = _stream(_deepseek(), "deepseek/deepseek-chat", OutputContract(schema=_SCHEMA))
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is True


def test_stream_and_build_response_format_agree():
    contract = OutputContract(schema=_SCHEMA, schema_name="verdict", strict=True)
    build_body = _build(_openai(), "openai/gpt-4.1", contract)
    stream_body = _stream(_openai(), "openai/gpt-4.1", contract)
    assert build_body["response_format"] == stream_body["response_format"]


# --------------------------------------------------------------------------- #
# Never raises on a contract for any tier (council-non-abort invariant)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model_id",
    [
        "openai/gpt-4.1",  # structured
        "deepseek/deepseek-chat",  # json_mode only
        "perplexity/sonar-pro",  # unsupported
        "custom/unknown-model",  # no capability record
    ],
)
def test_contract_never_raises_across_tiers(model_id):
    adapter = _adapter("custom", "https://x.example/v1/chat/completions", "CUSTOM_KEY")
    # Must not raise regardless of capability tier; degrades gracefully.
    adapter.build_request(
        model_id, _MESSAGES, 0.5, 120.0, "sk-secret", OutputContract(schema=_SCHEMA)
    )
    adapter.stream_request(
        model_id, _MESSAGES, 0.5, 120.0, "sk-secret", OutputContract(schema=_SCHEMA)
    )
