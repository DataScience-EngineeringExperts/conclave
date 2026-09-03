"""Typed failure categories are derived from status codes / exception types (DSE-1512)."""

from __future__ import annotations

import httpx
import pytest

from conclave import transport
from conclave.adapters import ProviderError, resolve_adapter
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter
from conclave.config import ConclaveConfig
from conclave.models import FAILOVER_CATEGORIES, categorize_http_status


def _openai_adapter() -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        prefix="openai",
        completions_url="https://api.openai.com/v1/chat/completions",
        env_vars=("OPENAI_API_KEY",),
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "auth"),
        (403, "auth"),
        (402, "quota"),
        (429, "quota"),
        (408, "timeout"),
        (500, "unavailable"),
        (502, "unavailable"),
        (503, "unavailable"),
        (529, "unavailable"),
        (400, "bad_request"),
        (404, "bad_request"),
        (422, "bad_request"),
    ],
)
def test_categorize_http_status(status, expected):
    assert categorize_http_status(status) == expected


def test_failover_set_is_infrastructure_only():
    assert FAILOVER_CATEGORIES == frozenset(
        {"unkeyed", "unresolved", "auth", "quota", "unavailable", "timeout", "transport"}
    )
    assert "bad_request" not in FAILOVER_CATEGORIES
    assert "malformed_response" not in FAILOVER_CATEGORIES
    assert "unexpected" not in FAILOVER_CATEGORIES


def test_provider_error_defaults_to_malformed_response():
    err = ProviderError("x: empty response")
    assert err.category == "malformed_response"
    assert err.http_status is None


def test_provider_error_carries_status_category():
    err = ProviderError("x: HTTP 429: slow down", category="quota", http_status=429)
    assert err.category == "quota"
    assert err.http_status == 429
    # message is still redacted on construction (existing contract)
    assert "sk-" not in str(ProviderError("leak sk-abc123def456ghi789", category="auth"))


def test_transport_error_category():
    assert (
        transport.TransportError("request timed out after 5s", category="timeout").category
        == "timeout"
    )
    assert transport.TransportError("network error: ConnectError").category == "transport"


@pytest.mark.parametrize("adapter", [_openai_adapter(), AnthropicAdapter(), GeminiAdapter()])
def test_adapters_type_non_2xx(adapter):
    with pytest.raises(ProviderError) as info:
        adapter.parse_response(401, {"error": {"message": "bad key"}})
    assert info.value.category == "auth"
    assert info.value.http_status == 401
    with pytest.raises(ProviderError) as info:
        adapter.parse_response(503, {"error": {"message": "down"}})
    assert info.value.category == "unavailable"


def test_adapter_malformed_is_not_failover():
    with pytest.raises(ProviderError) as info:
        _openai_adapter().parse_response(200, {"choices": []})
    assert info.value.category == "malformed_response"
    assert info.value.category not in FAILOVER_CATEGORIES


def test_unresolved_provider_is_typed():
    with pytest.raises(ProviderError) as info:
        resolve_adapter("nope/model", ConclaveConfig())
    assert info.value.category == "unresolved"


async def test_post_json_timeout_is_typed(monkeypatch):
    class _Client:
        is_closed = False

        async def post(self, *a, **k):
            raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(transport, "_client", _Client())
    with pytest.raises(transport.TransportError) as info:
        await transport.post_json("https://x", {}, {}, 1.0)
    assert info.value.category == "timeout"


async def test_post_json_network_is_typed(monkeypatch):
    class _Client:
        is_closed = False

        async def post(self, *a, **k):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(transport, "_client", _Client())
    with pytest.raises(transport.TransportError) as info:
        await transport.post_json("https://x", {}, {}, 1.0)
    assert info.value.category == "transport"


def test_transport_error_carries_http_status():
    err = transport.TransportError("HTTP 503: x", category="unavailable", http_status=503)
    assert err.http_status == 503
    assert transport.TransportError("network error: ConnectError").http_status is None
