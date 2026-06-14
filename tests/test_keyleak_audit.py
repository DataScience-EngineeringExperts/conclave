"""Key-leak audit regression suite (conclave v1.0, SECURITY.md threat model).

These tests back conclave's headline **bring-your-own-keys / key-rigor** claim.
Each test maps to one vector of the key-leak attack map in SECURITY.md's threat
model and the v1.0 readiness review. They plant an OBVIOUSLY-FAKE, key-shaped
secret somewhere a leak could occur and assert it never escapes -- so a security
reviewer (or a future refactor) cannot quietly break the contract.

All tests run offline. Planted secrets use synthetic ``...FAKE...`` patterns that
gitleaks will not flag (see ``.gitleaks.toml``); no real credential is ever used.

Vector map
==========
* **V1 cache write path** -- a key-shaped secret echoed in a provider error never
  reaches a cache file, the cache key, or the cache filename (cache stores the
  already-redacted ``CouncilResult``).
* **V2 streaming chunk path** -- a secret planted in a mid-stream provider error is
  absent from every streamed ``StreamEvent`` AND from the final ``ModelAnswer``.
* **V3 __repr__/__str__** -- no config/adapter/result object renders a planted key
  in its ``repr``/``str``; the key-holding request ``headers`` dict is built only
  inside the adapter and never stored on any object.
* **V4 provider 400/422 echo** -- a buffered HTTP error whose body echoes the
  submitted key is scrubbed in ``ModelAnswer.error`` (capture runs through
  ``redact()``).
* **V5 transport debug logging** -- the ``guard_transport_logging`` helper blocks
  httpx/httpcore DEBUG records (the only level that emits auth headers).
* **V8 transport cause-chain (RANK 1/5)** -- a ``TransportError`` surfaced from
  ``post_json``/``stream_sse`` retains no reference to the underlying httpx
  exception, whose ``.request.headers`` carries the live auth header; the chain is
  dropped (``raise ... from None``) so the key cannot leak via
  ``traceback.format_exception`` or a cause-chain repr.
* **V9 transport guard default-on (RANK 6)** -- ``Council.__init__`` installs the
  httpx/httpcore DEBUG guard by default; ``allow_transport_debug_logging=True``
  opts out.
* **V10 client close hygiene (RANK 8)** -- the pooled client closes cleanly with no
  ``ResourceWarning`` (cheap close-wiring regression guard).
"""

from __future__ import annotations

import json
import logging
import traceback
import warnings

import httpx
import pytest

from conclave import Council, guard_transport_logging, transport
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter
from conclave.config import ConclaveConfig, CustomEndpoint, clear_config_cache
from conclave.models import ModelAnswer
from conclave.providers import call_model, call_model_stream

# An obviously-fake, key-SHAPED secret. The ``sk-`` prefix makes it match
# redact()'s pattern path; ``FAKE`` makes it unmistakably synthetic to a human
# reader and to gitleaks (allowlisted). If this string ever appears in a cache
# file, a streamed event, a repr, or a result error, a leak has occurred.
PLANTED = "sk-FAKEconclaveLEAK0123456789abcdefSECRET"


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Isolate each test from the in-process config memo."""
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
async def mock_stream_client():
    """Install a MockTransport-backed pooled client; restore the global after.

    Mirrors the async fixture in ``test_streaming.py`` so the real
    ``call_model_stream`` -> ``transport.stream_sse`` -> adapter path runs against
    a caller-supplied handler with no network.
    """
    saved = transport._client
    created: list[httpx.AsyncClient] = []

    def use(handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        created.append(client)
        transport._client = client
        return client

    yield use

    for client in created:
        if not client.is_closed:
            await client.aclose()
    transport._client = saved


# --------------------------------------------------------------------------- #
# V1 -- cache write path: a planted secret never reaches the cache (HIGH)
# --------------------------------------------------------------------------- #


def _cache_config(cache: bool = True) -> ConclaveConfig:
    """A deterministic, on-disk-independent config for cache tests."""
    return ConclaveConfig(
        models={"grok": "xai/grok-4.3", "claude": "anthropic/claude-sonnet-4-6"},
        councils={"default": ["grok", "claude"]},
        synthesizer="claude",
        cache=cache,
    )


async def test_cache_never_persists_planted_secret_from_provider_error(monkeypatch, tmp_path):
    """V1: a key echoed in a provider error must not land in any cache artifact.

    End-to-end: run a cached council where every member's mocked transport returns
    a 401 whose error body echoes the planted key. The member errors are captured
    via redact() before they reach the CouncilResult, so the on-disk cache entry
    (and its filename and the cache key) must contain neither the secret value nor
    the env var name -- proving the cache write happens strictly post-redaction.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # The planted secret IS the live key value, so redact() can mask it by value
    # even though one member's error also echoes it inline.
    for var in ("XAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, PLANTED)

    async def echoing_401(url, headers, json_body, timeout):
        # A gateway that echoes the submitted credential back on auth failure.
        return 401, {"error": {"message": f"invalid api key: {PLANTED}"}}

    monkeypatch.setattr("conclave.transport.post_json", echoing_401)

    council = Council(
        models=["grok", "claude"], synthesizer="claude", config=_cache_config(), cache=True
    )
    result = await council.ask("audit prompt", synthesize=True)

    # Every member failed -> each error must already be redacted on the result.
    assert result.answers, "expected attempted members"
    for ans in result.answers:
        assert ans.error is not None
        assert PLANTED not in ans.error
        assert "[REDACTED]" in ans.error

    cache_home = tmp_path / "conclave"
    entries = list(cache_home.glob("*.json"))
    assert entries, "a cached entry should have been written"
    for entry in entries:
        blob = entry.read_text(encoding="utf-8")
        assert PLANTED not in blob, "planted secret leaked into a cache file"
        assert "XAI_API_KEY" not in blob
        assert "ANTHROPIC_API_KEY" not in blob
        # The filename (= cache key) must carry no secret either.
        assert PLANTED not in entry.name

    # And the computed cache key itself is secret-free.
    key = council._cache_key("audit prompt", "synthesize")
    assert PLANTED not in key


async def test_cache_key_and_payload_have_no_env_value(monkeypatch, tmp_path):
    """V1: even a SUCCESSFUL run never writes the key value or name to the cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for var in ("XAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, PLANTED)

    async def ok_post(url, headers, json_body, timeout):
        return 200, {"choices": [{"message": {"content": "benign answer"}}]}

    monkeypatch.setattr("conclave.transport.post_json", ok_post)

    council = Council(models=["grok"], synthesizer="claude", config=_cache_config(), cache=True)
    await council.ask("benign prompt", synthesize=False)

    cache_home = tmp_path / "conclave"
    for entry in cache_home.glob("*.json"):
        blob = entry.read_text(encoding="utf-8")
        assert PLANTED not in blob
        assert "XAI_API_KEY" not in blob


# --------------------------------------------------------------------------- #
# V2 -- streaming chunk path: planted secret absent from stream AND final (MED)
# --------------------------------------------------------------------------- #


async def _collect_stream(name, model_id, **kwargs):
    """Run call_model_stream, returning (text_chunks, final_ModelAnswer)."""
    chunks: list[str] = []
    final: ModelAnswer | None = None
    async for item in call_model_stream(
        name, model_id, [{"role": "user", "content": "hi"}], **kwargs
    ):
        if isinstance(item, ModelAnswer):
            final = item
        else:
            chunks.append(item)
    return chunks, final


def _sse(*frames: str) -> bytes:
    return ("".join(f"{frame}\n\n" for frame in frames)).encode("utf-8")


async def test_stream_planted_secret_absent_from_chunks_and_final(monkeypatch, mock_stream_client):
    """V2: a mid-stream error echoing the key leaks into no chunk and no final.

    The provider streams a couple of good deltas, then returns a non-2xx whose
    body echoes the planted key... but here we drive it via a 401 status so the
    transport raises before any delta. We separately assert the partial-text case
    below. This case proves: zero streamed chunks carry the secret, and the final
    ModelAnswer.error is redacted.
    """
    monkeypatch.setenv("OPENAI_API_KEY", PLANTED)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {PLANTED}"}})

    mock_stream_client(handler)
    chunks, final = await _collect_stream("openai", "openai/gpt-4.1")

    # No streamed chunk may carry the secret (error path yields no text deltas).
    for chunk in chunks:
        assert PLANTED not in chunk
    assert final is not None and not final.ok
    assert final.error is not None
    assert PLANTED not in final.error
    assert "[REDACTED]" in final.error


async def test_stream_midstream_error_after_partial_text_is_redacted(
    monkeypatch, mock_stream_client
):
    """V2: a structured error frame arriving mid-stream (after real deltas) leaks nothing.

    A provider can send good content deltas and THEN a structured error data frame
    that echoes the key. The good deltas must stream (they are answer content), the
    error must be captured + redacted on the final answer, the partial text must be
    preserved, and the planted key must appear in NO streamed event NOR the final.
    """
    monkeypatch.setenv("OPENAI_API_KEY", PLANTED)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse(
            'data: {"choices":[{"delta":{"content":"partial "}}]}',
            # A structured error frame mid-stream that echoes the credential.
            'data: {"error":{"message":"auth rejected: ' + PLANTED + '"}}',
            "data: [DONE]",
        )
        return httpx.Response(200, content=body)

    mock_stream_client(handler)
    chunks, final = await _collect_stream("openai", "openai/gpt-4.1")

    # The good content streamed; no chunk carries the secret.
    assert chunks == ["partial "]
    for chunk in chunks:
        assert PLANTED not in chunk
    assert final is not None and not final.ok
    assert final.error is not None
    assert PLANTED not in final.error
    assert "[REDACTED]" in final.error
    # Partial text preserved AND clean.
    assert (final.answer or "") == "partial "
    assert PLANTED not in (final.answer or "")


async def test_council_stream_events_carry_no_planted_secret(monkeypatch, mock_stream_client):
    """V2 at the council level: no StreamEvent in a full run carries the secret."""
    monkeypatch.setenv("OPENAI_API_KEY", PLANTED)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {PLANTED}"}})

    mock_stream_client(handler)

    config = ConclaveConfig(
        models={"openai": "openai/gpt-4.1"},
        councils={"default": ["openai"]},
        synthesizer="openai",
    )
    council = Council(models=["openai"], synthesizer="openai", config=config)

    events = [e async for e in council.ask_stream("hi", synthesize=False)]
    # Serialize every event the way a consumer would and scan the whole payload.
    for ev in events:
        dumped = json.dumps(ev.model_dump(mode="json"))
        assert PLANTED not in dumped, f"planted secret leaked into a {ev.type} event"


# --------------------------------------------------------------------------- #
# V3 -- __repr__/__str__ never render key material (MED)
# --------------------------------------------------------------------------- #


def test_adapters_hold_no_key_and_repr_is_clean():
    """V3: adapters never store a key; building a request does not retain it.

    The key VALUE is passed to ``build_request`` per call and used only to compose
    the (transient) headers dict the caller hands to the transport. It is never
    assigned to ``self``. So an adapter instance's repr -- the thing that would
    surface in a traceback frame referencing ``self`` -- cannot contain a key.
    """
    adapters = [
        OpenAICompatAdapter(
            prefix="openai",
            completions_url="https://api.openai.com/v1/chat/completions",
            env_vars=("OPENAI_API_KEY",),
        ),
        AnthropicAdapter(),
        GeminiAdapter(),
    ]
    messages = [{"role": "user", "content": "hi"}]
    for adapter in adapters:
        url, headers, _body = adapter.build_request(
            f"{adapter.prefix}/some-model", messages, 0.7, 30.0, PLANTED
        )
        # The headers the transport receives DO carry the key (they must, to auth)
        # -- that is the in-flight request, redaction-exempt by design and handled
        # by the transport-logging guard (V5), not by repr.
        header_blob = json.dumps(headers)
        assert PLANTED in header_blob, "sanity: the live request must carry the key"
        # But the adapter object itself retains nothing.
        assert PLANTED not in repr(adapter)
        assert PLANTED not in str(adapter)
        assert PLANTED not in str(adapter.__dict__)


def test_config_repr_has_no_key_value(monkeypatch):
    """V3: ConclaveConfig holds only env var NAMES, never values; repr is clean."""
    monkeypatch.setenv("TOGETHER_API_KEY", PLANTED)
    config = ConclaveConfig(
        models={"x": "together/m"},
        endpoints={
            "together": CustomEndpoint(
                completions_url="https://api.together.xyz/v1/chat/completions",
                env_var="TOGETHER_API_KEY",  # NAME only
            )
        },
    )
    assert PLANTED not in repr(config)
    assert PLANTED not in str(config)
    # The name is fine to render; the value must never be present.
    assert "TOGETHER_API_KEY" in repr(config)


async def test_model_answer_repr_clean_after_provider_error(monkeypatch):
    """V3 (async): the redacted ModelAnswer's repr/str/json carry no secret."""
    monkeypatch.setenv("OPENAI_API_KEY", PLANTED)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    async def echoing_401(url, headers, json_body, timeout):
        return 401, {"error": {"message": f"invalid api key: {PLANTED}"}}

    monkeypatch.setattr("conclave.transport.post_json", echoing_401)

    answer = await call_model("openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}])
    assert not answer.ok
    assert PLANTED not in repr(answer)
    assert PLANTED not in str(answer)
    assert PLANTED not in json.dumps(answer.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# V4 -- provider 400/422 echo is captured AFTER redact() (MED)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [400, 401, 422, 500])
async def test_buffered_error_echoing_key_is_redacted(monkeypatch, status):
    """V4: a buffered HTTP error whose body echoes the key is scrubbed in .error."""
    monkeypatch.setenv("OPENAI_API_KEY", PLANTED)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    async def echoing_error(url, headers, json_body, timeout):
        # Provider echoes the submitted Authorization header back in its error.
        return status, {
            "error": {"message": f"request failed; headers: Authorization: Bearer {PLANTED}"}
        }

    monkeypatch.setattr("conclave.transport.post_json", echoing_error)

    answer = await call_model("openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}])
    assert not answer.ok
    assert answer.error is not None
    assert PLANTED not in answer.error
    assert "[REDACTED]" in answer.error
    assert str(status) in answer.error


async def test_custom_endpoint_unprefixed_key_echo_is_redacted(monkeypatch, tmp_path):
    """V4: an UNPREFIXED custom-endpoint key (no sk-/AIza shape) echoed in a 400 is scrubbed.

    Only name-based scrubbing (sourced from config.endpoints[*].env_var) can catch
    a key with no recognizable shape -- this guards the BYO-keys leak class for
    user-declared providers.
    """
    unshaped = "togetherFAKEsecret_unprefixed_no_known_shape_0123456789"
    monkeypatch.setenv("TOGETHER_API_KEY", unshaped)

    config_file = tmp_path / "conclave.yml"
    config_file.write_text(
        "endpoints:\n"
        "  together:\n"
        "    completions_url: https://api.together.xyz/v1/chat/completions\n"
        "    env_var: TOGETHER_API_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONCLAVE_CONFIG", str(config_file))

    async def echoing_400(url, headers, json_body, timeout):
        return 400, {"error": {"message": f"bad request, key was {unshaped}"}}

    monkeypatch.setattr("conclave.transport.post_json", echoing_400)

    answer = await call_model(
        "together", "together/some-model", [{"role": "user", "content": "hi"}]
    )
    assert not answer.ok
    assert answer.error is not None
    assert unshaped not in answer.error
    assert "[REDACTED]" in answer.error


# --------------------------------------------------------------------------- #
# V5 -- transport debug-logging guard blocks header-bearing DEBUG records (HIGH)
# --------------------------------------------------------------------------- #


def test_guard_transport_logging_blocks_debug_records(monkeypatch):
    """V5: after guard install, httpx/httpcore DEBUG records are dropped, INFO+ kept.

    httpcore logs request headers (incl. Authorization) only at DEBUG. The guard
    installs a filter that discards DEBUG records on the httpx/httpcore loggers, so
    a header value can never reach a handler -- while INFO+ diagnostics survive.
    """
    # Reset the one-shot guard flag so this test exercises a fresh install, then
    # restore it so we don't perturb global state for other tests.
    saved_flag = transport._GUARD_INSTALLED
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    saved_httpx_filters = httpx_logger.filters[:]
    saved_httpcore_filters = httpcore_logger.filters[:]
    monkeypatch.setattr(transport, "_GUARD_INSTALLED", False)
    try:
        guard_transport_logging()

        # A DEBUG record carrying a (fake) auth header must be filtered out.
        debug_rec = httpcore_logger.makeRecord(
            "httpcore",
            logging.DEBUG,
            __file__,
            0,
            "send_request_headers.complete return_value=[(b'Authorization', b'Bearer %s')]",
            (PLANTED.encode(),),
            None,
        )
        assert all(f.filter(debug_rec) is False for f in httpcore_logger.filters), (
            "DEBUG httpcore record (header-bearing) must be dropped by the guard"
        )

        # An INFO record on the same logger must survive (guard is DEBUG-only).
        info_rec = httpcore_logger.makeRecord(
            "httpcore", logging.INFO, __file__, 0, "connection established", (), None
        )
        guard_filter = next(
            f for f in httpcore_logger.filters if f.__class__.__name__ == "_NoDebugHeadersFilter"
        )
        assert guard_filter.filter(info_rec) is True

        # The httpx logger got the same guard.
        assert any(f.__class__.__name__ == "_NoDebugHeadersFilter" for f in httpx_logger.filters)

        # Idempotent: a second install does not stack a second filter.
        before = len(httpcore_logger.filters)
        guard_transport_logging()
        assert len(httpcore_logger.filters) == before
    finally:
        httpx_logger.filters = saved_httpx_filters
        httpcore_logger.filters = saved_httpcore_filters
        transport._GUARD_INSTALLED = saved_flag


def test_guard_transport_logging_is_exported():
    """V5: the guard is part of the public API so library consumers can call it."""
    import conclave

    assert hasattr(conclave, "guard_transport_logging")
    assert "guard_transport_logging" in conclave.__all__


# --------------------------------------------------------------------------- #
# V7 -- residual catch-all error construction is redacted (audit-found gap)
# --------------------------------------------------------------------------- #
#
# Not in the original attack map: the partial-failure catch-alls in
# Council.fan_out and streaming._drive_member build a ModelAnswer.error from a raw
# exception (`f"{type(exc).__name__}: {exc}"`). These only fire on an UNEXPECTED
# raise that escapes call_model / call_model_stream (which already redact), but the
# invariant "every surfaced error string is scrubbed" must hold even there. The
# fix wraps both in redact(); these tests pin it by forcing the underlying call to
# RAISE (not return a ModelAnswer) with the planted key in the message.


async def test_fan_out_catch_all_error_is_redacted(monkeypatch):
    """V7 (buffered): an unexpected raise in fan_out yields a redacted error."""
    monkeypatch.setenv("XAI_API_KEY", PLANTED)

    import conclave.council as council_mod

    async def raising_call_model(name, model_id, messages, *, temperature=0.7, timeout=120.0):
        # Simulate an unexpected escape carrying the key in its text.
        raise RuntimeError(f"unexpected boom leaking {PLANTED}")

    monkeypatch.setattr(council_mod, "call_model", raising_call_model)

    config = ConclaveConfig(
        models={"grok": "xai/grok-4.3"},
        councils={"default": ["grok"]},
        synthesizer="grok",
    )
    council = Council(models=["grok"], synthesizer="grok", config=config)
    result = await council.ask("hi", synthesize=False)

    assert result.answers and not result.answers[0].ok
    err = result.answers[0].error
    assert err is not None
    assert PLANTED not in err
    assert "[REDACTED]" in err


async def test_stream_drive_member_catch_all_error_is_redacted(monkeypatch):
    """V7 (streaming): an unexpected raise in _drive_member yields a redacted error."""
    monkeypatch.setenv("XAI_API_KEY", PLANTED)

    import conclave.streaming as streaming_mod

    async def raising_stream(
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None
    ):
        # An unexpected raise (not a yielded error ModelAnswer) carrying the key.
        raise RuntimeError(f"stream boom leaking {PLANTED}")
        yield  # pragma: no cover -- make this an async generator

    monkeypatch.setattr(streaming_mod, "call_model_stream", raising_stream)

    config = ConclaveConfig(
        models={"grok": "xai/grok-4.3"},
        councils={"default": ["grok"]},
        synthesizer="grok",
    )
    council = Council(models=["grok"], synthesizer="grok", config=config)

    events = [e async for e in council.ask_stream("hi", synthesize=False)]
    done = events[-1]
    assert done.type == "done"
    answers = done.result.answers
    assert answers and not answers[0].ok
    err = answers[0].error
    assert err is not None
    assert PLANTED not in err
    assert "[REDACTED]" in err
    # No StreamEvent anywhere carries the secret.
    for ev in events:
        assert PLANTED not in json.dumps(ev.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# V6 -- fixtures sanity: the planted secret is obviously fake
# --------------------------------------------------------------------------- #


def test_planted_secret_is_obviously_fake():
    """V6: the audit's planted secret is a synthetic, gitleaks-allowlisted token."""
    assert "FAKE" in PLANTED
    assert PLANTED.startswith("sk-FAKE")


# --------------------------------------------------------------------------- #
# V8 -- transport drops the httpx exception from the cause chain (RANK 1/5, HIGH)
# --------------------------------------------------------------------------- #
#
# httpx exceptions carry a live ``.request`` whose ``.headers`` hold the
# ``Authorization``/``x-api-key`` value. If the surfaced TransportError kept that
# exception as ``__cause__``/``__context__``, the key would sit one cause-chain hop
# away and leak under ``traceback.format_exception``, ``logging.exception``, or a
# cause-chain repr. The transport raises ``... from None``; these tests pin that the
# formatted traceback + repr/str of the surfaced error are key-free and the cause
# chain is dropped.


def _planted_header_request(url: str = "https://x/y") -> httpx.Request:
    """A real httpx.Request carrying the planted auth header (so .request.headers populates)."""
    return httpx.Request("POST", url, headers={"Authorization": f"Bearer {PLANTED}"})


def _assert_cause_chain_key_free(err: transport.TransportError) -> None:
    """Shared V8 assertions: the surfaced TransportError leaks the key nowhere.

    The load-bearing guarantee is that ``traceback.format_exception`` (which honors
    ``__suppress_context__``) renders neither the httpx exception nor its headers,
    and that ``__cause__`` is dropped. The transport additionally clears
    ``__context__`` at a boundary (see ``transport._raise_transport_error``), so a
    *direct attribute walk* also finds no header-bearing httpx exception -- we pin
    that too.
    """
    formatted = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    assert PLANTED not in formatted, "planted key leaked via the formatted traceback"
    assert PLANTED not in repr(err)
    assert PLANTED not in str(err)
    assert err.__cause__ is None
    assert err.__suppress_context__ is True
    # Explicit context-clear makes even a direct ``err.__context__`` walk key-free.
    assert err.__context__ is None


async def test_transport_error_drops_httpx_cause_chain(monkeypatch, mock_stream_client):
    """V8 (buffered): post_json's TransportError keeps no header-bearing httpx cause.

    The MockTransport handler raises ``httpx.ConnectError`` carrying a request whose
    headers hold the planted Authorization token. ``post_json`` must raise a
    ``TransportError`` whose formatted traceback, repr, and str are all key-free and
    whose cause chain is dropped (audit RANK 1/5).
    """
    # Value-based redact would also mask the key in any *string* the transport built;
    # the POINT here is the cause chain, asserted on the formatted traceback.
    monkeypatch.setenv("OPENAI_API_KEY", PLANTED)
    monkeypatch.setenv("XAI_API_KEY", PLANTED)

    def handler(request: httpx.Request) -> httpx.Response:
        # Raise a network error carrying a header-bearing request (httpx.HTTPError arm).
        raise httpx.ConnectError("boom", request=_planted_header_request())

    mock_stream_client(handler)

    with pytest.raises(transport.TransportError) as excinfo:
        await transport.post_json(
            "https://x/y",
            {"Authorization": f"Bearer {PLANTED}"},
            {"hi": "there"},
            timeout=5.0,
        )
    _assert_cause_chain_key_free(excinfo.value)


async def test_stream_sse_error_drops_httpx_cause_chain(monkeypatch, mock_stream_client):
    """V8 (streaming): stream_sse's TransportError keeps no header-bearing httpx cause.

    Same guarantee at the streaming boundary: the handler raises an httpx error
    carrying the planted auth header; iterating the async generator surfaces a
    ``TransportError`` whose formatted traceback / repr / str are key-free and whose
    cause chain is dropped (audit RANK 1/5).
    """
    monkeypatch.setenv("OPENAI_API_KEY", PLANTED)
    monkeypatch.setenv("XAI_API_KEY", PLANTED)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=_planted_header_request())

    mock_stream_client(handler)

    async def drain() -> None:
        async for _event in transport.stream_sse(
            "https://x/y",
            {"Authorization": f"Bearer {PLANTED}"},
            {"hi": "there"},
            timeout=5.0,
        ):
            pass  # pragma: no cover -- the handler raises before any event

    with pytest.raises(transport.TransportError) as excinfo:
        await drain()
    _assert_cause_chain_key_free(excinfo.value)


# --------------------------------------------------------------------------- #
# V9 -- Council installs the transport-logging guard by default (RANK 6)
# --------------------------------------------------------------------------- #


def _minimal_council_config() -> ConclaveConfig:
    """A deterministic, on-disk-independent config for the guard-install tests."""
    return ConclaveConfig(
        models={"openai": "openai/gpt-4.1"},
        councils={"default": ["openai"]},
        synthesizer="openai",
    )


def _snapshot_transport_logger_state():
    """Snapshot the httpx/httpcore filter lists + the one-shot guard flag."""
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    return (
        httpx_logger,
        httpcore_logger,
        httpx_logger.filters[:],
        httpcore_logger.filters[:],
        transport._GUARD_INSTALLED,
    )


def test_council_init_installs_transport_guard_by_default(monkeypatch):
    """V9 (RANK 6): constructing a Council installs the httpx/httpcore DEBUG guard.

    With no ``allow_transport_debug_logging`` argument the guard is ON: after
    construction ``_GUARD_INSTALLED`` is True, the httpcore logger carries a
    ``_NoDebugHeadersFilter``, and a header-bearing DEBUG record is dropped. Global
    logging state is snapshotted and restored (mirrors the V5 teardown discipline).
    """
    httpx_logger, httpcore_logger, saved_httpx, saved_httpcore, saved_flag = (
        _snapshot_transport_logger_state()
    )
    monkeypatch.setattr(transport, "_GUARD_INSTALLED", False)
    try:
        Council(models=["openai"], synthesizer="openai", config=_minimal_council_config())

        assert transport._GUARD_INSTALLED is True
        guard_filter = next(
            f for f in httpcore_logger.filters if f.__class__.__name__ == "_NoDebugHeadersFilter"
        )
        # A header-bearing DEBUG record on httpcore must be dropped by the guard.
        debug_rec = httpcore_logger.makeRecord(
            "httpcore",
            logging.DEBUG,
            __file__,
            0,
            "send_request_headers.complete return_value=[(b'Authorization', b'Bearer %s')]",
            (PLANTED.encode(),),
            None,
        )
        assert guard_filter.filter(debug_rec) is False
    finally:
        httpx_logger.filters = saved_httpx
        httpcore_logger.filters = saved_httpcore
        transport._GUARD_INSTALLED = saved_flag


def test_council_opt_out_skips_transport_guard(monkeypatch):
    """V9 (RANK 6 opt-out): allow_transport_debug_logging=True skips guard install.

    Starting from ``_GUARD_INSTALLED = False``, constructing a Council with the
    opt-out flag must NOT flip the flag and must NOT add a new
    ``_NoDebugHeadersFilter`` to the httpcore logger. Global state is restored after.
    """
    httpx_logger, httpcore_logger, saved_httpx, saved_httpcore, saved_flag = (
        _snapshot_transport_logger_state()
    )
    monkeypatch.setattr(transport, "_GUARD_INSTALLED", False)
    filters_before = httpcore_logger.filters[:]
    try:
        Council(
            models=["openai"],
            synthesizer="openai",
            config=_minimal_council_config(),
            allow_transport_debug_logging=True,
        )

        # This construction must not have installed the guard.
        assert transport._GUARD_INSTALLED is False
        assert httpcore_logger.filters == filters_before
    finally:
        httpx_logger.filters = saved_httpx
        httpcore_logger.filters = saved_httpcore
        transport._GUARD_INSTALLED = saved_flag


# --------------------------------------------------------------------------- #
# V10 -- pooled client closes without a ResourceWarning (RANK 8, cheap)
# --------------------------------------------------------------------------- #


async def test_pooled_client_closes_without_resource_warning():
    """V10 (RANK 8): creating then closing the pooled client emits no ResourceWarning.

    A cheap close-wiring regression guard. ``_get_client()`` lazily builds the
    pooled client; ``aclose()`` must release it cleanly. We snapshot/restore the
    global client so this does not clobber other tests, and turn ResourceWarning
    into an error so a leaked client surfaces immediately.
    """
    saved = transport._client
    transport._client = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            client = transport._get_client()
            assert client is not None
            await transport.aclose()
            assert transport._client is None
    finally:
        transport._client = saved
