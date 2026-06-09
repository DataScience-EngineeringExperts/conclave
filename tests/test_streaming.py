"""Tests for streaming member answers + synthesis (issue #7).

All tests run offline. Two layers are exercised:

* **Per-adapter SSE parsing** drives the *real* ``call_model_stream`` ->
  ``transport.stream_sse`` -> adapter ``parse_sse_event`` path through an
  :class:`httpx.MockTransport` that emits a multi-chunk SSE byte stream for each
  adapter family (openai-compat, anthropic, gemini). We assert chunks arrive
  incrementally AND that the assembled ``ModelAnswer.answer`` equals the
  concatenation -- and matches what the buffered path produces for the same
  content.
* **Council / CLI streaming** drives ``Council.ask_stream`` and the CLI
  ``--stream`` flag with ``call_model_stream`` patched, covering interleaving,
  the terminal ``done`` result, the never-opened-stream guarantee for the
  default path, the mid-stream error contract, and the cache interaction.
"""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from conclave import Council, cli, transport
from conclave.config import ConclaveConfig
from conclave.models import ModelAnswer, StreamEvent
from conclave.providers import call_model, call_model_stream

runner = CliRunner()


# --------------------------------------------------------------------------- #
# MockTransport-backed streaming client (mirrors tests/test_transport.py)
# --------------------------------------------------------------------------- #


@pytest.fixture
async def mock_stream_client():
    """Install a MockTransport-backed pooled client; restore the global after.

    Returns an installer ``use(handler)`` where ``handler(request) -> Response``.
    For streaming, the handler returns an ``httpx.Response`` whose content is an
    iterable of SSE byte chunks, so ``transport.stream_sse`` reads them via
    ``aiter_lines`` exactly as it would a real network stream.
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


def _sse(*frames: str) -> bytes:
    """Join raw SSE frame blocks into a single body (blank-line separated)."""
    return ("".join(f"{frame}\n\n" for frame in frames)).encode("utf-8")


async def _collect(name, model_id, **kwargs):
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


# --------------------------------------------------------------------------- #
# Per-adapter SSE parsing (real adapter + real transport, mocked bytes)
# --------------------------------------------------------------------------- #


async def test_openai_compat_stream_assembles_and_increments(monkeypatch, mock_stream_client):
    """OpenAI-style deltas stream incrementally; final answer = concatenation."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        body = _sse(
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":", "}}]}',
            'data: {"choices":[{"delta":{"content":"world"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,'
            '"total_tokens":5}}',
            "data: [DONE]",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    mock_stream_client(handler)
    chunks, final = await _collect("openai", "openai/gpt-4.1")

    # Streamed incrementally as separate text chunks (role-only delta carries no text).
    assert chunks == ["Hello", ", ", "world"]
    assert final is not None and final.ok
    assert final.answer == "Hello, world"
    assert final.answer == "".join(chunks)
    assert final.usage is not None and final.usage.total_tokens == 5
    # The request actually enabled streaming + usage accounting.
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}


async def test_anthropic_stream_assembles_and_increments(monkeypatch, mock_stream_client):
    """Anthropic named-event SSE: text_delta -> text; usage from start+delta frames."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        body = _sse(
            "event: message_start\n"
            'data: {"type":"message_start","message":{"usage":{"input_tokens":10,'
            '"output_tokens":1}}}',
            "event: content_block_start\n"
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"text","text":""}}',
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Hel"}}',
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"lo"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            "event: message_delta\n"
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":7}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    mock_stream_client(handler)
    chunks, final = await _collect("claude", "anthropic/claude-sonnet-4-6")

    assert chunks == ["Hel", "lo"]
    assert final is not None and final.ok
    assert final.answer == "Hello"
    assert final.answer == "".join(chunks)
    # input_tokens from message_start, output_tokens from message_delta, total = sum.
    assert final.usage is not None
    assert final.usage.prompt_tokens == 10
    assert final.usage.completion_tokens == 7
    assert final.usage.total_tokens == 17
    assert captured["body"]["stream"] is True


async def test_gemini_stream_assembles_and_increments(monkeypatch, mock_stream_client):
    """Gemini alt=sse: each chunk carries parts[].text; cumulative usageMetadata."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        body = _sse(
            'data: {"candidates":[{"content":{"parts":[{"text":"Foo"}],"role":"model"}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"bar"}],"role":"model"}}],'
            '"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,'
            '"totalTokenCount":6}}',
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    mock_stream_client(handler)
    chunks, final = await _collect("gemini", "gemini/gemini-2.5-pro")

    assert chunks == ["Foo", "bar"]
    assert final is not None and final.ok
    assert final.answer == "Foobar"
    assert final.answer == "".join(chunks)
    assert final.usage is not None and final.usage.total_tokens == 6
    # The streaming URL targets streamGenerateContent with ?alt=sse.
    assert ":streamGenerateContent" in captured["url"]
    assert "alt=sse" in captured["url"]


async def test_stream_final_answer_matches_buffered(monkeypatch, mock_stream_client):
    """The assembled streamed answer equals the buffered parse for the same content."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    full = "The quick brown fox"

    def stream_handler(request: httpx.Request) -> httpx.Response:
        frames = [f'data: {{"choices":[{{"delta":{{"content":"{w} "}}}}]}}' for w in full.split()]
        frames.append("data: [DONE]")
        return httpx.Response(200, content=_sse(*frames))

    mock_stream_client(stream_handler)
    _chunks, streamed = await _collect("openai", "openai/gpt-4.1")

    def buffered_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": streamed.answer}}]})

    mock_stream_client(buffered_handler)
    buffered = await call_model("openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}])

    assert streamed.answer == buffered.answer


# --------------------------------------------------------------------------- #
# Mid-stream error: partial text preserved, error set, never raises
# --------------------------------------------------------------------------- #


async def test_midstream_malformed_frame_preserves_partial(monkeypatch, mock_stream_client):
    """A malformed SSE frame mid-stream -> error set, partial text kept, no raise."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse(
            'data: {"choices":[{"delta":{"content":"partial "}}]}',
            "data: {not valid json",  # malformed -> ProviderError in parse_sse_event
            "data: [DONE]",
        )
        return httpx.Response(200, content=body)

    mock_stream_client(handler)
    chunks, final = await _collect("openai", "openai/gpt-4.1")

    # The good chunk was streamed before the failure.
    assert chunks == ["partial "]
    assert final is not None
    assert not final.ok
    assert final.error is not None
    assert "partial " in (final.answer or "")  # partial text preserved on the answer


async def test_midstream_connection_drop_preserves_partial(monkeypatch, mock_stream_client):
    """A transport-level drop mid-stream -> error set, partial preserved, no raise."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    class DroppingStream(httpx.AsyncByteStream):
        """Emits one good SSE frame, then raises a read error mid-stream."""

        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"half "}}]}\n\n'
            raise httpx.ReadError("connection dropped")

        async def aclose(self):  # pragma: no cover - nothing to release
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=DroppingStream())

    mock_stream_client(handler)
    chunks, final = await _collect("openai", "openai/gpt-4.1")

    assert chunks == ["half "]
    assert final is not None and not final.ok
    assert "network error" in final.error
    assert (final.answer or "") == "half "


async def test_stream_non2xx_status_is_error(monkeypatch, mock_stream_client):
    """A non-2xx streaming status becomes a non-raising ModelAnswer.error."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    mock_stream_client(handler)
    chunks, final = await _collect("openai", "openai/gpt-4.1")

    assert chunks == []
    assert final is not None and not final.ok
    assert "401" in final.error


async def test_stream_missing_key_is_error(monkeypatch):
    """No key set -> a clean ModelAnswer.error naming the env var, no stream opened."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    chunks, final = await _collect("openai", "openai/gpt-4.1")
    assert chunks == []
    assert final is not None and not final.ok
    assert "OPENAI_API_KEY" in final.error


async def test_stream_key_redacted_in_error(monkeypatch, mock_stream_client):
    """A key value echoed in a streaming error body is scrubbed via redact()."""
    fake_key = "sk-streamleak-0123456789abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {fake_key}"}})

    mock_stream_client(handler)
    _chunks, final = await _collect("openai", "openai/gpt-4.1")
    assert final is not None and not final.ok
    assert fake_key not in final.error
    assert "[REDACTED]" in final.error


# --------------------------------------------------------------------------- #
# Non-streaming default path must never open a stream
# --------------------------------------------------------------------------- #


def _config() -> ConclaveConfig:
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": "anthropic/claude-sonnet-4-6",
        },
        councils={"default": ["grok", "gemini", "claude"]},
        synthesizer="claude",
    )


async def test_default_ask_never_opens_a_stream(monkeypatch, patch_call_model):
    """The buffered ask() path must not call transport.stream_sse at all."""
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    opened = {"n": 0}

    async def spy_stream_sse(*args, **kwargs):  # pragma: no cover - must not run
        opened["n"] += 1
        if False:
            yield  # make it an async generator

    monkeypatch.setattr(transport, "stream_sse", spy_stream_sse)

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        return make_response(f"ans-{model}")

    patch_call_model(handler)
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    result = await council.ask("hi", synthesize=False)
    assert all(a.ok for a in result.answers)
    assert opened["n"] == 0, "buffered ask() must never open a stream"


# --------------------------------------------------------------------------- #
# Council.ask_stream interleaving + terminal result
# --------------------------------------------------------------------------- #


def _patch_stream(monkeypatch, deltas_by_model):
    """Patch streaming.call_model_stream to emit canned deltas + a final answer."""
    import conclave.streaming as streaming_mod

    async def fake_stream(name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None):
        text_parts = deltas_by_model.get(model_id, ["x"])
        for part in text_parts:
            yield part
        yield ModelAnswer(name=name, model_id=model_id, answer="".join(text_parts))

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)


async def test_ask_stream_yields_member_and_done_events(monkeypatch):
    """ask_stream yields member deltas/dones for each member then a done result."""
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    _patch_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["A", "B"],
            "gemini/gemini-2.5-pro": ["C"],
            "anthropic/claude-sonnet-4-6": ["SYN"],  # synthesizer
        },
    )

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("hi", synthesize=True)]

    types = [e.type for e in events]
    assert types[-1] == "done"
    assert "member_delta" in types
    assert types.count("member_done") == 2
    assert "synthesis_delta" in types
    assert "synthesis_done" in types

    done = events[-1]
    assert done.result is not None
    # Final result shape matches non-streaming: 2 answers ordered by members list.
    assert [a.name for a in done.result.answers] == ["grok", "gemini"]
    assert done.result.answers[0].answer == "AB"
    assert done.result.synthesis == "SYN"


async def test_ask_stream_raw_skips_synthesis(monkeypatch):
    """raw mode (synthesize=False) yields no synthesis events."""
    for var in ("XAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    _patch_stream(monkeypatch, {"xai/grok-4.3": ["a"], "gemini/gemini-2.5-pro": ["b"]})
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("hi", synthesize=False)]

    assert not any(e.type.startswith("synthesis") for e in events)
    assert events[-1].type == "done"
    assert events[-1].result.synthesis is None


async def test_ask_stream_no_members_emits_done(monkeypatch):
    """Zero available members -> a single done event, empty answers, no raise."""
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("hi")]
    assert len(events) == 1
    assert events[0].type == "done"
    assert events[0].result.answers == []
    assert events[0].result.skipped == ["grok", "gemini"]


# --------------------------------------------------------------------------- #
# CLI --stream smoke + exit-code contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def patch_cli_config(monkeypatch):
    monkeypatch.setattr(cli, "load_config", _config)


def test_cli_stream_smoke_exits_zero(monkeypatch, patch_cli_config):
    """--stream renders live output and exits 0 on a successful mocked stream."""
    import conclave.streaming as streaming_mod

    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    async def fake_stream(name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None):
        yield f"tok-{name} "
        yield ModelAnswer(name=name, model_id=model_id, answer=f"tok-{name} ")

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)

    result = runner.invoke(
        cli.app, ["ask", "hello", "--council", "grok,gemini", "--mode", "raw", "--stream"]
    )
    assert result.exit_code == 0
    assert "tok-grok" in result.output
    assert "tok-gemini" in result.output


def test_cli_stream_zero_usable_exits_one(monkeypatch, patch_cli_config):
    """--stream honors the exit-code contract: zero usable answers -> exit 1."""
    import conclave.streaming as streaming_mod

    for var in ("XAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    async def fake_stream(name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None):
        yield ModelAnswer(name=name, model_id=model_id, error="provider down")

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)

    result = runner.invoke(
        cli.app, ["ask", "hello", "--council", "grok,gemini", "--mode", "raw", "--stream"]
    )
    assert result.exit_code == 1
    assert "No usable council answers" in result.output


def test_cli_stream_rejected_for_debate(patch_cli_config):
    """--stream is rejected (exit 2) for non synthesize/raw modes."""
    result = runner.invoke(
        cli.app, ["ask", "hello", "--council", "grok", "--mode", "debate", "--stream"]
    )
    assert result.exit_code == 2
    assert "only supported for synthesize/raw" in result.output


# --------------------------------------------------------------------------- #
# --stream + cache: first run streams + populates; second is a one-shot hit
# --------------------------------------------------------------------------- #


def test_cli_stream_cache_second_run_is_one_shot_hit(monkeypatch, patch_cli_config, tmp_path):
    """--stream --cache: 2nd identical run is a hit rendered in one shot (no calls)."""
    import conclave.streaming as streaming_mod

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for var in ("XAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    calls = {"n": 0}

    async def fake_stream(name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None):
        calls["n"] += 1
        yield "live "
        yield ModelAnswer(name=name, model_id=model_id, answer="live answer")

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)

    args = ["ask", "2+2?", "--council", "grok", "--mode", "raw", "--stream", "--cache"]
    first = runner.invoke(cli.app, args)
    assert first.exit_code == 0
    n_after_first = calls["n"]
    assert n_after_first > 0

    second = runner.invoke(cli.app, args)
    assert second.exit_code == 0
    # No new provider stream calls -> served from cache, rendered one-shot.
    assert calls["n"] == n_after_first
    assert "live answer" in second.output


async def test_ask_stream_cache_hit_replays_one_shot(monkeypatch, tmp_path):
    """A cache hit replays as single-shot deltas with cached result, no live stream."""
    import conclave.streaming as streaming_mod

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    live_calls = {"n": 0}

    async def fake_stream(name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None):
        live_calls["n"] += 1
        yield "x"
        yield ModelAnswer(name=name, model_id=model_id, answer="x")

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)

    council = Council(models=["grok", "gemini"], synthesizer="claude", config=_config(), cache=True)
    # First run populates the cache.
    first = [e async for e in council.ask_stream("hi", synthesize=False)]
    assert first[-1].result.cached is False
    n_after_first = live_calls["n"]

    # Second identical run is a cache hit -> no new live stream calls.
    second = [e async for e in council.ask_stream("hi", synthesize=False)]
    assert live_calls["n"] == n_after_first
    done = second[-1]
    assert done.type == "done"
    assert done.result.cached is True
    # One member_delta per member (one-shot), each followed by a member_done.
    member_deltas = [e for e in second if e.type == "member_delta"]
    assert len(member_deltas) == 2


def test_stream_event_done_carries_full_result_shape():
    """StreamEvent('done') carries a CouncilResult that serializes secret-free."""
    from conclave.models import CouncilResult

    ev = StreamEvent(type="done", result=CouncilResult(prompt="p"))
    dumped = ev.model_dump(mode="json")
    assert dumped["type"] == "done"
    assert dumped["result"]["prompt"] == "p"
