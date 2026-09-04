"""Tests for the transport record/replay seam (DSE-1517 Task 2).

Covers ``conclave.transport``'s ``ContextVar``-based seam -- ``RecordingContext``
/ ``ReplayContext``, ``recording()`` / ``replaying()``, and the process-global
strict backstop -- plus ``conclave.providers._resolve_key``'s offline-context
branch and the factored ``unkeyed_error_message`` helper. All tests run offline;
none touches the network.

Test map (docstring on each test names which binding-rule item it proves):

1. Seam unset -> ``post_json`` reaches the (stubbed) client exactly as before.
2. ``ReplayContext`` -> the override is called; ``_get_client`` is never
   invoked.
3. ``RecordingContext`` -> the override wraps the live path, and
   ``_resolve_key`` reads the REAL env var under it (the F1 guard).
4. ``_resolve_key`` returns the sentinel only under ``ReplayContext`` and reads
   no env var there.
5. ``stream_sse`` raises under ``ReplayContext``/strict, but not under
   ``RecordingContext``.
6. Task isolation: a live task started alongside a replaying task is
   unaffected.
7. The strict backstop blocks a thread-hop call; ``strict=False`` does not; the
   refcount always returns to 0.
8. ``unkeyed_error_message`` is byte-identical to what ``call_model`` surfaces.
"""

from __future__ import annotations

import asyncio

import pytest

import conclave.providers as providers_mod
from conclave import transport
from conclave.adapters import resolve_adapter
from conclave.config import ConclaveConfig
from conclave.providers import call_model, unkeyed_error_message
from conclave.tape import RecordingTransport
from conclave.transport import RecordingContext, ReplayContext, TransportError

RUN_HASH = "sha256:" + "a" * 64


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response`` -- only what ``post_json`` reads."""

    def __init__(self, status_code: int, json_body: object) -> None:
        self.status_code = status_code
        self._json_body = json_body

    def json(self) -> object:
        return self._json_body


class _FakeClient:
    """Stand-in for ``httpx.AsyncClient`` -- records every ``.post`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict, float]] = []

    async def post(self, url: str, headers: dict, json: dict, timeout: float) -> _FakeResponse:
        self.calls.append((url, headers, json, timeout))
        return _FakeResponse(200, {"ok": True})


# --------------------------------------------------------------------------- #
# 1. Seam unset -- unchanged default path
# --------------------------------------------------------------------------- #


async def test_post_json_reaches_stubbed_client_when_no_context_is_set(monkeypatch):
    """With no context active, ``post_json`` is today's code path, unchanged."""
    fake_client = _FakeClient()
    monkeypatch.setattr(transport, "_get_client", lambda: fake_client)

    status, body = await transport.post_json("https://x/y", {"H": "1"}, {"a": 1}, 5.0)

    assert (status, body) == (200, {"ok": True})
    assert fake_client.calls == [("https://x/y", {"H": "1"}, {"a": 1}, 5.0)]


# --------------------------------------------------------------------------- #
# 2. ReplayContext intercepts before _get_client
# --------------------------------------------------------------------------- #


async def test_replay_context_intercepts_before_get_client(monkeypatch):
    """A ``ReplayContext`` override is called; ``_get_client`` is never invoked."""

    def boom() -> None:
        raise AssertionError("_get_client must not be called under ReplayContext")

    monkeypatch.setattr(transport, "_get_client", boom)

    async def fake_override(url, headers, json_body, timeout):
        return 299, {"replayed": True}

    ctx = ReplayContext(post_json=fake_override)
    with transport.replaying(ctx, strict=False):
        status, body = await transport.post_json("https://x/y", {}, {}, 5.0)

    assert (status, body) == (299, {"replayed": True})


# --------------------------------------------------------------------------- #
# 3. RecordingContext wraps the live path; _resolve_key reads the real env var
#    under it (F1 guard)
# --------------------------------------------------------------------------- #


async def test_recording_context_wraps_live_post_json_with_same_args():
    """``RecordingContext`` wraps ``post_json``; the delegate sees identical args."""
    calls: list[tuple] = []

    async def fake_live(url, headers, json_body, timeout):
        calls.append((url, headers, json_body, timeout))
        return 200, {"ok": True}

    recorder = RecordingTransport(fake_live, run_identity_hash=RUN_HASH)
    ctx = RecordingContext(post_json=recorder)

    with transport.recording(ctx):
        status, body = await transport.post_json("https://x/y", {"H": "1"}, {"a": 1}, 5.0)

    assert (status, body) == (200, {"ok": True})
    assert calls == [("https://x/y", {"H": "1"}, {"a": 1}, 5.0)]
    assert len(recorder.tape().records) == 1


async def test_resolve_key_reads_real_env_var_under_recording_context(monkeypatch):
    """F1 guard: a RecordingContext never substitutes the replay sentinel.

    The adapter must receive the REAL env var value (``dummy-live``), not the
    ``"replay"`` sentinel, even though a transport context is active.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-live")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    seen: dict = {}

    async def fake_live(url, headers, json_body, timeout):
        seen["headers"] = headers
        return 200, {
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    recorder = RecordingTransport(fake_live, run_identity_hash=RUN_HASH)
    ctx = RecordingContext(post_json=recorder)

    with transport.recording(ctx):
        answer = await call_model(
            "claude", "anthropic/claude-sonnet-4-6", [{"role": "user", "content": "hi"}]
        )

    assert answer.ok
    assert seen["headers"]["x-api-key"] == "dummy-live"


# --------------------------------------------------------------------------- #
# 4. _resolve_key: sentinel only under ReplayContext, no env read there
# --------------------------------------------------------------------------- #


async def test_resolve_key_returns_sentinel_only_under_replay_context(monkeypatch):
    """No context and a RecordingContext both fall through to the real env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-live")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")
    adapter = resolve_adapter("anthropic/claude-sonnet-4-6", ConclaveConfig())

    assert providers_mod._resolve_key(adapter) == "dummy-live"

    async def fake_live(url, headers, json_body, timeout):
        return 200, {}

    recorder = RecordingTransport(fake_live, run_identity_hash=RUN_HASH)
    with transport.recording(RecordingContext(post_json=recorder)):
        assert providers_mod._resolve_key(adapter) == "dummy-live"

    async def fake_override(url, headers, json_body, timeout):
        return 200, {}

    with transport.replaying(ReplayContext(post_json=fake_override), strict=False):
        assert providers_mod._resolve_key(adapter) == "replay"


def test_resolve_key_reads_no_env_var_under_replay_context(monkeypatch):
    """Under an offline ReplayContext, os.environ is never consulted at all."""

    def boom(*args, **kwargs):
        raise AssertionError("must not read os.environ under an offline replay context")

    monkeypatch.setattr(providers_mod.os.environ, "get", boom)
    adapter = resolve_adapter("anthropic/claude-sonnet-4-6", ConclaveConfig())

    async def fake_override(url, headers, json_body, timeout):
        return 200, {}

    with transport.replaying(ReplayContext(post_json=fake_override), strict=False):
        assert providers_mod._resolve_key(adapter) == "replay"


# --------------------------------------------------------------------------- #
# 5. stream_sse: raises under ReplayContext/strict, not under RecordingContext
# --------------------------------------------------------------------------- #


async def test_stream_sse_raises_under_replay_context_and_strict_backstop():
    """``stream_sse`` refuses (category ``unexpected``) under a strict replay."""

    async def fake_override(url, headers, json_body, timeout):
        return 200, {}

    ctx = ReplayContext(post_json=fake_override)
    with transport.replaying(ctx, strict=True):
        with pytest.raises(TransportError) as exc_info:
            async for _ in transport.stream_sse("https://x/y", {}, {}, 5.0):
                pass

    assert exc_info.value.category == "unexpected"
    assert transport._OFFLINE_STRICT == 0


async def test_stream_sse_does_not_raise_offline_error_under_recording_context(monkeypatch):
    """``--record`` is buffered-only: RecordingContext never blocks streaming.

    Proven by patching ``_get_client`` to raise a distinct sentinel: reaching it
    (instead of the "strict offline replay" TransportError) proves the seam
    check did not intercept the call.
    """

    class _ReachedGetClient(Exception):
        pass

    def boom() -> None:
        raise _ReachedGetClient

    monkeypatch.setattr(transport, "_get_client", boom)

    async def fake_live(url, headers, json_body, timeout):
        return 200, {}

    ctx = RecordingContext(post_json=fake_live)
    with transport.recording(ctx):
        with pytest.raises(_ReachedGetClient):
            async for _ in transport.stream_sse("https://x/y", {}, {}, 5.0):
                pass


# --------------------------------------------------------------------------- #
# 6. Task isolation
# --------------------------------------------------------------------------- #


async def test_context_isolation_between_concurrent_tasks(monkeypatch):
    """A sibling live task is unaffected by a replaying task's context."""
    fake_client = _FakeClient()
    monkeypatch.setattr(transport, "_get_client", lambda: fake_client)

    async def replay_task():
        async def fake_override(url, headers, json_body, timeout):
            return 299, {"replayed": True}

        ctx = ReplayContext(post_json=fake_override)
        with transport.replaying(ctx, strict=False):
            return await transport.post_json("https://replay/x", {}, {}, 5.0)

    async def live_task():
        return await transport.post_json("https://live/y", {}, {}, 5.0)

    # The two coroutines are only wrapped into Tasks by gather() itself; only
    # replay_task's own body enters replaying() -- live_task never does.
    replay_result, live_result = await asyncio.gather(replay_task(), live_task())

    assert replay_result == (299, {"replayed": True})
    assert live_result == (200, {"ok": True})
    assert fake_client.calls == [("https://live/y", {}, {}, 5.0)]


# --------------------------------------------------------------------------- #
# 7. Strict backstop blocks a thread hop; strict=False does not
# --------------------------------------------------------------------------- #


async def test_strict_backstop_blocks_a_thread_hop_that_calls_post_json(monkeypatch):
    """A ``run_in_executor`` hop has no inherited ContextVar; the refcount blocks it."""
    fake_client = _FakeClient()
    monkeypatch.setattr(transport, "_get_client", lambda: fake_client)

    def call_post_json_in_a_fresh_event_loop():
        # A plain OS thread has no asyncio Task and therefore no contextvars
        # Context copied from the replaying() task: _TRANSPORT.get() here reads
        # the ContextVar's default (None). Only the process-global refcount can
        # block this call.
        async def _inner():
            return await transport.post_json("https://escaped/x", {}, {}, 5.0)

        return asyncio.run(_inner())

    async def fake_override(url, headers, json_body, timeout):
        return 299, {"replayed": True}

    ctx = ReplayContext(post_json=fake_override)
    loop = asyncio.get_running_loop()

    with transport.replaying(ctx, strict=True):
        with pytest.raises(TransportError) as exc_info:
            await loop.run_in_executor(None, call_post_json_in_a_fresh_event_loop)

    assert exc_info.value.category == "unexpected"
    assert transport._OFFLINE_STRICT == 0

    # With strict=False the same hop is NOT blocked -- it reaches the stub client.
    with transport.replaying(ctx, strict=False):
        status, body = await loop.run_in_executor(None, call_post_json_in_a_fresh_event_loop)

    assert (status, body) == (200, {"ok": True})
    assert transport._OFFLINE_STRICT == 0


# --------------------------------------------------------------------------- #
# 8. unkeyed_error_message is byte-identical to call_model's surfaced error
# --------------------------------------------------------------------------- #


async def test_unkeyed_error_message_matches_call_model_output(monkeypatch):
    """The factored helper produces the same string call_model has always used.

    Reuses ``tests/test_providers.py::test_call_model_missing_key_is_error``'s
    ``"OPENAI_API_KEY" in answer.error`` assertion, plus an exact equality check
    against the helper's own output.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    adapter = resolve_adapter("openai/gpt-4.1", ConclaveConfig())
    assert unkeyed_error_message(adapter) == "no API key in environment (set OPENAI_API_KEY)"

    answer = await call_model("openai", "openai/gpt-4.1", [{"role": "user", "content": "hi"}])
    assert not answer.ok
    assert "OPENAI_API_KEY" in answer.error  # existing test_providers.py assertion
    assert answer.error == unkeyed_error_message(adapter)
