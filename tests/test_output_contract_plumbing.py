"""CAC-06-PLUMB: OutputContract threading through the provider call path.

Wave 2 built native structured-output shaping on ``adapter.build_request`` /
``stream_request`` (both accept an ``output_contract`` kwarg), but
:func:`conclave.providers.call_model` never threaded an ``OutputContract`` to the
adapter, so that machinery was unreachable from the call path. CAC-06-PLUMB threads
``output_contract`` through ``call_model`` / ``call_model_stream`` and migrates
:func:`conclave.verdict_synthesis.extract_verdict` to ALSO request native structured
output while KEEPING the prompt-level parse/validate/repair as the fallback.

These tests prove, fully offline (no network):

* The contract reaches ``build_request`` and lands as the OpenAI native
  ``response_format`` ``json_schema`` directive in the request BODY for a capable
  model (``openai/gpt-4.1``) -- and that ``None`` leaves the body free-prose.
* ``call_model_stream`` threads the contract too (real adapter + MockTransport).
* ``extract_verdict`` now calls ``call_model`` WITH an ``OutputContract`` whose
  ``.schema`` is the extraction schema -- and STILL degrades gracefully to
  ``verdict=None`` (with the recorded reason + logged warning) on bad JSON.

``asyncio_mode = "auto"`` (pyproject ``[tool.pytest.ini_options]``) -> async tests
need no decorator. The ``conclave`` logger sets ``propagate=False`` (a key-leak
defense), so log assertions use the shared ``conclave_caplog`` fixture, never bare
``caplog``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from conclave import transport
from conclave.adapters.base import OutputContract
from conclave.models import ModelAnswer
from conclave.providers import call_model, call_model_stream
from conclave.verdict import verdict_extraction_json_schema

# A minimal, adapter-acceptable JSON-Schema dict for the call-path tests. The
# OpenAI adapter passes ``output_contract.schema`` straight through into
# ``response_format.json_schema.schema``; any object schema is fine here.
_MINIMAL_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# call_model: the contract reaches build_request and lands in the body
# --------------------------------------------------------------------------- #


async def test_call_model_threads_output_contract_into_body(monkeypatch):
    """An OutputContract on call_model lands as OpenAI's native response_format.

    ``openai/gpt-4.1`` has ``supports_structured_output=True`` in the static
    catalog, so the adapter emits the strict ``json_schema`` directive. We assert
    against the EXACT shape ``OpenAICompatAdapter._apply_output_contract`` builds:
    ``response_format = {"type": "json_schema", "json_schema": {"name", "strict",
    "schema"}}``.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}

    async def fake_post_json(url, headers, json_body, timeout):
        captured["body"] = json_body
        return 200, {
            "choices": [{"message": {"content": '{"answer": "ok"}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    monkeypatch.setattr("conclave.transport.post_json", fake_post_json)

    answer = await call_model(
        "openai",
        "openai/gpt-4.1",
        [{"role": "user", "content": "hi"}],
        output_contract=OutputContract(schema=_MINIMAL_SCHEMA, schema_name="X", strict=True),
    )

    assert answer.ok
    response_format = captured["body"]["response_format"]
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["name"] == "X"
    assert json_schema["strict"] is True
    assert json_schema["schema"] == _MINIMAL_SCHEMA


async def test_call_model_without_contract_leaves_body_free_prose(monkeypatch):
    """Omitting the contract (the default) leaves no native directive in the body."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}

    async def fake_post_json(url, headers, json_body, timeout):
        captured["body"] = json_body
        return 200, {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    monkeypatch.setattr("conclave.transport.post_json", fake_post_json)

    answer = await call_model(
        "openai",
        "openai/gpt-4.1",
        [{"role": "user", "content": "hi"}],
    )

    assert answer.ok
    # Free-prose path: the native structured-output key is absent entirely.
    assert "response_format" not in captured["body"]


# --------------------------------------------------------------------------- #
# call_model_stream: the contract is threaded through to stream_request too
# --------------------------------------------------------------------------- #


@pytest.fixture
async def mock_stream_client():
    """Install a MockTransport-backed pooled client; restore the global after.

    Mirrors ``tests/test_streaming.py``. ``use(handler)`` swaps the transport
    module's pooled client for one whose ``handler(request) -> Response`` lets us
    both capture the streamed request BODY and return an SSE byte stream that
    ``transport.stream_sse`` reads exactly as it would a real network stream.
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


async def _drain_stream(name, model_id, **kwargs) -> ModelAnswer | None:
    """Run call_model_stream to completion, returning the final ModelAnswer."""
    final: ModelAnswer | None = None
    async for item in call_model_stream(
        name, model_id, [{"role": "user", "content": "hi"}], **kwargs
    ):
        if isinstance(item, ModelAnswer):
            final = item
    return final


def _openai_stream_handler(captured):
    """An SSE handler that records the streamed request body, then streams text."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        body = _sse(
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,'
            '"total_tokens":5}}',
            "data: [DONE]",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return handler


async def test_call_model_stream_threads_output_contract_into_body(monkeypatch, mock_stream_client):
    """A contract on call_model_stream lands as response_format in the streamed body.

    Drives the REAL ``call_model_stream -> transport.stream_sse -> adapter`` path
    via MockTransport, so the assertion is on the body the adapter actually built
    with stream flags layered on the contract-shaped request.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}
    mock_stream_client(_openai_stream_handler(captured))

    final = await _drain_stream(
        "openai",
        "openai/gpt-4.1",
        output_contract=OutputContract(schema=_MINIMAL_SCHEMA, schema_name="X", strict=True),
    )

    assert final is not None and final.ok
    # Stream flags AND the native directive both present (response_format is
    # documented as stream-compatible).
    assert captured["body"]["stream"] is True
    response_format = captured["body"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == _MINIMAL_SCHEMA


async def test_call_model_stream_without_contract_leaves_body_free_prose(
    monkeypatch, mock_stream_client
):
    """Omitting the contract on the stream path leaves no native directive."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    captured = {}
    mock_stream_client(_openai_stream_handler(captured))

    final = await _drain_stream("openai", "openai/gpt-4.1")

    assert final is not None and final.ok
    assert captured["body"]["stream"] is True
    assert "response_format" not in captured["body"]


async def test_call_model_stream_success_assigns_stable_answer_identity(
    monkeypatch, mock_stream_client
) -> None:
    """The streaming provider path shares the successful-answer ID contract."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")
    captured = {}
    mock_stream_client(_openai_stream_handler(captured))

    first = await _drain_stream("openai", "openai/gpt-4.1")
    second = await _drain_stream("openai", "openai/gpt-4.1")

    assert first is not None and second is not None
    assert first.ok and second.ok
    assert first.answer_id == second.answer_id
    assert (first.answer_id or "").startswith("ca_")


# --------------------------------------------------------------------------- #
# extract_verdict now requests native structured output (+ keeps the fallback)
# --------------------------------------------------------------------------- #


def _answer(name: str, text: str) -> ModelAnswer:
    """A responding member answer with a stable within-run answer id."""
    return ModelAnswer(name=name, model_id=f"{name}/m", answer=text, answer_id=f"{name}-1")


async def test_extract_verdict_passes_native_output_contract(monkeypatch):
    """extract_verdict threads an OutputContract carrying the extraction schema.

    Patches ``conclave.verdict_synthesis.call_model`` with a fake that records the
    ``output_contract`` kwarg it receives. Two responding answers clear the N<2
    gate so a real ``call_model`` happens. We only assert on the contract here (a
    deliberately invalid extraction returns verdict=None, which is fine -- the
    contract is recorded before validation).
    """
    import conclave.verdict_synthesis as vs

    captured = {}

    async def fake_call_model(
        name,
        model_id,
        messages,
        *,
        temperature=0.7,
        timeout=120.0,
        config=None,
        output_contract=None,
    ):
        captured["output_contract"] = output_contract
        return ModelAnswer(name=name, model_id=model_id, answer="not json")

    monkeypatch.setattr(vs, "call_model", fake_call_model)

    result = await vs.extract_verdict(
        "decide?",
        [_answer("a", "yes"), _answer("b", "no")],
        synthesizer_name="claude",
        synthesizer_model_id="anthropic/claude-sonnet-4-6",
    )

    contract = captured["output_contract"]
    assert isinstance(contract, OutputContract)
    assert contract.schema_name == "VerdictExtraction"
    assert contract.strict is True
    assert contract.schema == verdict_extraction_json_schema()
    # Bad JSON still degrades gracefully (the contract is additive, not behavior-changing).
    assert result.verdict is None


def test_verdict_prompt_describes_ids_as_answer_provenance_not_external_evidence() -> None:
    """Compatibility field IDs trace council answers; they do not prove facts."""
    import conclave.verdict_synthesis as vs

    answers = [_answer("a", "yes"), _answer("b", "no")]
    messages = vs._build_messages("decide?", answers)
    rendered = "\n".join(message["content"] for message in messages)

    assert "within-run answer provenance" in rendered
    assert "answer id:" in rendered
    assert "evidence id:" not in rendered


async def test_extract_verdict_still_degrades_gracefully_on_bad_json(monkeypatch, conclave_caplog):
    """Native contract is additive: bad JSON still yields verdict=None gracefully.

    Patches ``call_model`` to always return non-JSON text, so both the initial
    extraction and the repair retry fail. The engine must record the
    extraction-failed reason and log the post-repair warning -- never raise.
    """
    import conclave.verdict_synthesis as vs

    async def fake_call_model(
        name,
        model_id,
        messages,
        *,
        temperature=0.7,
        timeout=120.0,
        config=None,
        output_contract=None,
    ):
        return ModelAnswer(name=name, model_id=model_id, answer="not json")

    monkeypatch.setattr(vs, "call_model", fake_call_model)

    result = await vs.extract_verdict(
        "decide?",
        [_answer("a", "yes"), _answer("b", "no")],
        synthesizer_name="claude",
        synthesizer_model_id="anthropic/claude-sonnet-4-6",
    )

    assert result.verdict is None
    assert result.verdict_absent_reason == vs._REASON_EXTRACTION_FAILED
    # Provenance is still recorded on the absent path.
    assert result.extraction.model_id == "anthropic/claude-sonnet-4-6"
    # The post-repair warning is logged (conclave logger has propagate=False).
    assert any(
        "verdict extraction failed schema validation after repair" in rec.message
        for rec in conclave_caplog.records
    )
