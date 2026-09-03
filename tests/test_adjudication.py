"""Council.adjudicate walks the synthesizer chain with the infra-only failover rule."""

from __future__ import annotations

import pytest

from conclave import council as council_mod
from conclave.config import ConclaveConfig
from conclave.council import Council
from conclave.manifest import AdjudicationAttempt, ModelHarnessManifest
from conclave.models import CouncilResult, ModelAnswer

CFG = ConclaveConfig(models={"claude": "anthropic/c", "grok": "xai/g", "gemini": "gemini/m"})


def _fail(name, model_id, category, status=None):
    return ModelAnswer(
        name=name,
        model_id=model_id,
        error=f"{name} failed",
        failure_category=category,
        http_status=status,
    )


def _ok(name, model_id):
    return ModelAnswer(
        name=name, model_id=model_id, answer=f"{name} says yes", answer_id=f"{name}-1"
    )


def _install(monkeypatch, script: dict[str, ModelAnswer]):
    calls = []

    async def fake(name, model_id, messages, **kw):
        calls.append(name)
        return script[name]

    monkeypatch.setattr(council_mod, "call_model", fake)
    return calls


@pytest.fixture
def keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "dummy")


async def test_chain_of_one_success(monkeypatch, keys):
    calls = _install(monkeypatch, {"claude": _ok("claude", "anthropic/c")})
    c = Council(models=["grok"], synthesizer="claude", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer.ok and out.name == "claude" and calls == ["claude"]
    assert [a.outcome for a in out.attempts] == ["success"]


async def test_auth_failure_advances(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {"claude": _fail("claude", "anthropic/c", "auth", 401), "grok": _ok("grok", "xai/g")},
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer.ok and out.name == "grok" and out.model_id == "xai/g"
    assert calls == ["claude", "grok"]
    assert [(a.candidate, a.outcome, a.failure_category, a.http_status) for a in out.attempts] == [
        ("claude", "failed_over", "auth", 401),
        ("grok", "success", None, None),
    ]


async def test_bad_request_is_terminal(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {
            "claude": _fail("claude", "anthropic/c", "bad_request", 400),
            "grok": _ok("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert not out.answer.ok and out.name == "claude"
    assert calls == ["claude"]  # grok was never consulted
    assert [a.outcome for a in out.attempts] == ["terminal_failure"]


async def test_malformed_is_terminal(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {
            "claude": _fail("claude", "anthropic/c", "malformed_response"),
            "grok": _ok("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("judge", "sys", "user")
    assert calls == ["claude"] and out.attempts[0].outcome == "terminal_failure"


async def test_unkeyed_candidate_is_skipped_without_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    calls = _install(monkeypatch, {"grok": _ok("grok", "xai/g")})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert calls == ["grok"]
    assert [a.outcome for a in out.attempts] == ["skipped_unkeyed", "success"]


async def test_chain_exhausted(monkeypatch, keys):
    calls = _install(
        monkeypatch,
        {
            "claude": _fail("claude", "anthropic/c", "quota", 429),
            "grok": _fail("grok", "xai/g", "unavailable", 503),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert calls == ["claude", "grok"]
    assert not out.answer.ok and out.name == "grok"
    assert [a.outcome for a in out.attempts] == ["failed_over", "exhausted"]


async def test_all_unkeyed_returns_no_answer(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    calls = _install(monkeypatch, {})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer is None and calls == []
    assert [a.outcome for a in out.attempts] == ["skipped_unkeyed", "skipped_unkeyed"]


def test_record_adjudication_appends_ledger_and_receipts():
    c = Council(models=["grok"], synthesizer="claude", config=CFG)
    result = CouncilResult(
        prompt="p",
        manifest=ModelHarnessManifest(request_id="r", conclave_version="t", mode="synthesize"),
    )
    attempts = [
        AdjudicationAttempt(
            role="synthesis",
            candidate="claude",
            model_id="anthropic/c",
            attempt_index=1,
            outcome="failed_over",
            failure_category="auth",
            http_status=401,
        ),
        AdjudicationAttempt(
            role="synthesis", candidate="grok", model_id="xai/g", attempt_index=2, outcome="success"
        ),
    ]
    called = [_fail("claude", "anthropic/c", "auth", 401), _ok("grok", "xai/g")]
    c._record_adjudication(result, attempts, called, phase="synthesis")
    m = result.manifest
    assert m.adjudication_succession == attempts
    assert [(r.phase, r.attempt, r.outcome) for r in m.receipts] == [
        ("synthesis", 1, "failed"),
        ("synthesis", 2, "success"),
    ]
    assert m.secret_safety == "verified_no_secrets"


def test_ledger_has_no_free_text_fields():
    fields = set(AdjudicationAttempt.model_fields)
    assert fields == {
        "role",
        "candidate",
        "model_id",
        "attempt_index",
        "outcome",
        "failure_category",
        "http_status",
    }
