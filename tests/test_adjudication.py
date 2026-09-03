"""Council.adjudicate walks the synthesizer chain with the infra-only failover rule."""

from __future__ import annotations

from conclave.config import ConclaveConfig
from conclave.council import Council
from conclave.manifest import AdjudicationAttempt, ModelHarnessManifest
from conclave.models import CouncilResult
from tests.conftest import install_council_script, make_failed_answer, make_ok_answer

CFG = ConclaveConfig(models={"claude": "anthropic/c", "grok": "xai/g", "gemini": "gemini/m"})


async def test_chain_of_one_success(monkeypatch, keys):
    calls = install_council_script(monkeypatch, {"claude": make_ok_answer("claude", "anthropic/c")})
    c = Council(models=["grok"], synthesizer="claude", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer.ok and out.name == "claude" and calls == ["claude"]
    assert [a.outcome for a in out.attempts] == ["success"]


async def test_auth_failure_advances(monkeypatch, keys):
    calls = install_council_script(
        monkeypatch,
        {
            "claude": make_failed_answer("claude", "anthropic/c", "auth", 401),
            "grok": make_ok_answer("grok", "xai/g"),
        },
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
    calls = install_council_script(
        monkeypatch,
        {
            "claude": make_failed_answer("claude", "anthropic/c", "bad_request", 400),
            "grok": make_ok_answer("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert not out.answer.ok and out.name == "claude"
    assert calls == ["claude"]  # grok was never consulted
    assert [a.outcome for a in out.attempts] == ["terminal_failure"]


async def test_malformed_is_terminal(monkeypatch, keys):
    calls = install_council_script(
        monkeypatch,
        {
            "claude": make_failed_answer("claude", "anthropic/c", "malformed_response"),
            "grok": make_ok_answer("grok", "xai/g"),
        },
    )
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("judge", "sys", "user")
    assert calls == ["claude"] and out.attempts[0].outcome == "terminal_failure"


async def test_unkeyed_candidate_is_skipped_without_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    calls = install_council_script(monkeypatch, {"grok": make_ok_answer("grok", "xai/g")})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert calls == ["grok"]
    assert [a.outcome for a in out.attempts] == ["skipped_unkeyed", "success"]


async def test_chain_exhausted(monkeypatch, keys):
    calls = install_council_script(
        monkeypatch,
        {
            "claude": make_failed_answer("claude", "anthropic/c", "quota", 429),
            "grok": make_failed_answer("grok", "xai/g", "unavailable", 503),
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
    calls = install_council_script(monkeypatch, {})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    out = await c.adjudicate("synthesis", "sys", "user")
    assert out.answer is None and calls == []
    assert [a.outcome for a in out.attempts] == ["skipped_unkeyed", "skipped_unkeyed"]


async def test_synthesize_blocks_all_unkeyed_synthetic_answer_is_typed(monkeypatch):
    """The synthetic error answer for an all-unkeyed chain is typed 'unkeyed'.

    ``synthesize_blocks`` fabricates a ``ModelAnswer`` when no chain candidate
    has a key. It must carry ``failure_category="unkeyed"`` like every other
    unkeyed outcome, not the pre-DSE-1512 untyped default of ``None``.
    """
    for var in ("ANTHROPIC_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    install_council_script(monkeypatch, {})
    c = Council(models=["gemini"], synthesizer="claude>grok", config=CFG)
    answer = await c.synthesize_blocks("sys", "user")
    assert not answer.ok
    assert answer.failure_category == "unkeyed"


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
    called = [
        make_failed_answer("claude", "anthropic/c", "auth", 401),
        make_ok_answer("grok", "xai/g"),
    ]
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
