"""The pre-flight spend gate refuses BEFORE the first provider call (DSE-1514)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from conclave.council import Council
from conclave.pricing import SpendCapExceeded, SpendRefused, SpendUnboundable
from tests.test_pricing_receipts import _install_snapshot, _snapshot


def test_a_spend_cap_without_an_output_cap_is_refused_at_construction(keys):
    with pytest.raises(SpendUnboundable) as excinfo:
        Council(models=["grok"], synthesizer="grok", max_spend_usd=Decimal("0.40"))
    assert str(excinfo.value) == (
        "cannot bound spend: no output cap (set --max-output-tokens or config max_output_tokens)"
    )
    assert isinstance(excinfo.value, SpendRefused)


async def test_an_over_budget_plan_refuses_before_any_provider_call(monkeypatch, keys):
    import conclave.council as council_mod

    _install_snapshot(
        monkeypatch,
        _snapshot("xai/grok-4.3", "gemini/gemini-2.5-pro", "anthropic/claude-sonnet-4-6"),
    )
    calls: list[str] = []

    async def tripwire(name, model_id, messages, **kwargs):
        calls.append(name)
        raise AssertionError("the gate must refuse before any provider call")

    monkeypatch.setattr(council_mod, "call_model", tripwire)
    council = Council(
        models=["grok", "gemini"],
        synthesizer="claude",
        max_output_tokens=100_000,
        max_spend_usd=Decimal("0.000001"),
    )
    with pytest.raises(SpendCapExceeded) as excinfo:
        await council.ask("q")

    error = excinfo.value
    assert error.cap == Decimal("0.000001")
    assert error.reserved > error.cap
    assert error.call_count == 2 + 1 + 2
    assert "reserved" in str(error) and "cap" in str(error) and "calls" in str(error)
    assert calls == []


async def test_an_under_budget_plan_runs_normally(monkeypatch, keys, patch_call_model):
    from tests.conftest import make_response

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
        extract_verdict=False,
    )
    result = await council.ask("q", synthesize=False)
    assert result.successful_answers
    assert result.manifest.cost_ceiling_usd is not None


async def test_an_unpriced_model_in_the_plan_refuses_rather_than_guessing(monkeypatch, keys):
    import conclave.council as council_mod

    # grok is priced; the claude synthesizer is not.
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def tripwire(name, model_id, messages, **kwargs):
        raise AssertionError("no call may happen")

    monkeypatch.setattr(council_mod, "call_model", tripwire)
    council = Council(
        models=["grok"],
        synthesizer="claude",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
    )
    with pytest.raises(SpendUnboundable, match="no priced rate for anthropic/claude-sonnet-4-6"):
        await council.ask("q")


async def test_a_missing_snapshot_refuses_the_gate(monkeypatch, keys):
    _install_snapshot(monkeypatch, None)
    council = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
    )
    with pytest.raises(SpendUnboundable, match="price snapshot unavailable"):
        await council.ask("q", synthesize=False)


async def test_a_cache_hit_is_never_gated(monkeypatch, tmp_path, keys, patch_call_model):
    from tests.conftest import make_response

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    patch_call_model(lambda model_id, messages: make_response("ok"))

    cheap = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("100.00"),
        cache=True,
        extract_verdict=False,
    )
    await cheap.ask("q", synthesize=False)

    # Same identity, an impossible cap: the hit costs nothing, so it is served.
    strict = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=64,
        max_spend_usd=Decimal("0.000001"),
        cache=True,
        extract_verdict=False,
    )
    hit = await strict.ask("q", synthesize=False)
    assert hit.cached is True


async def test_the_gate_also_guards_the_streaming_path(monkeypatch, keys):
    import conclave.streaming as streaming_mod

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def tripwire(*args, **kwargs):
        raise AssertionError("no stream may start")
        yield  # pragma: no cover

    monkeypatch.setattr(streaming_mod, "call_model_stream", tripwire)
    council = Council(
        models=["grok"],
        synthesizer="grok",
        max_output_tokens=100_000,
        max_spend_usd=Decimal("0.000001"),
        extract_verdict=False,
    )
    with pytest.raises(SpendCapExceeded):
        async for _event in council.ask_stream("q", synthesize=False):
            pass


def test_no_spend_flags_means_no_gate_at_all(keys):
    council = Council(models=["grok"], synthesizer="grok")
    assert council.max_spend_usd is None
    assert council.max_output_tokens is None
