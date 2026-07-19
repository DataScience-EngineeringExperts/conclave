from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from pydantic import ValidationError

import conclave.evals.live as live_module
from conclave.evals.live import (
    BudgetExceededError,
    GatewayStoppedError,
    LiveProviderClient,
    ProviderCallReceipt,
    ReservationBreachError,
)
from conclave.evals.live_protocols import ChatMessage, StageCall
from conclave.evals.pricing import ModelPrice, PriceBook
from conclave.models import ModelAnswer, TokenUsage


def _price_book(
    *,
    input_rate: str = "1000000",
    output_rate: str = "1000000",
) -> PriceBook:
    return PriceBook(
        snapshot_id="fictional-gateway-prices",
        captured_at="2026-07-18T12:00:00Z",
        currency="USD",
        entries=(
            ModelPrice(
                provider_id="fictional-provider",
                model_id="fictional/model",
                model_revision="fixture-r1",
                input_ceiling_usd_per_million_tokens=Decimal(input_rate),
                output_ceiling_usd_per_million_tokens=Decimal(output_rate),
            ),
        ),
    )


def _stage_call(*, cap: int = 5, content: str = "x") -> StageCall:
    return StageCall(
        stage="initial",
        provider_id="fictional-provider",
        model_id="fictional/model",
        model_revision="fixture-r1",
        messages=(ChatMessage(role="user", content=content),),
        max_output_tokens=cap,
    )


@pytest.mark.asyncio
async def test_gateway_persists_reservation_before_calling_provider() -> None:
    checkpoints: list[tuple[object | None, tuple[ProviderCallReceipt, ...]]] = []
    observed_caps: list[int | None] = []
    client: LiveProviderClient

    async def checkpoint(pending, receipts) -> None:
        checkpoints.append((pending, receipts))

    async def fake_call_model(name, model_id, messages, **kwargs):
        assert client.pending_call is not None
        assert checkpoints[-1][0] == client.pending_call
        assert checkpoints[-1][1] == ()
        assert messages == [{"role": "user", "content": "x"}]
        observed_caps.append(kwargs.get("max_output_tokens"))
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="ok",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    client = LiveProviderClient(
        price_book=_price_book(),
        hard_cap_usd=Decimal("100"),
        call_model_func=fake_call_model,
        checkpoint=checkpoint,
    )

    answer = await client.call(_stage_call(cap=5))

    assert answer.ok
    assert observed_caps == [5]
    assert checkpoints[0][0] is not None
    assert checkpoints[-1][0] is None
    assert len(checkpoints[-1][1]) == 1


@pytest.mark.asyncio
async def test_gateway_allows_only_one_in_flight_call() -> None:
    active = 0
    maximum_active = 0

    async def fake_call_model(name, model_id, messages, **kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="ok",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    client = LiveProviderClient(
        price_book=_price_book(input_rate="1", output_rate="1"),
        hard_cap_usd=Decimal("1"),
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: None,
    )

    answers = await asyncio.gather(client.call(_stage_call()), client.call(_stage_call()))

    assert all(answer.ok for answer in answers)
    assert maximum_active == 1
    assert len(client.receipts) == 2


@pytest.mark.asyncio
async def test_gateway_rejects_call_that_would_cross_hard_cap() -> None:
    provider_calls = 0

    async def fake_call_model(name, model_id, messages, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return ModelAnswer(name=name, model_id=model_id, answer="must not run")

    client = LiveProviderClient(
        price_book=_price_book(),
        hard_cap_usd=Decimal("60"),
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: None,
    )

    with pytest.raises(BudgetExceededError, match="hard cap"):
        await client.call(_stage_call(cap=5))

    assert provider_calls == 0
    assert client.pending_call is None
    assert client.receipts == ()
    assert client.committed_cost_usd == Decimal("0")


@pytest.mark.asyncio
async def test_gateway_prices_complete_usage_and_charges_reservation_when_missing() -> None:
    answers = iter(
        (
            ModelAnswer(
                name="fictional-provider",
                model_id="fictional/model",
                answer="metered",
                usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            ),
            ModelAnswer(
                name="fictional-provider",
                model_id="fictional/model",
                answer="unmetered",
            ),
        )
    )

    async def fake_call_model(name, model_id, messages, **kwargs):
        return next(answers)

    client = LiveProviderClient(
        price_book=_price_book(),
        hard_cap_usd=Decimal("200"),
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: None,
    )

    await client.call(_stage_call(cap=5))
    await client.call(_stage_call(cap=5))

    metered, unmetered = client.receipts
    assert metered.cost_basis.source == "reported_usage"
    assert metered.cost_basis.prompt_tokens == 2
    assert metered.cost_basis.completion_tokens == 3
    assert metered.charged_cost_usd == Decimal("5.000000")
    assert unmetered.cost_basis.source == "full_reservation"
    assert unmetered.cost_basis.prompt_tokens is None
    assert unmetered.cost_basis.completion_tokens is None
    assert unmetered.charged_cost_usd == unmetered.reserved_cost_usd
    assert client.committed_cost_usd == (metered.charged_cost_usd + unmetered.charged_cost_usd)


@pytest.mark.asyncio
async def test_gateway_stops_on_reservation_breach() -> None:
    provider_calls = 0

    async def fake_call_model(name, model_id, messages, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="provider exceeded cap",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=6, total_tokens=7),
        )

    client = LiveProviderClient(
        price_book=_price_book(input_rate="1", output_rate="1"),
        hard_cap_usd=Decimal("1"),
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: None,
    )

    with pytest.raises(ReservationBreachError, match="reservation"):
        await client.call(_stage_call(cap=5))

    assert client.stopped is True
    assert client.receipts[-1].outcome == "failed"
    assert client.receipts[-1].error_category == "reservation_breach"
    with pytest.raises(GatewayStoppedError, match="stopped"):
        await client.call(_stage_call(cap=5))
    assert provider_calls == 1


@pytest.mark.asyncio
async def test_gateway_records_actual_cost_when_breach_exceeds_hard_cap() -> None:
    hard_cap = Decimal("0.000086")

    async def fake_call_model(name, model_id, messages, **kwargs):
        del messages, kwargs
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="provider exceeded the reservation and hard cap",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=6, total_tokens=106),
        )

    client = LiveProviderClient(
        price_book=_price_book(input_rate="1", output_rate="1"),
        hard_cap_usd=hard_cap,
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: None,
    )

    with pytest.raises(ReservationBreachError, match="reservation"):
        await client.call(_stage_call(cap=5))

    receipt = client.receipts[-1]
    assert receipt.reserved_cost_usd == hard_cap
    assert receipt.charged_cost_usd == Decimal("0.000106")
    assert receipt.charged_cost_usd > receipt.reserved_cost_usd
    assert receipt.charged_cost_usd > hard_cap
    assert receipt.cost_basis.source == "reported_usage"
    assert receipt.error_category == "reservation_breach"
    assert receipt.hard_cap_breached is True
    assert client.committed_cost_usd == receipt.charged_cost_usd


def test_token_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1, completion_tokens=0, total_tokens=0)


@pytest.mark.parametrize(
    "usage",
    (
        TokenUsage.model_construct(prompt_tokens=-1, completion_tokens=1, total_tokens=0),
        TokenUsage.model_construct(prompt_tokens="malformed", completion_tokens=1, total_tokens=1),
    ),
)
@pytest.mark.asyncio
async def test_gateway_persists_reconciliation_error_for_malformed_usage(usage) -> None:
    checkpoints: list[tuple[object | None, tuple[ProviderCallReceipt, ...]]] = []

    async def fake_call_model(name, model_id, messages, **kwargs):
        del messages, kwargs
        return ModelAnswer.model_construct(
            name=name,
            model_id=model_id,
            answer="malformed usage fixture",
            usage=usage,
            error=None,
        )

    client = LiveProviderClient(
        price_book=_price_book(input_rate="1", output_rate="1"),
        hard_cap_usd=Decimal("1"),
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: checkpoints.append((pending, receipts)),
    )

    with pytest.raises(GatewayStoppedError, match="reconciliation"):
        await client.call(_stage_call())

    receipt = client.receipts[-1]
    assert receipt.error_category == "reconciliation_error"
    assert receipt.outcome == "failed"
    assert receipt.usage is None
    assert receipt.charged_cost_usd == receipt.reserved_cost_usd
    assert receipt.cost_basis.source == "full_reservation"
    assert client.pending_call is None
    assert client.stopped is True
    assert checkpoints[-1] == (None, client.receipts)


@pytest.mark.asyncio
async def test_gateway_persists_reconciliation_error_when_costing_raises(monkeypatch) -> None:
    async def fake_call_model(name, model_id, messages, **kwargs):
        del messages, kwargs
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="valid provider response",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def broken_costing(price, usage):
        del price, usage
        raise ArithmeticError("unbounded internal fixture detail")

    monkeypatch.setattr(live_module, "_reported_usage_cost", broken_costing)
    client = LiveProviderClient(
        price_book=_price_book(input_rate="1", output_rate="1"),
        hard_cap_usd=Decimal("1"),
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: None,
    )

    with pytest.raises(GatewayStoppedError, match="reconciliation"):
        await client.call(_stage_call())

    receipt = client.receipts[-1]
    assert receipt.error_category == "reconciliation_error"
    assert receipt.charged_cost_usd == receipt.reserved_cost_usd
    assert "unbounded internal fixture detail" not in receipt.model_dump_json()


@pytest.mark.asyncio
async def test_gateway_receipt_contains_bounded_error_category_not_raw_exception() -> None:
    secret = "sk-live-secret-value"
    raw_error = (
        f"401 from https://provider.invalid/private Authorization: Bearer {secret}; "
        "x-api-key: raw-secret"
    )

    async def fake_call_model(name, model_id, messages, **kwargs):
        raise RuntimeError(raw_error)

    client = LiveProviderClient(
        price_book=_price_book(input_rate="1", output_rate="1"),
        hard_cap_usd=Decimal("1"),
        call_model_func=fake_call_model,
        checkpoint=lambda pending, receipts: None,
    )

    answer = await client.call(_stage_call())
    receipt = client.receipts[-1]
    payload = receipt.model_dump_json()

    assert answer.error == "authentication"
    assert receipt.outcome == "failed"
    assert receipt.error_category == "authentication"
    assert receipt.cost_basis.source == "full_reservation"
    for forbidden in (secret, raw_error, "https://", "Authorization", "Bearer", "x-api-key"):
        assert forbidden not in payload
    with pytest.raises(ValidationError):
        ProviderCallReceipt.model_validate({**receipt.model_dump(), "raw_exception": raw_error})
