# tests/test_pricing_core.py
"""Shared Decimal/ROUND_CEILING price arithmetic (DSE-1514, Round 2)."""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

import pytest
from pydantic import ValidationError

from conclave.pricing import (
    USD_MICROCENT,
    PriceRates,
    reported_usage_cost,
    reserve_cost,
)

RATES = PriceRates(
    provider_id="fictional-provider-a",
    model_id="fictional-provider-a/fictional-model-a",
    input_ceiling_usd_per_million_tokens=Decimal("1.234567"),
    output_ceiling_usd_per_million_tokens=Decimal("4.567891"),
    max_output_bytes_per_token=4,
)


def test_rates_reject_float_input_and_output_ceilings():
    with pytest.raises(ValidationError):
        PriceRates(
            provider_id="p",
            model_id="p/m",
            input_ceiling_usd_per_million_tokens=1.23,
            output_ceiling_usd_per_million_tokens=Decimal("2"),
            max_output_bytes_per_token=4,
        )
    with pytest.raises(ValidationError):
        PriceRates(
            provider_id="p",
            model_id="p/m",
            input_ceiling_usd_per_million_tokens=Decimal("1"),
            output_ceiling_usd_per_million_tokens=2.5,
            max_output_bytes_per_token=4,
        )


def test_reserve_cost_covers_prompt_template_framing_and_upstream():
    amounts = reserve_cost(
        RATES,
        prompt_token_upper_bound=1_000,
        prompt_template_token_allowance=100,
        provider_framing_token_allowance=96,
        upstream_output_token_ceilings=(500, 500),
        upstream_output_bytes_per_token=4,
        max_output_tokens=2_000,
    )
    assert amounts.input_token_upper_bound == 1_000 + 100 + 96 + (1_000 * 4)
    assert amounts.output_token_upper_bound == 2_000
    assert amounts.reserved_cost_usd == (
        amounts.input_cost_upper_bound_usd + amounts.output_cost_upper_bound_usd
    ).quantize(USD_MICROCENT, rounding=ROUND_CEILING)
    assert isinstance(amounts.reserved_cost_usd, Decimal)


def test_reserve_cost_always_rounds_up():
    tiny = PriceRates(
        provider_id="p",
        model_id="p/m",
        input_ceiling_usd_per_million_tokens=Decimal("1"),
        output_ceiling_usd_per_million_tokens=Decimal("1"),
        max_output_bytes_per_token=4,
    )
    amounts = reserve_cost(
        tiny,
        prompt_token_upper_bound=1,
        prompt_template_token_allowance=0,
        provider_framing_token_allowance=0,
        upstream_output_token_ceilings=(),
        upstream_output_bytes_per_token=4,
        max_output_tokens=1,
    )
    # 2 tokens at $1/M is $0.000002 exactly; a sub-microcent residue must round UP.
    assert amounts.reserved_cost_usd == Decimal("0.000002")


def test_reported_usage_cost_charges_unattributed_at_the_higher_rate():
    cost = reported_usage_cost(
        RATES,
        prompt_tokens=1_000,
        completion_tokens=2_000,
        total_tokens=3_500,
    )
    expected = (
        Decimal(1_000) * RATES.input_ceiling_usd_per_million_tokens
        + Decimal(2_000) * RATES.output_ceiling_usd_per_million_tokens
        + Decimal(500) * RATES.output_ceiling_usd_per_million_tokens
    ) / Decimal(1_000_000)
    assert cost == expected.quantize(USD_MICROCENT, rounding=ROUND_CEILING)


def test_reported_usage_cost_rejects_impossible_totals():
    with pytest.raises(ValueError, match="smaller than its attributed usage"):
        reported_usage_cost(RATES, prompt_tokens=10, completion_tokens=10, total_tokens=5)


def test_token_bounds_reject_bools_and_negatives():
    with pytest.raises(TypeError):
        reserve_cost(
            RATES,
            prompt_token_upper_bound=True,
            prompt_template_token_allowance=0,
            provider_framing_token_allowance=0,
            upstream_output_token_ceilings=(),
            upstream_output_bytes_per_token=4,
            max_output_tokens=1,
        )
    with pytest.raises(ValueError, match="max_output_tokens must be positive"):
        reserve_cost(
            RATES,
            prompt_token_upper_bound=0,
            prompt_template_token_allowance=0,
            provider_framing_token_allowance=0,
            upstream_output_token_ceilings=(),
            upstream_output_bytes_per_token=4,
            max_output_tokens=0,
        )


def test_eval_reserve_call_cost_delegates_to_the_shared_arithmetic():
    from conclave.evals.pricing import ModelPrice, reserve_call_cost

    price = ModelPrice(
        provider_id="fictional-provider-a",
        model_id="fictional-model-a",
        model_revision="fixture-r1",
        input_ceiling_usd_per_million_tokens=Decimal("1.234567"),
        output_ceiling_usd_per_million_tokens=Decimal("4.567891"),
        max_output_bytes_per_token=4,
    )
    rates = price.as_rates()
    assert isinstance(rates, PriceRates)
    assert rates.model_id == "fictional-provider-a/fictional-model-a"
    assert rates.source_url is None

    kwargs = {
        "prompt_token_upper_bound": 900,
        "prompt_template_token_allowance": 12,
        "provider_framing_token_allowance": 96,
        "upstream_output_token_ceilings": (256,),
        "upstream_output_bytes_per_token": 4,
        "max_output_tokens": 1_024,
    }
    reservation = reserve_call_cost(price, **kwargs)
    amounts = reserve_cost(rates, **kwargs)

    assert reservation.reserved_cost_usd == amounts.reserved_cost_usd
    assert reservation.input_token_upper_bound == amounts.input_token_upper_bound
    assert reservation.input_cost_upper_bound_usd == amounts.input_cost_upper_bound_usd
    assert reservation.output_cost_upper_bound_usd == amounts.output_cost_upper_bound_usd
    # The eval CallReservation contract is unchanged: it still carries revision
    # identity and the eval schema_version that hash_price_entries depends on.
    assert reservation.model_revision == "fixture-r1"
    assert reservation.schema_version == "conclave_eval_v1"
