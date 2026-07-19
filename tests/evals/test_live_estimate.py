from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

import conclave.config as config_module
import conclave.evals.live as live_module
import conclave.registry as registry_module
import conclave.transport as transport_module
from conclave.evals.live import (
    BudgetExceededError,
    LiveProviderClient,
    _LiveEstimateClient,
    estimate_live_study,
)
from conclave.evals.live_protocols import (
    ChatMessage,
    StageCall,
    allocate_stage_caps,
    stage_call_sequence,
)
from conclave.evals.pricing import ModelPrice, PriceBook
from conclave.models import ModelAnswer, TokenUsage
from conclave.verdict_synthesis import VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES
from tests.evals.test_live_runner import _live_inputs


@pytest.mark.asyncio
async def test_dry_run_walks_same_calls_without_loading_keys_or_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks, manifest, price_book = _live_inputs()

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry-run must not load configuration or provider keys")

    async def forbidden_async(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry-run must not touch provider or transport seams")

    monkeypatch.setattr(config_module, "load_config", forbidden)
    monkeypatch.setattr(registry_module, "key_present", forbidden)
    monkeypatch.setattr(live_module, "call_model", forbidden_async)
    monkeypatch.setattr(transport_module, "post_json", forbidden_async)

    estimate = await estimate_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
    )

    assert estimate.planned_cells == len(manifest.planned_runs)


@pytest.mark.asyncio
async def test_dry_run_reports_calls_costs_largest_reservation_and_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks, manifest, price_book = _live_inputs()
    observed_caps: list[int] = []
    observed_costs: list[Decimal] = []
    real_reserve = live_module.reserve_call_cost

    def recording_reserve(*args, **kwargs):
        reservation = real_reserve(*args, **kwargs)
        observed_caps.append(kwargs["max_output_tokens"])
        observed_costs.append(reservation.reserved_cost_usd)
        return reservation

    monkeypatch.setattr(live_module, "reserve_call_cost", recording_reserve)

    estimate = await estimate_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
    )

    roster_sizes = {
        roster.roster_id: len(roster.members) for roster in manifest.frozen_design.rosters
    }
    expected_caps = tuple(
        cap
        for planned_run in manifest.planned_runs
        for cap in allocate_stage_caps(
            planned_run.condition_id,
            roster_size=roster_sizes[planned_run.roster_id],
            cell_ceiling=planned_run.max_output_tokens,
        )
    )
    expected_calls = sum(
        len(
            stage_call_sequence(
                planned_run.condition_id,
                roster_size=roster_sizes[planned_run.roster_id],
            )
        )
        for planned_run in manifest.planned_runs
    )

    assert tuple(observed_caps) == expected_caps
    assert estimate.maximum_call_count == expected_calls == len(observed_costs)
    assert estimate.largest_reservation_usd == max(observed_costs)
    assert estimate.total_upper_bound_usd == sum(observed_costs, Decimal("0"))
    assert estimate.ceiling_usd == Decimal("10.00")
    assert estimate.headroom_usd == estimate.ceiling_usd - estimate.total_upper_bound_usd
    assert estimate.fits_ceiling is True
    with pytest.raises(ValidationError):
        estimate.maximum_call_count = 0


@pytest.mark.asyncio
async def test_dry_run_prices_the_optional_elite_verdict_repair_max_graph() -> None:
    tasks, manifest, price_book = _live_inputs()

    estimate = await estimate_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
    )

    elite_max_graph = stage_call_sequence("elite_full", roster_size=3)
    assert elite_max_graph[-2:] == (("verdict", 0), ("verdict_repair", 0))
    expected_calls = sum(
        len(stage_call_sequence(run.condition_id, roster_size=3)) for run in manifest.planned_runs
    )
    assert estimate.maximum_call_count == expected_calls


@pytest.mark.asyncio
async def test_dry_run_models_maximum_bounded_verdict_repair_error() -> None:
    price_book = PriceBook(
        snapshot_id="fictional-repair-bound",
        captured_at="2026-07-18T12:00:00Z",
        currency="USD",
        entries=(
            ModelPrice(
                provider_id="fictional-provider",
                model_id="fictional/model",
                model_revision="fixture-r1",
                input_ceiling_usd_per_million_tokens=Decimal("1"),
                output_ceiling_usd_per_million_tokens=Decimal("1"),
                max_output_bytes_per_token=4,
            ),
        ),
    )
    client = _LiveEstimateClient(price_book)
    verdict = StageCall(
        stage="verdict",
        provider_id="fictional-provider",
        model_id="fictional/model",
        model_revision="fixture-r1",
        messages=(ChatMessage(role="user", content="extract verdict"),),
        max_output_tokens=2,
    )

    answer = await client.call(verdict)

    assert answer.error is not None
    assert len(answer.error.encode("utf-8")) == VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES


@pytest.mark.asyncio
async def test_dry_run_breaks_down_upper_bound_by_roster_and_condition() -> None:
    tasks, manifest, price_book = _live_inputs()

    estimate = await estimate_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
    )

    assert set(estimate.per_roster_upper_bound_usd) == {
        roster.roster_id for roster in manifest.frozen_design.rosters
    }
    assert set(estimate.per_condition_upper_bound_usd) == {
        planned_run.condition_id for planned_run in manifest.planned_runs
    }
    assert sum(estimate.per_roster_upper_bound_usd.values(), Decimal("0")) == (
        estimate.total_upper_bound_usd
    )
    assert sum(estimate.per_condition_upper_bound_usd.values(), Decimal("0")) == (
        estimate.total_upper_bound_usd
    )
    assert all(cost > 0 for cost in estimate.per_roster_upper_bound_usd.values())
    assert all(cost > 0 for cost in estimate.per_condition_upper_bound_usd.values())


@pytest.mark.asyncio
async def test_dry_run_rejects_plan_whose_worst_case_exceeds_frozen_ceiling() -> None:
    tasks, manifest, price_book = _live_inputs(rate="100000000")

    with pytest.raises(BudgetExceededError, match="worst-case.*frozen ceiling"):
        await estimate_live_study(
            manifest=manifest,
            tasks=tasks,
            price_book=price_book,
        )


@pytest.mark.asyncio
async def test_estimate_bounds_multibyte_max_expansion_execution_reservation() -> None:
    price_book = PriceBook(
        snapshot_id="fictional-byte-bound",
        captured_at="2026-07-18T12:00:00Z",
        currency="USD",
        entries=(
            ModelPrice(
                provider_id="fictional-provider",
                model_id="fictional/model",
                model_revision="fixture-r1",
                input_ceiling_usd_per_million_tokens=Decimal("1"),
                output_ceiling_usd_per_million_tokens=Decimal("1"),
                max_output_bytes_per_token=4,
            ),
        ),
    )
    first_call = StageCall(
        stage="draft",
        provider_id="fictional-provider",
        model_id="fictional/model",
        model_revision="fixture-r1",
        messages=(ChatMessage(role="user", content="static prompt"),),
        max_output_tokens=2,
    )
    estimate_client = _LiveEstimateClient(price_book)
    estimated_answer = await estimate_client.call(first_call)
    estimated_second_call = StageCall(
        stage="self_revision",
        provider_id="fictional-provider",
        model_id="fictional/model",
        model_revision="fixture-r1",
        messages=(
            ChatMessage(
                role="user",
                content=f"static wrapper:{estimated_answer.answer}:end",
            ),
        ),
        max_output_tokens=1,
        upstream_output_token_ceilings=(2,),
    )
    await estimate_client.call(estimated_second_call)
    estimated_reservation = estimate_client.reservations[-1]

    pending_reservations = []
    provider_answers = iter(("😀😀", "done"))

    async def provider(name, model_id, messages, **kwargs):
        del messages, kwargs
        answer = next(provider_answers)
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer=answer,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def checkpoint(pending, receipts) -> None:
        del receipts
        if pending is not None:
            pending_reservations.append(pending.reservation)

    execution_client = LiveProviderClient(
        price_book=price_book,
        hard_cap_usd=Decimal("1"),
        checkpoint=checkpoint,
        call_model_func=provider,
    )
    actual_answer = await execution_client.call(first_call)
    actual_second_call = estimated_second_call.model_copy(
        update={
            "messages": (
                ChatMessage(
                    role="user",
                    content=f"static wrapper:{actual_answer.answer}:end",
                ),
            )
        }
    )
    await execution_client.call(actual_second_call)
    execution_reservation = pending_reservations[-1]

    assert len("😀😀".encode()) == 2 * price_book.entries[0].max_output_bytes_per_token
    assert execution_reservation.input_token_upper_bound <= (
        estimated_reservation.input_token_upper_bound
    )
    assert execution_reservation.reserved_cost_usd <= estimated_reservation.reserved_cost_usd
