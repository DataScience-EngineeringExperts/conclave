from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

import conclave.config as config_module
import conclave.evals.live as live_module
import conclave.registry as registry_module
import conclave.transport as transport_module
from conclave.evals.live import BudgetExceededError, estimate_live_study
from conclave.evals.live_protocols import allocate_stage_caps, stage_call_sequence
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
