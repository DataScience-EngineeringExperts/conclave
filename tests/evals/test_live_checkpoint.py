from __future__ import annotations

import json
import os
from decimal import Decimal

import pytest

import conclave.evals.live as live_module
from conclave.adapters.base import redact as existing_redact
from conclave.evals.live import (
    ActiveCell,
    CheckpointBindings,
    CheckpointSecurityError,
    CheckpointValidationError,
    GatewayStoppedError,
    LiveProviderClient,
    PendingCall,
    ProviderCallCostBasis,
    ProviderCallReceipt,
    ReservationBreachError,
    create_live_checkpoint,
    finish_active_cell,
    load_live_checkpoint,
    recover_interrupted_checkpoint,
    start_active_cell,
    update_live_checkpoint,
    write_live_checkpoint,
)
from conclave.evals.live_protocols import ChatMessage, StageCall
from conclave.evals.models import RunRecord
from conclave.evals.pricing import ModelPrice, PriceBook, reserve_call_cost
from conclave.models import ModelAnswer, TokenUsage

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
RUN_A = "run_" + "a" * 24
RUN_B = "run_" + "b" * 24


def _bindings(**updates: object) -> CheckpointBindings:
    values: dict[str, object] = {
        "manifest_hash": DIGEST_A,
        "price_book_hash": DIGEST_B,
        "public_tasks_hash": DIGEST_C,
        "hard_cap_usd": Decimal("10.00"),
    }
    values.update(updates)
    return CheckpointBindings(**values)


def _price_book() -> PriceBook:
    return PriceBook(
        snapshot_id="fictional-checkpoint-prices",
        captured_at="2026-07-18T12:00:00Z",
        currency="USD",
        entries=(
            ModelPrice(
                provider_id="fictional-provider",
                model_id="fictional/model",
                model_revision="fixture-r1",
                input_ceiling_usd_per_million_tokens=Decimal("1"),
                output_ceiling_usd_per_million_tokens=Decimal("1"),
            ),
        ),
    )


def _stage_call() -> StageCall:
    return StageCall(
        stage="initial",
        provider_id="fictional-provider",
        model_id="fictional/model",
        model_revision="fixture-r1",
        messages=(ChatMessage(role="user", content="safe fixture prompt"),),
        max_output_tokens=5,
    )


def _pending_call() -> PendingCall:
    price = _price_book().entries[0]
    reservation = reserve_call_cost(
        price,
        prompt_token_upper_bound=10,
        prompt_template_token_allowance=0,
        provider_framing_token_allowance=2,
        upstream_output_token_ceilings=(),
        max_output_tokens=5,
    )
    return PendingCall(
        stage="initial",
        provider_id=price.provider_id,
        model_id=price.model_id,
        model_revision=price.model_revision,
        max_output_tokens=5,
        reservation=reservation,
    )


def _receipt(*, charged: str = "0.000002") -> ProviderCallReceipt:
    return ProviderCallReceipt(
        stage="initial",
        provider_id="fictional-provider",
        model_id="fictional/model",
        model_revision="fixture-r1",
        max_output_tokens=5,
        outcome="success",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        reserved_cost_usd=Decimal("0.000017"),
        charged_cost_usd=Decimal(charged),
        cost_basis=ProviderCallCostBasis(
            source="reported_usage",
            prompt_tokens=1,
            completion_tokens=1,
            input_ceiling_usd_per_million_tokens=Decimal("1"),
            output_ceiling_usd_per_million_tokens=Decimal("1"),
        ),
    )


@pytest.mark.asyncio
async def test_checkpoint_write_is_flush_fsync_replace_and_secret_scanned(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "nested" / "live-checkpoint.json"
    destination.parent.mkdir()
    checkpoint = create_live_checkpoint(bindings=_bindings())
    events: list[tuple[str, object]] = []
    redacted_payloads: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_redact(payload: str) -> str:
        redacted_payloads.append(payload)
        return existing_redact(payload)

    def tracking_fsync(fd: int) -> None:
        assert os.fstat(fd).st_size > 0
        events.append(("fsync", fd))
        real_fsync(fd)

    def tracking_replace(source, target) -> None:
        assert events and events[-1][0] == "fsync"
        assert os.fspath(source).startswith(os.fspath(destination.parent))
        events.append(("replace", os.fspath(target)))
        real_replace(source, target)

    monkeypatch.setattr(live_module, "redact", tracking_redact)
    monkeypatch.setattr(live_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(live_module.os, "replace", tracking_replace)

    write_live_checkpoint(destination, checkpoint)

    assert [event[0] for event in events] == ["fsync", "replace"]
    assert redacted_payloads and existing_redact(redacted_payloads[-1]) == redacted_payloads[-1]
    assert load_live_checkpoint(destination, expected_bindings=_bindings()) == checkpoint

    secret = "opaque-provider-key-fixture"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    unsafe = create_live_checkpoint(
        bindings=_bindings(),
        records=(RunRecord(planned_run_id=RUN_A, outcome="success", output=secret),),
    )
    replacements_before = len([event for event in events if event[0] == "replace"])
    with pytest.raises(CheckpointSecurityError, match="secret"):
        write_live_checkpoint(destination, unsafe)
    assert len([event for event in events if event[0] == "replace"]) == replacements_before
    assert load_live_checkpoint(destination, expected_bindings=_bindings()) == checkpoint

    provider_calls = 0
    persist_calls = 0

    async def provider(name, model_id, messages, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return ModelAnswer(name=name, model_id=model_id, answer="ok")

    def failing_persistence(pending, receipts) -> None:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise OSError("fixture persistence failure")

    client = LiveProviderClient(
        price_book=_price_book(),
        hard_cap_usd=Decimal("10"),
        checkpoint=failing_persistence,
        call_model_func=provider,
    )

    with pytest.raises(OSError, match="persistence failure"):
        await client.call(_stage_call())
    with pytest.raises(GatewayStoppedError, match="stopped"):
        await client.call(_stage_call())
    assert provider_calls == 1


def test_checkpoint_rejects_manifest_price_task_or_ceiling_drift(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = create_live_checkpoint(bindings=_bindings())
    equivalent = create_live_checkpoint(bindings=_bindings(hard_cap_usd=Decimal("10.0")))
    assert equivalent.checkpoint_hash == checkpoint.checkpoint_hash
    write_live_checkpoint(path, checkpoint)

    for field, changed in (
        ("manifest_hash", "sha256:" + "d" * 64),
        ("price_book_hash", "sha256:" + "e" * 64),
        ("public_tasks_hash", "sha256:" + "f" * 64),
        ("hard_cap_usd", Decimal("9.99")),
    ):
        expected = _bindings().model_copy(update={field: changed})
        with pytest.raises(CheckpointValidationError, match=field):
            load_live_checkpoint(path, expected_bindings=expected)


@pytest.mark.asyncio
async def test_resume_charges_pending_reservation_and_never_repeats_interrupted_cell(
    tmp_path,
) -> None:
    class SimulatedInterruption(BaseException):
        pass

    path = tmp_path / "checkpoint.json"
    state = start_active_cell(create_live_checkpoint(bindings=_bindings()), planned_run_id=RUN_A)

    def persist(pending, receipts) -> None:
        nonlocal state
        state = update_live_checkpoint(state, pending_call=pending, receipts=receipts)
        write_live_checkpoint(path, state)

    seen_cells: list[str] = []
    current_cell = RUN_A

    async def interrupted_provider(name, model_id, messages, **kwargs):
        seen_cells.append(current_cell)
        raise SimulatedInterruption

    client = LiveProviderClient(
        price_book=_price_book(),
        hard_cap_usd=Decimal("10"),
        checkpoint=persist,
        call_model_func=interrupted_provider,
    )
    with pytest.raises(SimulatedInterruption):
        await client.call(_stage_call())

    interrupted = load_live_checkpoint(path, expected_bindings=_bindings())
    assert interrupted.active_cell is not None
    assert interrupted.active_cell.pending_call is not None
    pending_cost = interrupted.active_cell.pending_call.reservation.reserved_cost_usd

    recovered = recover_interrupted_checkpoint(interrupted)
    write_live_checkpoint(path, recovered)
    resumed = load_live_checkpoint(path, expected_bindings=_bindings())

    async def resumed_provider(name, model_id, messages, **kwargs):
        seen_cells.append(current_cell)
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="resumed",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    resumed_client = LiveProviderClient(
        price_book=_price_book(),
        hard_cap_usd=Decimal("10"),
        checkpoint=lambda pending, receipts: None,
        call_model_func=resumed_provider,
        resume_from=resumed,
    )
    for planned_run_id in (RUN_A, RUN_B):
        if resumed.should_execute(planned_run_id):
            current_cell = planned_run_id
            await resumed_client.call(_stage_call())

    assert seen_cells == [RUN_A, RUN_B]
    interrupted_record = next(
        record for record in resumed.records if record.planned_run_id == RUN_A
    )
    assert interrupted_record.outcome == "incomplete"
    assert interrupted_record.deviation_codes == ("interrupted_cell_not_retried",)
    assert interrupted_record.cost_receipt_complete is True
    assert Decimal(str(interrupted_record.cost_usd)) == pending_cost
    assert resumed.committed_cost_usd == pending_cost
    assert resumed.receipts[-1].charged_cost_usd == pending_cost
    assert resumed.receipts[-1].cost_basis.source == "full_reservation"


@pytest.mark.asyncio
async def test_checkpoint_persists_honest_hard_cap_breach_evidence(tmp_path) -> None:
    hard_cap = Decimal("0.000104")
    bindings = _bindings(hard_cap_usd=hard_cap)
    path = tmp_path / "hard-cap-breach.json"
    state = start_active_cell(create_live_checkpoint(bindings=bindings), planned_run_id=RUN_A)

    def persist(pending, receipts) -> None:
        nonlocal state
        state = update_live_checkpoint(state, pending_call=pending, receipts=receipts)
        write_live_checkpoint(path, state)

    async def provider(name, model_id, messages, **kwargs):
        del messages, kwargs
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="over-cap fixture",
            usage=TokenUsage(prompt_tokens=200, completion_tokens=6, total_tokens=206),
        )

    client = LiveProviderClient(
        price_book=_price_book(),
        hard_cap_usd=hard_cap,
        checkpoint=persist,
        call_model_func=provider,
    )
    with pytest.raises(ReservationBreachError):
        await client.call(_stage_call())

    persisted = load_live_checkpoint(path, expected_bindings=bindings)
    assert persisted.committed_cost_usd == Decimal("0.000206")
    assert persisted.committed_cost_usd > hard_cap
    assert persisted.hard_cap_breached is True
    assert persisted.receipts[-1].hard_cap_breached is True


def test_resume_preserves_completed_records_and_call_receipts() -> None:
    completed_record = RunRecord(
        planned_run_id=RUN_A,
        outcome="success",
        output="completed fixture output",
        completion_tokens=1,
        cost_usd=0.000002,
        cost_receipt_complete=True,
    )
    completed_receipt = _receipt()
    pending = _pending_call()
    checkpoint = create_live_checkpoint(
        bindings=_bindings(),
        records=(completed_record,),
        receipts=(completed_receipt,),
        active_cell=ActiveCell(
            planned_run_id=RUN_B,
            receipt_start_index=1,
            pending_call=pending,
        ),
        committed_cost_usd=completed_receipt.charged_cost_usd,
    )

    recovered = recover_interrupted_checkpoint(checkpoint)

    assert recovered.records[0] == completed_record
    assert recovered.receipts[0] == completed_receipt
    assert recovered.records[1].planned_run_id == RUN_B
    assert recovered.records[1].outcome == "incomplete"
    assert recovered.receipts[1].charged_cost_usd == pending.reservation.reserved_cost_usd
    assert recovered.committed_cost_usd == (
        completed_receipt.charged_cost_usd + pending.reservation.reserved_cost_usd
    )


def test_finish_active_cell_refuses_to_clear_pending_paid_call() -> None:
    pending = _pending_call()
    checkpoint = create_live_checkpoint(
        bindings=_bindings(),
        active_cell=ActiveCell(
            planned_run_id=RUN_A,
            receipt_start_index=0,
            pending_call=pending,
        ),
    )
    record = RunRecord(
        planned_run_id=RUN_A,
        outcome="failed",
        error_category="protocol_error",
        cost_receipt_complete=True,
    )

    with pytest.raises(CheckpointValidationError, match="pending"):
        finish_active_cell(checkpoint, record=record)


def test_corrupt_or_tampered_checkpoint_fails_closed(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = create_live_checkpoint(bindings=_bindings())
    write_live_checkpoint(path, checkpoint)

    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="invalid"):
        load_live_checkpoint(path, expected_bindings=_bindings())

    write_live_checkpoint(path, checkpoint)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["committed_cost_usd"] = "0.01"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="integrity"):
        load_live_checkpoint(path, expected_bindings=_bindings())

    write_live_checkpoint(path, checkpoint)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="invalid"):
        load_live_checkpoint(path, expected_bindings=_bindings())
