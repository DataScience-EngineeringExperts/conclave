"""Ceiling fields on the manifest: additive, Decimal-only, all-or-nothing (DSE-1514)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from conclave.council import Council
from conclave.manifest import (
    ModelHarnessManifest,
    ProviderExecutionReceipt,
    scan_for_secret_material,
)
from conclave.pricing import PriceRates, PriceSnapshot
from tests.conftest import make_response


def _receipt(**overrides) -> ProviderExecutionReceipt:
    payload = {
        "name": "claude",
        "provider": "anthropic",
        "model_id": "anthropic/claude-sonnet-4-6",
    }
    payload.update(overrides)
    return ProviderExecutionReceipt(**payload)


def test_new_receipt_fields_default_to_none_and_estimated_cost_is_untouched():
    receipt = _receipt()
    assert receipt.cost_ceiling_usd is None
    assert receipt.cost_basis is None
    assert receipt.estimated_cost is None


def test_receipt_ceiling_is_decimal_and_rejects_a_float():
    receipt = _receipt(cost_ceiling_usd=Decimal("0.001234"), cost_basis="reported_usage")
    assert isinstance(receipt.cost_ceiling_usd, Decimal)
    assert receipt.cost_ceiling_usd == Decimal("0.001234")
    with pytest.raises(ValidationError):
        _receipt(cost_ceiling_usd=0.001234)


def test_receipt_cost_basis_is_a_closed_vocabulary():
    assert _receipt(cost_basis="reservation").cost_basis == "reservation"
    with pytest.raises(ValidationError):
        _receipt(cost_basis="guess")


def test_manifest_ceiling_fields_default_empty_and_estimated_cost_stays_none():
    manifest = ModelHarnessManifest(request_id="r", conclave_version="1.3.0", mode="synthesize")
    assert manifest.cost_ceiling_usd is None
    assert manifest.price_snapshot_digest is None
    assert manifest.priced_as_of is None
    assert manifest.unpriced_models == []
    assert manifest.unpriced_receipts == 0
    assert manifest.pricing_warnings == []
    assert manifest.estimated_cost is None
    assert manifest.pricing_snapshot_date is None


def test_populated_pricing_fields_keep_the_secret_scan_clean():
    manifest = ModelHarnessManifest(
        request_id="r",
        conclave_version="1.3.0",
        mode="elite",
        model_ids=["anthropic/claude-sonnet-4-6", "deepseek/deepseek-chat"],
        receipts=[_receipt(cost_ceiling_usd=Decimal("0.5"), cost_basis="reported_usage")],
        cost_ceiling_usd=Decimal("0.5"),
        price_snapshot_digest="sha256:" + "b" * 64,
        priced_as_of="2026-09-03",
        unpriced_models=["deepseek/deepseek-chat"],
        unpriced_receipts=0,
        pricing_warnings=["unpriced_models_present", "price_snapshot_stale"],
    )
    assert scan_for_secret_material(manifest) is True


def test_ceilings_round_trip_through_json_as_exact_decimals():
    manifest = ModelHarnessManifest(
        request_id="r",
        conclave_version="1.3.0",
        mode="synthesize",
        receipts=[_receipt(cost_ceiling_usd=Decimal("0.000001"), cost_basis="reservation")],
        cost_ceiling_usd=Decimal("0.000001"),
    )
    restored = ModelHarnessManifest.model_validate(manifest.model_dump(mode="json"))
    assert restored.cost_ceiling_usd == Decimal("0.000001")
    assert restored.receipts[0].cost_ceiling_usd == Decimal("0.000001")
    assert isinstance(restored.cost_ceiling_usd, Decimal)


def _snapshot(*model_ids: str, captured_at: date = date(2026, 9, 3)) -> PriceSnapshot:
    return PriceSnapshot(
        snapshot_id="test-prices",
        captured_at=captured_at,
        currency="USD",
        entries=tuple(
            PriceRates(
                provider_id=model_id.split("/", 1)[0],
                model_id=model_id,
                input_ceiling_usd_per_million_tokens=Decimal("3.00"),
                output_ceiling_usd_per_million_tokens=Decimal("15.00"),
                max_output_bytes_per_token=8,
                source_url="https://example.test/pricing",
            )
            for model_id in model_ids
        ),
    )


def _install_snapshot(monkeypatch, snapshot):
    import conclave.council as council_mod

    monkeypatch.setattr(council_mod, "load_default_price_snapshot", lambda: snapshot)


async def test_a_fully_priced_run_carries_a_ceiling_digest_and_date(
    monkeypatch, patch_call_model, keys
):
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", "anthropic/claude-sonnet-4-6"))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="claude", extract_verdict=False)
    result = await council.ask("q")

    manifest = result.manifest
    assert manifest.unpriced_models == []
    assert manifest.unpriced_receipts == 0
    assert isinstance(manifest.cost_ceiling_usd, Decimal)
    assert manifest.cost_ceiling_usd == sum(
        (receipt.cost_ceiling_usd for receipt in manifest.receipts), Decimal("0")
    )
    assert manifest.price_snapshot_digest.startswith("sha256:")
    assert manifest.priced_as_of == "2026-09-03"
    assert all(receipt.cost_basis == "reported_usage" for receipt in manifest.receipts)
    # The estimate slot is untouched, forever.
    assert manifest.estimated_cost is None
    assert all(receipt.estimated_cost is None for receipt in manifest.receipts)


async def test_one_unpriced_model_nulls_the_whole_run_ceiling(monkeypatch, patch_call_model, keys):
    # grok is priced, claude is NOT -> all-or-nothing.
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="claude", extract_verdict=False)
    result = await council.ask("q")

    manifest = result.manifest
    assert manifest.cost_ceiling_usd is None
    assert manifest.unpriced_models == ["anthropic/claude-sonnet-4-6"]
    assert manifest.unpriced_receipts == 1
    assert "unpriced_models_present" in manifest.pricing_warnings
    # The PRICED receipts still carry their own ceilings -- only the SUM is withheld.
    priced = [r for r in manifest.receipts if r.model_id == "xai/grok-4.3"]
    assert priced and all(r.cost_ceiling_usd is not None for r in priced)


async def test_a_stale_snapshot_warns_but_never_changes_a_rate(monkeypatch, patch_call_model, keys):
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", captured_at=date(2026, 9, 3)))
    fresh = (await council.ask("q", synthesize=False)).manifest

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", captured_at=date(2026, 1, 1)))
    stale = (await council.ask("q", synthesize=False)).manifest

    assert fresh.pricing_warnings == []
    assert "price_snapshot_stale" in stale.pricing_warnings
    assert stale.cost_ceiling_usd == fresh.cost_ceiling_usd


async def test_a_missing_snapshot_leaves_no_ceiling_and_never_raises(
    monkeypatch, patch_call_model, keys
):
    _install_snapshot(monkeypatch, None)
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)
    manifest = (await council.ask("q", synthesize=False)).manifest

    assert manifest.cost_ceiling_usd is None
    assert manifest.price_snapshot_digest is None
    assert manifest.pricing_warnings == ["price_snapshot_unavailable"]
    assert manifest.secret_safety == "verified_no_secrets"


async def test_a_failed_call_with_no_usage_is_unpriced_without_an_output_cap(
    monkeypatch, patch_call_model, keys
):
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    def handler(model_id, messages):
        raise RuntimeError("boom")

    patch_call_model(handler)
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)
    manifest = (await council.ask("q", synthesize=False)).manifest

    assert manifest.receipts[0].cost_ceiling_usd is None
    assert manifest.receipts[0].cost_basis is None
    assert manifest.unpriced_receipts == 1
    assert manifest.cost_ceiling_usd is None
    assert "unpriced_receipts_present" in manifest.pricing_warnings
    assert "no_output_cap_configured" in manifest.pricing_warnings


async def test_an_impossible_usage_total_degrades_to_unpriced(monkeypatch, keys):
    import conclave.council as council_mod
    from conclave.models import ModelAnswer, TokenUsage

    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def bad_usage(name, model_id, messages, **kwargs):
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="ok",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=5),
        )

    monkeypatch.setattr(council_mod, "call_model", bad_usage)
    council = Council(models=["grok"], synthesizer="grok", extract_verdict=False)
    manifest = (await council.ask("q", synthesize=False)).manifest

    assert manifest.receipts[0].cost_ceiling_usd is None
    assert manifest.cost_ceiling_usd is None
    assert manifest.unpriced_receipts == 1
