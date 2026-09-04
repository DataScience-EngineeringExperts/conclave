"""Ceiling fields on the manifest: additive, Decimal-only, all-or-nothing (DSE-1514)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from conclave.manifest import (
    ModelHarnessManifest,
    ProviderExecutionReceipt,
    scan_for_secret_material,
)


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
