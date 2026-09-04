"""The dated product price snapshot: shape, digest, lookup, staleness (DSE-1514)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from conclave.pricing import PRICE_SNAPSHOT_MAX_AGE_DAYS, PriceRates, PriceSnapshot


def _rates(model_id: str, *, source_url: str | None = "https://example.test/pricing") -> PriceRates:
    return PriceRates(
        provider_id=model_id.split("/", 1)[0],
        model_id=model_id,
        input_ceiling_usd_per_million_tokens=Decimal("3.00"),
        output_ceiling_usd_per_million_tokens=Decimal("15.00"),
        max_output_bytes_per_token=8,
        source_url=source_url,
    )


def _snapshot(**overrides) -> PriceSnapshot:
    payload = {
        "snapshot_id": "test-prices-2026-09-03",
        "captured_at": date(2026, 9, 3),
        "currency": "USD",
        "entries": (_rates("anthropic/claude-sonnet-4-6"), _rates("openai/gpt-4.1")),
    }
    payload.update(overrides)
    return PriceSnapshot(**payload)


def test_every_entry_must_cite_a_source_url():
    with pytest.raises(ValidationError, match="source_url"):
        _snapshot(entries=(_rates("openai/gpt-4.1", source_url=None),))


def test_duplicate_model_ids_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        _snapshot(entries=(_rates("openai/gpt-4.1"), _rates("openai/gpt-4.1")))


def test_rates_for_is_exact_and_never_fuzzy():
    snapshot = _snapshot()
    assert snapshot.rates_for("openai/gpt-4.1").model_id == "openai/gpt-4.1"
    assert snapshot.rates_for("openai/gpt-4.1-mini") is None
    assert snapshot.rates_for("anthropic/claude-opus-4") is None


def test_digest_is_order_independent_and_ignores_citations():
    forward = _snapshot()
    reversed_entries = _snapshot(entries=tuple(reversed(forward.entries)))
    recited = _snapshot(
        entries=tuple(
            entry.model_copy(update={"source_url": "https://elsewhere.test/prices"})
            for entry in forward.entries
        )
    )
    assert forward.digest() == reversed_entries.digest()
    assert forward.digest() == recited.digest()
    assert forward.digest().startswith("sha256:")


def test_digest_changes_when_any_rate_changes():
    forward = _snapshot()
    bumped = _snapshot(
        entries=(
            forward.entries[0].model_copy(
                update={"output_ceiling_usd_per_million_tokens": Decimal("15.000001")}
            ),
            forward.entries[1],
        )
    )
    assert forward.digest() != bumped.digest()


def test_staleness_is_reported_but_never_changes_a_rate():
    fresh = _snapshot(captured_at=date(2026, 9, 1))
    stale = _snapshot(captured_at=date(2026, 1, 1))
    assert PRICE_SNAPSHOT_MAX_AGE_DAYS == 90
    assert fresh.is_stale(as_of=date(2026, 9, 3)) is False
    assert stale.is_stale(as_of=date(2026, 9, 3)) is True
    entry = stale.rates_for("openai/gpt-4.1")
    assert entry.output_ceiling_usd_per_million_tokens == Decimal("15.00")
