from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from conclave.evals.models import (
    AnalysisGateConfig,
    BootstrapConfig,
    ExclusionDeviationPolicy,
    FrozenStudyDesign,
    PriceSnapshot,
    ProviderModelSpec,
    RandomizationConfig,
    RosterSpec,
    TimeoutRetryPolicy,
)
from conclave.evals.pricing import (
    CallReservation,
    PriceBook,
    hash_price_entries,
    load_price_book,
    reserve_call_cost,
    validate_price_book,
)
from conclave.evals.protocols import CONDITION_IDS

DIGEST = "sha256:" + "a" * 64
PRICE_FIXTURE = Path(__file__).parents[1] / "fixtures/evals/live_smoke/price_book.json"
EXPECTED_PRICE_HASH = "sha256:df85e17c12985ada8a350c6f2adc1778a0218bf3dc70cf3204ac8d1057939aac"


def _payload() -> dict[str, object]:
    return json.loads(PRICE_FIXTURE.read_text(encoding="utf-8"))


def _book(payload: dict[str, object] | None = None) -> PriceBook:
    return PriceBook.model_validate(payload or _payload())


def _design(*, prices_hash: str, snapshot_updates: dict[str, str] | None = None):
    snapshot = {
        "snapshot_id": "fictional-live-smoke-prices-2026-07-18",
        "captured_at": "2026-07-18T12:00:00Z",
        "currency": "USD",
        "prices_hash": prices_hash,
    }
    snapshot.update(snapshot_updates or {})
    rosters = (
        RosterSpec(
            roster_id="fictional-roster-a",
            members=(
                ProviderModelSpec(
                    provider_id="fictional-provider-a",
                    model_id="fictional-model-a",
                    model_revision="fixture-r1",
                ),
                ProviderModelSpec(
                    provider_id="fictional-provider-b",
                    model_id="fictional-model-b",
                    model_revision="fixture-r2",
                ),
            ),
        ),
        RosterSpec(
            roster_id="fictional-roster-b",
            members=(
                ProviderModelSpec(
                    provider_id="fictional-provider-c",
                    model_id="fictional-model-c",
                    model_revision="fixture-r3",
                ),
            ),
        ),
    )
    return FrozenStudyDesign(
        evidence_classification="paid_exploratory_pilot",
        base_commit="1" * 40,
        task_family_map={"fictional-task": "fixture-family"},
        rosters=rosters,
        condition_prompt_versions={condition: "prompt-v1" for condition in CONDITION_IDS},
        condition_protocol_versions={condition: "protocol-v1" for condition in CONDITION_IDS},
        generation_settings_hash=DIGEST,
        evaluator_version="evaluator-v1",
        analysis_code_hash=DIGEST,
        rubric_hash=DIGEST,
        grader_instructions_hash=DIGEST,
        grader_keys_hash=DIGEST,
        exclusion_deviation_policy=ExclusionDeviationPolicy(),
        timeout_retry_policy=TimeoutRetryPolicy(timeout_seconds=30, retry_attempts=0),
        randomization=RandomizationConfig(master_seed=20260718),
        bootstrap=BootstrapConfig(seed=20260718, samples=10),
        analysis_gates=AnalysisGateConfig(
            primary_baseline="single_frontier",
            absolute_p95_latency_seconds=60,
            minimum_confirmatory_tasks=2,
        ),
        price_snapshot=PriceSnapshot(**snapshot),
        approved_spend_ceiling_usd=10,
    )


def test_price_book_hash_is_canonical_and_binds_exact_frozen_snapshot() -> None:
    book = _book()
    price_hash = hash_price_entries(book.entries)

    assert price_hash == EXPECTED_PRICE_HASH
    assert hash_price_entries(reversed(book.entries)) == price_hash
    first_precise = book.entries[0].model_copy(
        update={"input_ceiling_usd_per_million_tokens": Decimal("1.12345678901234567890123456781")}
    )
    second_precise = first_precise.model_copy(
        update={"input_ceiling_usd_per_million_tokens": Decimal("1.12345678901234567890123456782")}
    )
    assert hash_price_entries((first_precise, *book.entries[1:])) != hash_price_entries(
        (second_precise, *book.entries[1:])
    )
    assert load_price_book(PRICE_FIXTURE, frozen_design=_design(prices_hash=price_hash)) == book
    with pytest.raises(ValidationError):
        book.snapshot_id = "mutated"

    for field, changed, error in (
        ("snapshot_id", "other-snapshot", "snapshot_id"),
        ("captured_at", "2026-07-18T12:00:01Z", "captured_at"),
        ("prices_hash", DIGEST, "prices_hash"),
    ):
        design = _design(prices_hash=price_hash, snapshot_updates={field: changed})
        with pytest.raises(ValueError, match=error):
            validate_price_book(book, frozen_design=design)


def test_price_book_rejects_duplicate_missing_unknown_or_revision_drift() -> None:
    payload = _payload()
    entries = payload["entries"]
    assert isinstance(entries, list)
    entries.append(dict(entries[0]))
    with pytest.raises(ValidationError, match="unique"):
        _book(payload)

    complete = _book()
    design = _design(prices_hash=hash_price_entries(complete.entries))

    missing = complete.model_copy(update={"entries": complete.entries[:-1]})
    with pytest.raises(ValueError, match="missing"):
        validate_price_book(missing, frozen_design=design)

    unknown_entry = complete.entries[0].model_copy(
        update={
            "provider_id": "fictional-provider-unknown",
            "model_id": "fictional-model-unknown",
        }
    )
    unknown = complete.model_copy(update={"entries": (*complete.entries, unknown_entry)})
    with pytest.raises(ValueError, match="unknown"):
        validate_price_book(unknown, frozen_design=design)

    drifted_entry = complete.entries[0].model_copy(update={"model_revision": "fixture-r99"})
    drifted = complete.model_copy(update={"entries": (drifted_entry, *complete.entries[1:])})
    with pytest.raises(ValueError, match="missing.*unknown"):
        validate_price_book(drifted, frozen_design=design)


def test_price_book_requires_usd_positive_pessimistic_rates() -> None:
    payload = _payload()
    payload["currency"] = "EUR"
    with pytest.raises(ValidationError, match="USD"):
        _book(payload)

    for field in (
        "input_ceiling_usd_per_million_tokens",
        "output_ceiling_usd_per_million_tokens",
    ):
        for invalid in ("0", "-0.000001"):
            payload = _payload()
            entries = payload["entries"]
            assert isinstance(entries, list)
            entries[0][field] = invalid
            with pytest.raises(ValidationError, match="greater than 0"):
                _book(payload)


def test_call_reservation_rounds_up_and_covers_input_output_and_framing() -> None:
    price = _book().entries[0]

    reservation = reserve_call_cost(
        price,
        prompt_token_upper_bound=101,
        prompt_template_token_allowance=17,
        provider_framing_token_allowance=11,
        upstream_output_token_ceilings=(50, 60),
        max_output_tokens=75,
    )

    assert isinstance(reservation, CallReservation)
    assert reservation.input_token_upper_bound == 239
    assert reservation.output_token_upper_bound == 75
    assert reservation.input_cost_upper_bound_usd == Decimal("0.000295061513")
    assert reservation.output_cost_upper_bound_usd == Decimal("0.000342591825")
    assert reservation.reserved_cost_usd == Decimal("0.000638")
    assert all(
        type(value) is Decimal
        for value in (
            reservation.input_ceiling_usd_per_million_tokens,
            reservation.output_ceiling_usd_per_million_tokens,
            reservation.input_cost_upper_bound_usd,
            reservation.output_cost_upper_bound_usd,
            reservation.reserved_cost_usd,
        )
    )
    with pytest.raises(ValidationError):
        reservation.reserved_cost_usd = Decimal("0")
