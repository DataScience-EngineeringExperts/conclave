"""Frozen external price snapshots and pessimistic call reservations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..pricing import TOKENS_PER_MILLION as TOKENS_PER_MILLION
from ..pricing import USD_MICROCENT as USD_MICROCENT
from ..pricing import PriceRates, reserve_cost
from .models import EvalModel, FrozenStudyDesign

_PRICE_HASH_NAMESPACE = "conclave_model_prices_v1"
_ModelIdentity = tuple[str, str, str]


class ModelPrice(EvalModel):
    """Pessimistic ceiling rates for one exact provider model revision."""

    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    input_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    output_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    max_output_bytes_per_token: int = Field(gt=0, strict=True)

    @field_validator(
        "input_ceiling_usd_per_million_tokens",
        "output_ceiling_usd_per_million_tokens",
        mode="before",
    )
    @classmethod
    def require_exact_decimal_rate(cls, value: object) -> object:
        if isinstance(value, (bool, float)):
            raise ValueError("price rates must be exact decimal values, not floats")
        return value

    @property
    def identity(self) -> _ModelIdentity:
        return (self.provider_id, self.model_id, self.model_revision)

    def as_rates(self) -> PriceRates:
        """Project this eval price entry onto the shared arithmetic type.

        The eval harness keys prices by ``(provider, model, revision)`` while the
        product keys them by the full provider-prefixed model id, so the
        projection joins provider and model with ``"/"``. The revision is not
        carried: it participates in eval identity and in
        :func:`hash_price_entries`, never in the arithmetic.
        """
        return PriceRates(
            provider_id=self.provider_id,
            model_id=f"{self.provider_id}/{self.model_id}",
            input_ceiling_usd_per_million_tokens=self.input_ceiling_usd_per_million_tokens,
            output_ceiling_usd_per_million_tokens=self.output_ceiling_usd_per_million_tokens,
            max_output_bytes_per_token=self.max_output_bytes_per_token,
        )


class PriceBook(EvalModel):
    """One immutable external snapshot covering a frozen study roster."""

    snapshot_id: str = Field(min_length=1)
    captured_at: str = Field(min_length=1)
    currency: Literal["USD"]
    entries: tuple[ModelPrice, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_entries(self) -> PriceBook:
        identities = [entry.identity for entry in self.entries]
        if len(set(identities)) != len(identities):
            raise ValueError("price book provider/model/revision identities must be unique")
        return self


class CallReservation(EvalModel):
    """Auditable worst-case token and cost bounds for one provider call."""

    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    prompt_token_upper_bound: int = Field(ge=0)
    prompt_template_token_allowance: int = Field(ge=0)
    provider_framing_token_allowance: int = Field(ge=0)
    upstream_output_token_ceilings: tuple[int, ...]
    upstream_output_bytes_per_token: int = Field(gt=0)
    input_token_upper_bound: int = Field(ge=0)
    output_token_upper_bound: int = Field(gt=0)
    input_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    output_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    input_cost_upper_bound_usd: Decimal = Field(ge=0)
    output_cost_upper_bound_usd: Decimal = Field(gt=0)
    reserved_cost_usd: Decimal = Field(gt=0)


def _canonical_decimal(value: Decimal) -> str:
    sign, digits, exponent = value.as_tuple()
    digit_text = "".join(str(digit) for digit in digits)
    if exponent >= 0:
        integer = digit_text + ("0" * exponent)
        fraction = ""
    else:
        split_at = len(digit_text) + exponent
        if split_at > 0:
            integer = digit_text[:split_at]
            fraction = digit_text[split_at:]
        else:
            integer = "0"
            fraction = ("0" * -split_at) + digit_text
        fraction = fraction.rstrip("0")
    canonical = integer if not fraction else f"{integer}.{fraction}"
    return f"-{canonical}" if sign else canonical


def _canonical_entry(entry: ModelPrice) -> dict[str, str]:
    return {
        "schema_version": entry.schema_version,
        "provider_id": entry.provider_id,
        "model_id": entry.model_id,
        "model_revision": entry.model_revision,
        "input_ceiling_usd_per_million_tokens": _canonical_decimal(
            entry.input_ceiling_usd_per_million_tokens
        ),
        "output_ceiling_usd_per_million_tokens": _canonical_decimal(
            entry.output_ceiling_usd_per_million_tokens
        ),
        "max_output_bytes_per_token": str(entry.max_output_bytes_per_token),
    }


def hash_price_entries(entries: Iterable[ModelPrice]) -> str:
    """Return an order- and representation-independent digest of model rates."""

    ordered = sorted(entries, key=lambda entry: entry.identity)
    canonical = json.dumps(
        {
            "namespace": _PRICE_HASH_NAMESPACE,
            "entries": [_canonical_entry(entry) for entry in ordered],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _format_identities(identities: set[_ModelIdentity]) -> str:
    return ", ".join("/".join(identity) for identity in sorted(identities))


def validate_price_book(price_book: PriceBook, *, frozen_design: FrozenStudyDesign) -> None:
    """Validate exact snapshot metadata, roster coverage, and canonical entry hash."""

    snapshot = frozen_design.price_snapshot
    if snapshot.snapshot_id != price_book.snapshot_id:
        raise ValueError("price book snapshot_id does not match frozen design")
    if snapshot.captured_at != price_book.captured_at:
        raise ValueError("price book captured_at does not match frozen design")
    if snapshot.currency != "USD" or snapshot.currency != price_book.currency:
        raise ValueError("price book currency must be USD and match frozen design")

    expected = {
        (member.provider_id, member.model_id, member.model_revision)
        for roster in frozen_design.rosters
        for member in roster.members
    }
    actual = {entry.identity for entry in price_book.entries}
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing={_format_identities(missing)}")
        if unknown:
            parts.append(f"unknown={_format_identities(unknown)}")
        raise ValueError(f"price book roster coverage mismatch: {'; '.join(parts)}")

    if hash_price_entries(price_book.entries) != snapshot.prices_hash:
        raise ValueError("price book prices_hash does not match frozen design")


def load_price_book(path: str | Path, *, frozen_design: FrozenStudyDesign) -> PriceBook:
    """Load and bind an external JSON price book to one frozen study design."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_float=Decimal)
    price_book = PriceBook.model_validate(payload)
    validate_price_book(price_book, frozen_design=frozen_design)
    return price_book


def reserve_call_cost(
    price: ModelPrice,
    *,
    prompt_token_upper_bound: int,
    prompt_template_token_allowance: int,
    provider_framing_token_allowance: int,
    upstream_output_token_ceilings: Sequence[int],
    upstream_output_bytes_per_token: int,
    max_output_tokens: int,
) -> CallReservation:
    """Reserve the pessimistic USD cost of all possible input and output tokens.

    Thin adapter over :func:`conclave.pricing.reserve_cost` (DSE-1514): the
    arithmetic, the validation, and the ROUND_CEILING behaviour are shared with
    the product run path, while the eval-only ``CallReservation`` contract --
    including ``model_revision`` and the frozen eval ``schema_version`` that
    :func:`hash_price_entries` depends on -- is assembled here unchanged.

    ``upstream_output_token_ceilings`` is materialized into a tuple ONCE,
    before either use below (QA M3): a one-shot iterable (e.g. a generator
    expression) passed by a caller would otherwise be fully consumed by
    :func:`conclave.pricing.reserve_cost`'s own internal ``tuple(...)`` call,
    leaving nothing for the second ``tuple(...)`` that used to build
    ``CallReservation.upstream_output_token_ceilings`` here -- silently
    recording an empty tuple on the reservation even though the real values
    were correctly priced. Passing the SAME materialized tuple to both keeps
    the recorded reservation and the priced amounts provably consistent
    regardless of what kind of iterable the caller passed in.
    """
    upstream_ceilings = tuple(upstream_output_token_ceilings)
    amounts = reserve_cost(
        price.as_rates(),
        prompt_token_upper_bound=prompt_token_upper_bound,
        prompt_template_token_allowance=prompt_template_token_allowance,
        provider_framing_token_allowance=provider_framing_token_allowance,
        upstream_output_token_ceilings=upstream_ceilings,
        upstream_output_bytes_per_token=upstream_output_bytes_per_token,
        max_output_tokens=max_output_tokens,
    )
    return CallReservation(
        provider_id=price.provider_id,
        model_id=price.model_id,
        model_revision=price.model_revision,
        prompt_token_upper_bound=prompt_token_upper_bound,
        prompt_template_token_allowance=prompt_template_token_allowance,
        provider_framing_token_allowance=provider_framing_token_allowance,
        upstream_output_token_ceilings=upstream_ceilings,
        upstream_output_bytes_per_token=upstream_output_bytes_per_token,
        input_token_upper_bound=amounts.input_token_upper_bound,
        output_token_upper_bound=amounts.output_token_upper_bound,
        input_ceiling_usd_per_million_tokens=price.input_ceiling_usd_per_million_tokens,
        output_ceiling_usd_per_million_tokens=price.output_ceiling_usd_per_million_tokens,
        input_cost_upper_bound_usd=amounts.input_cost_upper_bound_usd,
        output_cost_upper_bound_usd=amounts.output_cost_upper_bound_usd,
        reserved_cost_usd=amounts.reserved_cost_usd,
    )
