"""Frozen external price snapshots and pessimistic call reservations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import EvalModel, FrozenStudyDesign

USD_MICROCENT = Decimal("0.000001")
TOKENS_PER_MILLION = Decimal(1_000_000)
_PRICE_HASH_NAMESPACE = "conclave_model_prices_v1"
_ModelIdentity = tuple[str, str, str]


class ModelPrice(EvalModel):
    """Pessimistic ceiling rates for one exact provider model revision."""

    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    input_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    output_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)

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


def _validate_token_bound(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        comparator = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {comparator}")


def reserve_call_cost(
    price: ModelPrice,
    *,
    prompt_token_upper_bound: int,
    prompt_template_token_allowance: int,
    provider_framing_token_allowance: int,
    upstream_output_token_ceilings: Sequence[int],
    max_output_tokens: int,
) -> CallReservation:
    """Reserve the pessimistic USD cost of all possible input and output tokens."""

    _validate_token_bound("prompt_token_upper_bound", prompt_token_upper_bound)
    _validate_token_bound("prompt_template_token_allowance", prompt_template_token_allowance)
    _validate_token_bound("provider_framing_token_allowance", provider_framing_token_allowance)
    _validate_token_bound("max_output_tokens", max_output_tokens, positive=True)
    upstream_ceilings = tuple(upstream_output_token_ceilings)
    for index, ceiling in enumerate(upstream_ceilings):
        _validate_token_bound(f"upstream_output_token_ceilings[{index}]", ceiling)

    input_token_upper_bound = (
        prompt_token_upper_bound
        + prompt_template_token_allowance
        + provider_framing_token_allowance
        + sum(upstream_ceilings)
    )
    precision = max(
        64,
        len(str(input_token_upper_bound))
        + len(price.input_ceiling_usd_per_million_tokens.as_tuple().digits)
        + 20,
        len(str(max_output_tokens))
        + len(price.output_ceiling_usd_per_million_tokens.as_tuple().digits)
        + 20,
    )
    with localcontext() as context:
        context.prec = precision
        input_cost = (
            Decimal(input_token_upper_bound)
            * price.input_ceiling_usd_per_million_tokens
            / TOKENS_PER_MILLION
        )
        output_cost = (
            Decimal(max_output_tokens)
            * price.output_ceiling_usd_per_million_tokens
            / TOKENS_PER_MILLION
        )
        reserved_cost = (input_cost + output_cost).quantize(USD_MICROCENT, rounding=ROUND_CEILING)

    return CallReservation(
        provider_id=price.provider_id,
        model_id=price.model_id,
        model_revision=price.model_revision,
        prompt_token_upper_bound=prompt_token_upper_bound,
        prompt_template_token_allowance=prompt_template_token_allowance,
        provider_framing_token_allowance=provider_framing_token_allowance,
        upstream_output_token_ceilings=upstream_ceilings,
        input_token_upper_bound=input_token_upper_bound,
        output_token_upper_bound=max_output_tokens,
        input_ceiling_usd_per_million_tokens=price.input_ceiling_usd_per_million_tokens,
        output_ceiling_usd_per_million_tokens=price.output_ceiling_usd_per_million_tokens,
        input_cost_upper_bound_usd=input_cost,
        output_cost_upper_bound_usd=output_cost,
        reserved_cost_usd=reserved_cost,
    )
