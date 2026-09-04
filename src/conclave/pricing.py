"""Shared, float-free price arithmetic for ceilings and reservations (DSE-1514).

This module owns the ONE implementation of conclave's cost arithmetic. Both the
paid eval harness (:mod:`conclave.evals.pricing`, :mod:`conclave.evals.live`)
and the product run path (:mod:`conclave.council`) delegate to it, so a ceiling
computed for an audit receipt and a reservation computed for a paid eval cell
can never drift apart.

Three invariants hold everywhere in here and must survive every future edit:

* **No floats.** Rates are exact :class:`decimal.Decimal`; a ``float`` (or
  ``bool``) rate is rejected at validation time, not coerced.
* **Always up.** Every emitted amount is quantized to :data:`USD_MICROCENT`
  with :data:`decimal.ROUND_CEILING`. A number produced here is an upper bound,
  never a best guess, so it is safe to put inside an audit receipt.
* **Bytes bound tokens.** An input bound is computed from UTF-8 *bytes* with
  each byte treated as at most one token, and an upstream model's output is
  bounded by its output cap times ``max_output_bytes_per_token``. Both
  directions are deliberately pessimistic; this is the attestation the paid
  eval runner has shipped since the H1 live runner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .logging import get_logger

logger = get_logger("pricing")

# The quantum every emitted amount is rounded up to: one USD micro-cent.
USD_MICROCENT = Decimal("0.000001")
TOKENS_PER_MILLION = Decimal(1_000_000)


class PriceRates(BaseModel):
    """Pessimistic ceiling rates for one model, as exact decimals.

    This is the arithmetic input type shared by the eval harness and the product
    run path. ``model_id`` is the FULL provider-prefixed id the product uses
    (``"anthropic/claude-sonnet-4-6"``); the eval harness passes its own
    ``model_id`` through :meth:`conclave.evals.pricing.ModelPrice.as_rates`.

    Attributes:
        provider_id: Provider prefix (``"anthropic"``).
        model_id: Full model identifier. For the product snapshot this is the
            provider-prefixed id resolved by ``ConclaveConfig.resolve_model_id``.
        input_ceiling_usd_per_million_tokens: Exact upper-bound input rate.
        output_ceiling_usd_per_million_tokens: Exact upper-bound output rate.
        max_output_bytes_per_token: Attested upper bound on the UTF-8 byte
            length of one output token, used to bound a DOWNSTREAM call whose
            input embeds this model's output.
        source_url: Where the rate was read from. Required by the product
            snapshot loader; ``None`` for eval-derived rates, which are bound to
            a frozen study design instead. Deliberately NOT part of
            :meth:`PriceSnapshot.digest` -- correcting a citation must not
            invalidate every cache entry priced under identical rates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    input_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    output_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    max_output_bytes_per_token: int = Field(gt=0, strict=True)
    source_url: str | None = None

    @field_validator(
        "input_ceiling_usd_per_million_tokens",
        "output_ceiling_usd_per_million_tokens",
        mode="before",
    )
    @classmethod
    def require_exact_decimal_rate(cls, value: object) -> object:
        """Reject float/bool rates outright rather than coercing them.

        ``Decimal(0.4)`` is ``0.4000000000000000222``; a rate that has been
        through a float is no longer the number the vendor published, and a
        ceiling built on it is not falsifiable. Mirrors
        ``conclave.evals.pricing.ModelPrice`` byte for byte.
        """
        if isinstance(value, (bool, float)):
            raise ValueError("price rates must be exact decimal values, not floats")
        return value


@dataclass(frozen=True)
class ReservationAmounts:
    """The pessimistic token and USD bounds for one planned provider call."""

    input_token_upper_bound: int
    output_token_upper_bound: int
    input_cost_upper_bound_usd: Decimal
    output_cost_upper_bound_usd: Decimal
    reserved_cost_usd: Decimal


def validate_token_bound(name: str, value: int, *, positive: bool = False) -> None:
    """Reject a non-integer, boolean, or out-of-range token bound.

    ``bool`` is an ``int`` subclass in Python, so ``isinstance(True, int)`` is
    ``True``; a silently-accepted ``True`` would become the token count ``1``.

    Args:
        name: Parameter name, used verbatim in the raised message.
        value: The candidate bound.
        positive: When ``True`` the bound must be at least ``1`` rather than ``0``.

    Raises:
        TypeError: ``value`` is not an ``int`` (or is a ``bool``).
        ValueError: ``value`` is below the permitted minimum.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        comparator = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {comparator}")


def _precision_for(*pairs: tuple[int, Decimal]) -> int:
    """Return a decimal context precision that cannot lose a significant digit."""
    return max(64, *(len(str(count)) + len(rate.as_tuple().digits) + 20 for count, rate in pairs))


def reserve_cost(
    rates: PriceRates,
    *,
    prompt_token_upper_bound: int,
    prompt_template_token_allowance: int,
    provider_framing_token_allowance: int,
    upstream_output_token_ceilings: Sequence[int],
    upstream_output_bytes_per_token: int,
    max_output_tokens: int,
) -> ReservationAmounts:
    """Reserve the pessimistic USD cost of every token one call could consume.

    The input bound is the sum of four independently attested terms:

    #. ``prompt_token_upper_bound`` -- UTF-8 bytes of the exact known message
       content (plus any structured-output contract JSON), each byte counted as
       one token;
    #. ``prompt_template_token_allowance`` -- UTF-8 bytes of the fixed system
       and user template wording that will surround it;
    #. ``provider_framing_token_allowance`` -- per-request and per-message
       provider overhead;
    #. ``sum(upstream_output_token_ceilings) * upstream_output_bytes_per_token``
       -- the not-yet-produced output of every upstream call this call's input
       will embed, converted from its token cap to a byte bound.

    Args:
        rates: The model's exact ceiling rates.
        prompt_token_upper_bound: Bound on the known prompt content.
        prompt_template_token_allowance: Bound on fixed template wording.
        provider_framing_token_allowance: Bound on provider request framing.
        upstream_output_token_ceilings: One output cap per upstream call whose
            output is embedded in this call's input.
        upstream_output_bytes_per_token: Attested bytes-per-output-token used to
            convert those caps into an input bound.
        max_output_tokens: The hard output cap this call will be issued with.

    Returns:
        A :class:`ReservationAmounts` whose ``reserved_cost_usd`` is a true
        upper bound, quantized up to :data:`USD_MICROCENT`.

    Raises:
        TypeError: Any bound is not an integer.
        ValueError: Any bound is out of range.
    """
    validate_token_bound("prompt_token_upper_bound", prompt_token_upper_bound)
    validate_token_bound("prompt_template_token_allowance", prompt_template_token_allowance)
    validate_token_bound("provider_framing_token_allowance", provider_framing_token_allowance)
    validate_token_bound(
        "upstream_output_bytes_per_token",
        upstream_output_bytes_per_token,
        positive=True,
    )
    validate_token_bound("max_output_tokens", max_output_tokens, positive=True)
    upstream = tuple(upstream_output_token_ceilings)
    for index, ceiling in enumerate(upstream):
        validate_token_bound(f"upstream_output_token_ceilings[{index}]", ceiling)

    input_token_upper_bound = (
        prompt_token_upper_bound
        + prompt_template_token_allowance
        + provider_framing_token_allowance
        + (sum(upstream) * upstream_output_bytes_per_token)
    )
    precision = _precision_for(
        (input_token_upper_bound, rates.input_ceiling_usd_per_million_tokens),
        (max_output_tokens, rates.output_ceiling_usd_per_million_tokens),
    )
    with localcontext() as context:
        context.prec = precision
        input_cost = (
            Decimal(input_token_upper_bound)
            * rates.input_ceiling_usd_per_million_tokens
            / TOKENS_PER_MILLION
        )
        output_cost = (
            Decimal(max_output_tokens)
            * rates.output_ceiling_usd_per_million_tokens
            / TOKENS_PER_MILLION
        )
        reserved = (input_cost + output_cost).quantize(USD_MICROCENT, rounding=ROUND_CEILING)

    return ReservationAmounts(
        input_token_upper_bound=input_token_upper_bound,
        output_token_upper_bound=max_output_tokens,
        input_cost_upper_bound_usd=input_cost,
        output_cost_upper_bound_usd=output_cost,
        reserved_cost_usd=reserved,
    )


def reported_usage_cost(
    rates: PriceRates,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> Decimal:
    """Price a completed call from the usage the provider actually reported.

    Prompt tokens are charged at the input ceiling, completion tokens at the
    output ceiling, and any *unattributed* remainder
    (``total - prompt - completion``, e.g. provider-side reasoning or cache
    tokens) at the HIGHER of the two rates, because nothing in the reported
    usage says which side it belongs to.

    Args:
        rates: The model's exact ceiling rates.
        prompt_tokens: Reported prompt tokens.
        completion_tokens: Reported completion tokens.
        total_tokens: Reported total tokens.

    Returns:
        The cost ceiling in USD, quantized up to :data:`USD_MICROCENT`.

    Raises:
        ValueError: ``total_tokens`` is below the attributed sum, which would
            make the remainder negative and the result an under-bound.
    """
    attributed = prompt_tokens + completion_tokens
    if total_tokens < attributed:
        raise ValueError("provider total_tokens is smaller than its attributed usage")
    unattributed = total_tokens - attributed
    higher = max(
        rates.input_ceiling_usd_per_million_tokens,
        rates.output_ceiling_usd_per_million_tokens,
    )
    precision = _precision_for(
        (prompt_tokens, rates.input_ceiling_usd_per_million_tokens),
        (completion_tokens, rates.output_ceiling_usd_per_million_tokens),
        (total_tokens, higher),
    )
    with localcontext() as context:
        context.prec = precision
        cost = (
            Decimal(prompt_tokens) * rates.input_ceiling_usd_per_million_tokens
            + Decimal(completion_tokens) * rates.output_ceiling_usd_per_million_tokens
            + Decimal(unattributed) * higher
        ) / TOKENS_PER_MILLION
        return cost.quantize(USD_MICROCENT, rounding=ROUND_CEILING)


# A snapshot older than this many days emits a bounded warning on the manifest.
# It is a STALENESS SIGNAL ONLY: an old snapshot still prices at exactly the
# rates it records. Substituting a rate because one looks old would replace a
# falsifiable claim with a guess, which is the failure this module exists to
# prevent.
PRICE_SNAPSHOT_MAX_AGE_DAYS = 90

# Digest namespace for the PRODUCT snapshot. Deliberately distinct from the eval
# harness's ``conclave_model_prices_v1`` so the two hash spaces can never collide
# and an eval price book can never be mistaken for a product snapshot.
_PRODUCT_PRICE_HASH_NAMESPACE = "conclave_product_prices_v1"


def _canonical_decimal(value: Decimal) -> str:
    """Render a Decimal in a normalized, representation-independent form.

    ``Decimal("3.00")`` and ``Decimal("3")`` are the same rate and must produce
    the same digest, so trailing fractional zeros are stripped and exponent
    notation is expanded.
    """
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


class PriceSnapshot(BaseModel):
    """One dated, frozen, hand-verified set of product price ceilings.

    Snapshots are checked-in artifacts under ``src/conclave/data/``, never
    fetched at runtime (explicitly out of scope). Every entry cites the vendor
    page its rate was read from; a model whose published list price could not be
    verified is OMITTED, which makes it *unpriced* rather than guessed.

    Attributes:
        snapshot_id: Stable identifier, conventionally
            ``conclave-default-prices-<YYYY-MM-DD>``.
        captured_at: The date the rates were read from the vendor pages.
        currency: Always ``"USD"``.
        entries: The priced models, keyed by full provider-prefixed model id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str = Field(min_length=1)
    captured_at: date
    currency: Literal["USD"]
    entries: tuple[PriceRates, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> PriceSnapshot:
        """Require unique model ids and a citation on every entry."""
        model_ids = [entry.model_id for entry in self.entries]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("price snapshot model ids must be unique")
        uncited = sorted(entry.model_id for entry in self.entries if not entry.source_url)
        if uncited:
            raise ValueError(f"price snapshot entries must carry a source_url: {uncited}")
        return self

    def rates_for(self, model_id: str) -> PriceRates | None:
        """Return the EXACT entry for ``model_id``, or ``None``.

        Matching is exact and total: there is no prefix match, no provider
        fallback, and no nearest-neighbour rate. An absent model is unpriced,
        full stop.
        """
        for entry in self.entries:
            if entry.model_id == model_id:
                return entry
        return None

    def is_stale(self, *, as_of: date | None = None) -> bool:
        """Return whether this snapshot is older than the staleness threshold."""
        reference = as_of or date.today()
        return (reference - self.captured_at).days > PRICE_SNAPSHOT_MAX_AGE_DAYS

    def digest(self) -> str:
        """Return an order- and representation-independent digest of the RATES.

        Covers the namespace plus, per entry, ``provider_id``, ``model_id``, both
        ceiling rates in canonical decimal form, and
        ``max_output_bytes_per_token``. It deliberately EXCLUDES ``snapshot_id``,
        ``captured_at``, and ``source_url``: this digest joins cache identity, and
        re-dating or re-citing a snapshot whose rates are byte-identical must not
        invalidate entries whose ceiling would come out exactly the same.
        """
        ordered = sorted(self.entries, key=lambda entry: (entry.provider_id, entry.model_id))
        canonical = json.dumps(
            {
                "namespace": _PRODUCT_PRICE_HASH_NAMESPACE,
                "entries": [
                    {
                        "provider_id": entry.provider_id,
                        "model_id": entry.model_id,
                        "input_ceiling_usd_per_million_tokens": _canonical_decimal(
                            entry.input_ceiling_usd_per_million_tokens
                        ),
                        "output_ceiling_usd_per_million_tokens": _canonical_decimal(
                            entry.output_ceiling_usd_per_million_tokens
                        ),
                        "max_output_bytes_per_token": str(entry.max_output_bytes_per_token),
                    }
                    for entry in ordered
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _price_data_dir() -> Path:
    """Return the packaged directory holding the dated price snapshots."""
    return Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_default_price_snapshot() -> PriceSnapshot | None:
    """Load the newest packaged price snapshot, or ``None`` when unavailable.

    Picks the lexicographically-last ``prices-YYYY-MM-DD.json`` in
    ``conclave/data`` (ISO dates sort chronologically), parses it with
    ``parse_float=Decimal`` so a JSON number can never become a float rate, and
    validates it. Memoized for the life of the process: a snapshot is a frozen
    artifact, so re-reading it per run would be pure overhead.

    Returns ``None`` -- never raises -- when the directory is absent, empty,
    unreadable, or holds a file that fails validation, and logs a warning. A
    missing snapshot means a run carries NO ceiling; it must never mean a failed
    run. (The pre-flight spend gate treats the same condition as a refusal
    instead; that asymmetry is deliberate.)
    """
    try:
        candidates = sorted(_price_data_dir().glob("prices-*.json"))
    except OSError as exc:
        logger.warning("price snapshot directory unreadable: %s; pricing disabled", exc)
        return None
    if not candidates:
        logger.warning("no packaged price snapshot found; pricing disabled")
        return None
    path = candidates[-1]
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, parse_float=Decimal)
        payload.pop("_note", None)
        return PriceSnapshot.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.warning("price snapshot %s is unusable: %s; pricing disabled", path.name, exc)
        return None


class SpendRefused(Exception):
    """Base class for a pre-flight spend refusal.

    Raised BEFORE any provider call. Every subclass maps to CLI exit code 4.
    Refusing is the honest outcome when a run cannot be bounded: the alternative
    is to invent a number, which is the exact failure this module exists to
    prevent.
    """


class SpendUnboundable(SpendRefused):
    """The call plan cannot be priced at all, so no cap can be enforced.

    Causes: no output cap configured, no price snapshot available, or a model in
    the plan with no snapshot entry. Never a fallback rate, never a guess.
    """


class SpendCapExceeded(SpendRefused):
    """The fully-priced worst-case plan reserves more than the stated cap.

    Attributes:
        reserved: The pessimistic total for the whole plan, in USD.
        cap: The operator's stated cap, in USD.
        call_count: How many provider calls the plan enumerated.
    """

    def __init__(self, reserved: Decimal, cap: Decimal, call_count: int) -> None:
        self.reserved = reserved
        self.cap = cap
        self.call_count = call_count
        super().__init__(
            f"refusing to run: reserved {reserved} USD for {call_count} calls "
            f"exceeds the cap of {cap} USD"
        )
