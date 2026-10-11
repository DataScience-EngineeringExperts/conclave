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

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
