"""Sequential paid-provider gateway with pessimistic budget reservations."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import Literal

from pydantic import Field

from conclave.models import ModelAnswer, TokenUsage
from conclave.providers import _receipt_error_category, call_model

from .live_protocols import StageCall
from .models import EvalModel
from .pricing import (
    TOKENS_PER_MILLION,
    USD_MICROCENT,
    CallReservation,
    ModelPrice,
    PriceBook,
    reserve_call_cost,
)

ProviderCallOutcome = Literal["success", "failed"]
ProviderCallErrorCategory = Literal[
    "authentication",
    "rate_limit",
    "timeout",
    "transport",
    "provider_error",
    "reservation_breach",
]
CostBasisSource = Literal["reported_usage", "full_reservation"]
_BOUNDED_PROVIDER_ERROR_CATEGORIES = frozenset(
    {"authentication", "rate_limit", "timeout", "transport", "provider_error"}
)


class LiveGatewayError(RuntimeError):
    """Base class for fail-closed live gateway errors."""


class BudgetExceededError(LiveGatewayError):
    """The next pessimistic reservation would cross the approved hard cap."""


class ReservationBreachError(LiveGatewayError):
    """A provider reported usage outside the pre-call reservation."""


class GatewayStoppedError(LiveGatewayError):
    """The gateway stopped after an accounting invariant was breached."""


class ProviderCallCostBasis(EvalModel):
    """Exact inputs used to reconcile one provider charge."""

    source: CostBasisSource
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    input_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)
    output_ceiling_usd_per_million_tokens: Decimal = Field(gt=0)


class PendingCall(EvalModel):
    """Secret-free reservation persisted before any provider call begins."""

    stage: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    reservation: CallReservation


class ProviderCallReceipt(EvalModel):
    """Bounded provider outcome and exact USD reconciliation facts."""

    stage: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    outcome: ProviderCallOutcome
    error_category: ProviderCallErrorCategory | None = None
    usage: TokenUsage | None = None
    reserved_cost_usd: Decimal = Field(gt=0)
    charged_cost_usd: Decimal = Field(ge=0)
    cost_basis: ProviderCallCostBasis


CheckpointCallback = Callable[
    [PendingCall | None, tuple[ProviderCallReceipt, ...]],
    Awaitable[None] | None,
]
CallModel = Callable[..., Awaitable[ModelAnswer]]


def _prompt_token_upper_bound(call: StageCall) -> int:
    """Bound prompt content by UTF-8 bytes, conservatively above BPE tokens."""

    return sum(len(message.content.encode("utf-8")) for message in call.messages)


def _provider_framing_token_allowance(call: StageCall) -> int:
    """Reserve fixed per-request and per-message provider framing overhead."""

    return 64 + (16 * len(call.messages))


def _reported_usage_cost(price: ModelPrice, usage: TokenUsage) -> Decimal:
    precision = max(
        64,
        len(str(usage.prompt_tokens))
        + len(price.input_ceiling_usd_per_million_tokens.as_tuple().digits)
        + 20,
        len(str(usage.completion_tokens))
        + len(price.output_ceiling_usd_per_million_tokens.as_tuple().digits)
        + 20,
    )
    with localcontext() as context:
        context.prec = precision
        cost = (
            Decimal(usage.prompt_tokens) * price.input_ceiling_usd_per_million_tokens
            + Decimal(usage.completion_tokens) * price.output_ceiling_usd_per_million_tokens
        ) / TOKENS_PER_MILLION
        return cost.quantize(USD_MICROCENT, rounding=ROUND_CEILING)


def _bounded_error_category(error: str) -> ProviderCallErrorCategory:
    if error in _BOUNDED_PROVIDER_ERROR_CATEGORIES:
        return error  # type: ignore[return-value]
    return _receipt_error_category(error)


class LiveProviderClient:
    """Guard all eval provider calls with serialization and exact cost accounting."""

    def __init__(
        self,
        *,
        price_book: PriceBook,
        hard_cap_usd: Decimal,
        checkpoint: CheckpointCallback,
        call_model_func: CallModel = call_model,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        if not isinstance(hard_cap_usd, Decimal):
            raise TypeError("hard_cap_usd must be a Decimal")
        if hard_cap_usd <= 0:
            raise ValueError("hard_cap_usd must be positive")
        self._price_book = price_book
        self._hard_cap_usd = hard_cap_usd
        self._checkpoint = checkpoint
        self._call_model = call_model_func
        self._temperature = temperature
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._pending_call: PendingCall | None = None
        self._receipts: list[ProviderCallReceipt] = []
        self._committed_cost_usd = Decimal("0")
        self._stopped = False

    @property
    def pending_call(self) -> PendingCall | None:
        return self._pending_call

    @property
    def receipts(self) -> tuple[ProviderCallReceipt, ...]:
        return tuple(self._receipts)

    @property
    def committed_cost_usd(self) -> Decimal:
        return self._committed_cost_usd

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _resolve_price(self, call: StageCall) -> ModelPrice:
        identity = (call.provider_id, call.model_id, call.model_revision)
        for price in self._price_book.entries:
            if price.identity == identity:
                return price
        raise ValueError(
            f"provider/model/revision is absent from the frozen price book: {'/'.join(identity)}"
        )

    async def _persist(self) -> None:
        result = self._checkpoint(self._pending_call, self.receipts)
        if inspect.isawaitable(result):
            await result

    def _reserve(self, call: StageCall, price: ModelPrice) -> CallReservation:
        return reserve_call_cost(
            price,
            prompt_token_upper_bound=_prompt_token_upper_bound(call),
            prompt_template_token_allowance=0,
            provider_framing_token_allowance=_provider_framing_token_allowance(call),
            upstream_output_token_ceilings=call.upstream_output_token_ceilings,
            max_output_tokens=call.max_output_tokens,
        )

    @staticmethod
    def _usage_breaches(usage: TokenUsage, reservation: CallReservation) -> bool:
        return (
            usage.prompt_tokens > reservation.input_token_upper_bound
            or usage.completion_tokens > reservation.output_token_upper_bound
        )

    async def call(self, call: StageCall) -> ModelAnswer:
        """Reserve, persist, execute, and reconcile one sequential stage call."""

        async with self._lock:
            if self._stopped:
                raise GatewayStoppedError("live provider gateway is stopped")

            price = self._resolve_price(call)
            reservation = self._reserve(call, price)
            if self._committed_cost_usd + reservation.reserved_cost_usd > self._hard_cap_usd:
                raise BudgetExceededError("provider call reservation would cross the hard cap")

            self._pending_call = PendingCall(
                stage=call.stage,
                provider_id=call.provider_id,
                model_id=call.model_id,
                model_revision=call.model_revision,
                max_output_tokens=call.max_output_tokens,
                reservation=reservation,
            )
            await self._persist()

            messages = [
                {"role": message.role, "content": message.content} for message in call.messages
            ]
            try:
                answer = await self._call_model(
                    call.provider_id,
                    call.model_id,
                    messages,
                    temperature=self._temperature,
                    timeout=self._timeout,
                    max_output_tokens=call.max_output_tokens,
                )
            except Exception as exc:  # noqa: BLE001 -- injected seam may raise
                raw_error = f"{type(exc).__name__}: {exc}"
                category = _bounded_error_category(raw_error)
                answer = ModelAnswer(
                    name=call.provider_id,
                    model_id=call.model_id,
                    error=category,
                )

            category: ProviderCallErrorCategory | None = None
            if answer.error is not None:
                category = _bounded_error_category(answer.error)
                answer = answer.model_copy(update={"error": category})

            usage = answer.usage
            if usage is None:
                charged_cost = reservation.reserved_cost_usd
                cost_basis = ProviderCallCostBasis(
                    source="full_reservation",
                    input_ceiling_usd_per_million_tokens=(
                        reservation.input_ceiling_usd_per_million_tokens
                    ),
                    output_ceiling_usd_per_million_tokens=(
                        reservation.output_ceiling_usd_per_million_tokens
                    ),
                )
                breached = False
            else:
                charged_cost = _reported_usage_cost(price, usage)
                cost_basis = ProviderCallCostBasis(
                    source="reported_usage",
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    input_ceiling_usd_per_million_tokens=(
                        reservation.input_ceiling_usd_per_million_tokens
                    ),
                    output_ceiling_usd_per_million_tokens=(
                        reservation.output_ceiling_usd_per_million_tokens
                    ),
                )
                breached = self._usage_breaches(usage, reservation)

            if breached:
                category = "reservation_breach"
                answer = answer.model_copy(update={"answer": None, "error": category})

            receipt = ProviderCallReceipt(
                stage=call.stage,
                provider_id=call.provider_id,
                model_id=call.model_id,
                model_revision=call.model_revision,
                max_output_tokens=call.max_output_tokens,
                outcome="failed" if category is not None else "success",
                error_category=category,
                usage=usage,
                reserved_cost_usd=reservation.reserved_cost_usd,
                charged_cost_usd=charged_cost,
                cost_basis=cost_basis,
            )
            self._committed_cost_usd += charged_cost
            self._receipts.append(receipt)
            self._pending_call = None
            if breached:
                self._stopped = True
            await self._persist()

            if breached:
                raise ReservationBreachError("provider usage breached the call reservation")
            return answer


__all__ = [
    "BudgetExceededError",
    "GatewayStoppedError",
    "LiveGatewayError",
    "LiveProviderClient",
    "PendingCall",
    "ProviderCallCostBasis",
    "ProviderCallReceipt",
    "ReservationBreachError",
]
