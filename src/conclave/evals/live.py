"""Sequential paid-provider gateway with pessimistic budget reservations."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from conclave.adapters.base import redact
from conclave.models import ModelAnswer, TokenUsage
from conclave.providers import _receipt_error_category, call_model
from conclave.registry import PROVIDER_ENV_VARS

from .live_protocols import StageCall
from .models import EVAL_SCHEMA_VERSION, EvalModel, RunRecord, Sha256Digest, StudyManifest
from .pricing import (
    TOKENS_PER_MILLION,
    USD_MICROCENT,
    CallReservation,
    ModelPrice,
    PriceBook,
    hash_price_entries,
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
    "interrupted",
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


class CheckpointValidationError(LiveGatewayError):
    """A checkpoint is corrupt, tampered, or bound to different study inputs."""


class CheckpointSecurityError(LiveGatewayError):
    """A checkpoint payload failed the no-secret persistence boundary."""


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


class CheckpointBindings(EvalModel):
    """Immutable inputs that make a checkpoint safe for exactly one live study."""

    manifest_hash: Sha256Digest
    price_book_hash: Sha256Digest
    public_tasks_hash: Sha256Digest
    hard_cap_usd: Decimal = Field(gt=0)


class ActiveCell(EvalModel):
    """Cell state persisted while a planned run is not yet a final record."""

    planned_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")
    receipt_start_index: int = Field(ge=0)
    pending_call: PendingCall | None = None


def _canonical_value(value: object) -> object:
    if isinstance(value, Decimal):
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"-0", ""} else text
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_hash(payload: object) -> str:
    canonical = json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


class LiveCheckpoint(EvalModel):
    """Secret-free, integrity-sealed state for one sequential live study."""

    bindings: CheckpointBindings
    committed_cost_usd: Decimal = Field(ge=0)
    records: tuple[RunRecord, ...] = ()
    receipts: tuple[ProviderCallReceipt, ...] = ()
    active_cell: ActiveCell | None = None
    checkpoint_hash: Sha256Digest

    @model_validator(mode="after")
    def validate_integrity(self) -> LiveCheckpoint:
        if self.checkpoint_hash != hash_live_checkpoint(self):
            raise ValueError("checkpoint integrity hash mismatch")
        record_ids = [record.planned_run_id for record in self.records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("checkpoint records must have unique planned_run_id values")
        if self.active_cell is not None:
            if self.active_cell.planned_run_id in record_ids:
                raise ValueError("active checkpoint cell must not already have a final record")
            if self.active_cell.receipt_start_index > len(self.receipts):
                raise ValueError("active checkpoint receipt_start_index exceeds receipt count")
        receipt_cost = sum((receipt.charged_cost_usd for receipt in self.receipts), Decimal("0"))
        if self.committed_cost_usd != receipt_cost:
            raise ValueError("checkpoint committed cost must equal receipt charges")
        if self.committed_cost_usd > self.bindings.hard_cap_usd:
            raise ValueError("checkpoint committed cost exceeds the hard cap")
        return self

    def should_execute(self, planned_run_id: str) -> bool:
        """Return false for every completed or currently interrupted cell."""

        if any(record.planned_run_id == planned_run_id for record in self.records):
            return False
        return self.active_cell is None or self.active_cell.planned_run_id != planned_run_id


def hash_study_manifest(manifest: StudyManifest) -> str:
    """Return the canonical digest of the exact frozen live manifest."""

    return _canonical_hash(manifest.model_dump(mode="python"))


def build_checkpoint_bindings(
    manifest: StudyManifest,
    price_book: PriceBook,
    *,
    hard_cap_usd: Decimal,
) -> CheckpointBindings:
    """Bind checkpoint state to the exact manifest, prices, tasks, and ceiling."""

    return CheckpointBindings(
        manifest_hash=hash_study_manifest(manifest),
        price_book_hash=hash_price_entries(price_book.entries),
        public_tasks_hash=manifest.public_tasks_hash,
        hard_cap_usd=hard_cap_usd,
    )


def hash_live_checkpoint(checkpoint: LiveCheckpoint) -> str:
    """Return a canonical digest over checkpoint content and all study bindings."""

    return _canonical_hash(checkpoint.model_dump(mode="python", exclude={"checkpoint_hash"}))


def create_live_checkpoint(
    *,
    bindings: CheckpointBindings,
    records: tuple[RunRecord, ...] = (),
    receipts: tuple[ProviderCallReceipt, ...] = (),
    active_cell: ActiveCell | None = None,
    committed_cost_usd: Decimal | None = None,
) -> LiveCheckpoint:
    """Create and integrity-seal one immutable checkpoint snapshot."""

    if committed_cost_usd is None:
        committed_cost_usd = sum((receipt.charged_cost_usd for receipt in receipts), Decimal("0"))
    unsealed = LiveCheckpoint.model_construct(
        schema_version=EVAL_SCHEMA_VERSION,
        bindings=bindings,
        committed_cost_usd=committed_cost_usd,
        records=tuple(records),
        receipts=tuple(receipts),
        active_cell=active_cell,
        checkpoint_hash="sha256:" + ("0" * 64),
    )
    payload = unsealed.model_dump(mode="python")
    payload["checkpoint_hash"] = hash_live_checkpoint(unsealed)
    return LiveCheckpoint.model_validate(payload)


def _active_provider_key_values() -> tuple[str, ...]:
    values = []
    for names in PROVIDER_ENV_VARS.values():
        for name in names:
            value = os.environ.get(name, "").strip()
            if value:
                values.append(value)
    return tuple(dict.fromkeys(values))


def _checkpoint_json(checkpoint: LiveCheckpoint) -> str:
    try:
        validated = LiveCheckpoint.model_validate(checkpoint.model_dump(mode="python"))
    except (ValidationError, ValueError) as exc:
        raise CheckpointValidationError("invalid live checkpoint state") from exc
    return (
        json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def write_live_checkpoint(path: str | Path, checkpoint: LiveCheckpoint) -> None:
    """Secret-scan and atomically replace a checkpoint in its destination directory."""

    destination = Path(path)
    payload = _checkpoint_json(checkpoint)
    if redact(payload) != payload or any(
        key_value in payload for key_value in _active_provider_key_values()
    ):
        raise CheckpointSecurityError("checkpoint payload contains secret material")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def load_live_checkpoint(
    path: str | Path,
    *,
    expected_bindings: CheckpointBindings,
) -> LiveCheckpoint:
    """Strictly load and bind an integrity-sealed checkpoint, failing closed."""

    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle, parse_float=Decimal)
        checkpoint = LiveCheckpoint.model_validate(payload)
    except json.JSONDecodeError as exc:
        raise CheckpointValidationError("invalid live checkpoint JSON") from exc
    except (OSError, TypeError, ValidationError, ValueError) as exc:
        label = "integrity" if "integrity" in str(exc).lower() else "invalid"
        raise CheckpointValidationError(f"{label} live checkpoint") from exc

    for field in (
        "manifest_hash",
        "price_book_hash",
        "public_tasks_hash",
        "hard_cap_usd",
    ):
        if getattr(checkpoint.bindings, field) != getattr(expected_bindings, field):
            raise CheckpointValidationError(f"checkpoint {field} drift")
    return checkpoint


def start_active_cell(checkpoint: LiveCheckpoint, *, planned_run_id: str) -> LiveCheckpoint:
    """Persist the start of a cell before its first provider reservation."""

    if checkpoint.active_cell is not None:
        raise CheckpointValidationError("a checkpoint cell is already active")
    if not checkpoint.should_execute(planned_run_id):
        raise CheckpointValidationError("planned run already has a final record")
    return create_live_checkpoint(
        bindings=checkpoint.bindings,
        records=checkpoint.records,
        receipts=checkpoint.receipts,
        active_cell=ActiveCell(
            planned_run_id=planned_run_id,
            receipt_start_index=len(checkpoint.receipts),
        ),
        committed_cost_usd=checkpoint.committed_cost_usd,
    )


def update_live_checkpoint(
    checkpoint: LiveCheckpoint,
    *,
    pending_call: PendingCall | None,
    receipts: tuple[ProviderCallReceipt, ...],
) -> LiveCheckpoint:
    """Apply one gateway persistence transition without rewriting prior history."""

    if checkpoint.active_cell is None:
        raise CheckpointValidationError("gateway state requires an active checkpoint cell")
    if (
        len(receipts) < len(checkpoint.receipts)
        or tuple(receipts[: len(checkpoint.receipts)]) != checkpoint.receipts
    ):
        raise CheckpointValidationError("gateway receipts must preserve checkpoint history")
    receipt_tuple = tuple(receipts)
    committed = sum((receipt.charged_cost_usd for receipt in receipt_tuple), Decimal("0"))
    return create_live_checkpoint(
        bindings=checkpoint.bindings,
        records=checkpoint.records,
        receipts=receipt_tuple,
        active_cell=checkpoint.active_cell.model_copy(update={"pending_call": pending_call}),
        committed_cost_usd=committed,
    )


def _interrupted_receipt(pending: PendingCall) -> ProviderCallReceipt:
    reservation = pending.reservation
    return ProviderCallReceipt(
        stage=pending.stage,
        provider_id=pending.provider_id,
        model_id=pending.model_id,
        model_revision=pending.model_revision,
        max_output_tokens=pending.max_output_tokens,
        outcome="failed",
        error_category="interrupted",
        reserved_cost_usd=reservation.reserved_cost_usd,
        charged_cost_usd=reservation.reserved_cost_usd,
        cost_basis=ProviderCallCostBasis(
            source="full_reservation",
            input_ceiling_usd_per_million_tokens=(reservation.input_ceiling_usd_per_million_tokens),
            output_ceiling_usd_per_million_tokens=(
                reservation.output_ceiling_usd_per_million_tokens
            ),
        ),
    )


def recover_interrupted_checkpoint(checkpoint: LiveCheckpoint) -> LiveCheckpoint:
    """Charge pending work and finalize the interrupted cell without retrying it."""

    active = checkpoint.active_cell
    if active is None:
        return checkpoint
    receipts = checkpoint.receipts
    if active.pending_call is not None:
        receipts = (*receipts, _interrupted_receipt(active.pending_call))
    cell_cost = sum(
        (receipt.charged_cost_usd for receipt in receipts[active.receipt_start_index :]),
        Decimal("0"),
    )
    record = RunRecord(
        planned_run_id=active.planned_run_id,
        outcome="incomplete",
        error_category="interrupted",
        cost_usd=float(cell_cost),
        cost_receipt_complete=True,
        deviation_codes=("interrupted_cell_not_retried",),
    )
    return create_live_checkpoint(
        bindings=checkpoint.bindings,
        records=(*checkpoint.records, record),
        receipts=receipts,
        committed_cost_usd=sum((receipt.charged_cost_usd for receipt in receipts), Decimal("0")),
    )


def finish_active_cell(checkpoint: LiveCheckpoint, *, record: RunRecord) -> LiveCheckpoint:
    """Finalize the active cell while preserving all receipt and cost history."""

    active = checkpoint.active_cell
    if active is None:
        raise CheckpointValidationError("a final record requires an active checkpoint cell")
    if active.planned_run_id != record.planned_run_id:
        raise CheckpointValidationError("final record does not match the active checkpoint cell")
    return create_live_checkpoint(
        bindings=checkpoint.bindings,
        records=(*checkpoint.records, record),
        receipts=checkpoint.receipts,
        committed_cost_usd=checkpoint.committed_cost_usd,
    )


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
        resume_from: LiveCheckpoint | None = None,
    ) -> None:
        if not isinstance(hard_cap_usd, Decimal):
            raise TypeError("hard_cap_usd must be a Decimal")
        if hard_cap_usd <= 0:
            raise ValueError("hard_cap_usd must be positive")
        if resume_from is not None:
            if resume_from.active_cell is not None:
                raise CheckpointValidationError(
                    "recover the active checkpoint cell before resuming the gateway"
                )
            if resume_from.bindings.hard_cap_usd != hard_cap_usd:
                raise CheckpointValidationError("checkpoint hard_cap_usd drift")
        self._price_book = price_book
        self._hard_cap_usd = hard_cap_usd
        self._checkpoint = checkpoint
        self._call_model = call_model_func
        self._temperature = temperature
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._pending_call: PendingCall | None = None
        self._receipts = list(resume_from.receipts) if resume_from is not None else []
        self._committed_cost_usd = (
            resume_from.committed_cost_usd if resume_from is not None else Decimal("0")
        )
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
        try:
            result = self._checkpoint(self._pending_call, self.receipts)
            if inspect.isawaitable(result):
                await result
        except BaseException:
            self._stopped = True
            raise

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
    "ActiveCell",
    "BudgetExceededError",
    "CheckpointBindings",
    "CheckpointSecurityError",
    "CheckpointValidationError",
    "GatewayStoppedError",
    "LiveCheckpoint",
    "LiveGatewayError",
    "LiveProviderClient",
    "PendingCall",
    "ProviderCallCostBasis",
    "ProviderCallReceipt",
    "ReservationBreachError",
    "build_checkpoint_bindings",
    "create_live_checkpoint",
    "finish_active_cell",
    "hash_live_checkpoint",
    "hash_study_manifest",
    "load_live_checkpoint",
    "recover_interrupted_checkpoint",
    "start_active_cell",
    "update_live_checkpoint",
    "write_live_checkpoint",
]
