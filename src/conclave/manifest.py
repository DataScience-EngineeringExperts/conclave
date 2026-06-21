"""ModelHarnessManifest — the auditable receipt of WHAT ran and HOW (CAC-04).

A :class:`ModelHarnessManifest` is first-class on **every**
:class:`conclave.models.CouncilResult` (not behind a debug flag — Scope Plan §3).
It records, in one secret-free object:

* WHAT ran — ``request_id``, ``conclave_version``, deliberation ``mode``,
  providers considered / called / **skipped (with reasons)**, the concrete
  resolved model ids, the generation settings used, per-member execution
  receipts, total latency, and total token usage;
* cost (carefully — Scope Plan §8) — token ``total_usage`` is always present;
  ``estimated_cost`` is left ``None`` (a wrong number inside an audit receipt is
  worse than none) and ``pricing_snapshot_date`` is the dated-estimate slot a
  later pricing table would stamp;
* HOW the verdict was made — ``verdict_extraction`` provenance (which model +
  prompt version produced the disagreement analysis), ``verdict_type``,
  ``consensus_method``, and the ``verdict_absent_reason`` (DD-2 ripple). These
  verdict-provenance slots are defined here but **populated by CAC-05**; CAC-04
  leaves them ``None``.

**Secret-safety (Scope Plan §5 — non-negotiable).** Key VALUES never appear in a
manifest. Per-member errors are redacted upstream (in :mod:`conclave.providers`)
before they reach a receipt, and :func:`receipt_from_answer` re-applies
:func:`conclave.adapters.base.redact` belt-and-suspenders. After assembling the
manifest the council runs :func:`scan_for_secret_material` and stamps
``secret_safety`` VERIFIED only when the serialized manifest is provably clean.

This module deliberately does NOT import :mod:`conclave.models`; the dependency
runs the other way (``models`` imports the manifest types and calls
``model_rebuild()``) so there is no cycle — the same no-cycle pattern
:mod:`conclave.verdict` uses. It DOES import :class:`~conclave.models.TokenUsage`
(a leaf type with no back-edge to the manifest) for the usage fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import TokenUsage

# ``secret_safety`` status literals. UNVERIFIED is the safe default (a manifest is
# untrusted until the self-scan proves it clean); VERIFIED is stamped only by
# :func:`verified_secret_safety` when :func:`scan_for_secret_material` returns True.
SECRET_SAFETY_VERIFIED = "verified_no_secrets"
SECRET_SAFETY_UNVERIFIED = "unverified"

# The forbidden raw patterns the self-scan looks for (case-insensitive). A
# serialized manifest containing any of these has leaked key/header/raw-body
# material. The redaction marker ``[REDACTED]`` is intentionally NOT here — a
# redacted error string legitimately carries it and must not trip the scan.
_FORBIDDEN_SUBSTRINGS = ("sk-", "bearer", "authorization", "api_key", "x-api-key")


class ProviderSkip(BaseModel):
    """One council member skipped before any call was made.

    A skipped member never produced a receipt (it was never called); it is
    recorded here with a human-readable reason so the manifest can answer "why
    isn't provider X in the result?" without exposing anything secret.

    Attributes:
        name: Friendly council member name (e.g. ``"grok"``).
        reason: Why it was skipped (e.g. ``"no API key in environment"``).
    """

    name: str
    reason: str


class VerdictExtraction(BaseModel):
    """Provenance of the verdict-extraction step (DD-2 ripple, filled by CAC-05).

    Records WHICH model and prompt version produced the disagreement/consensus
    analysis, so a human can audit the extractor's identity rather than trusting
    an opaque score (the Scope Plan §4.1 auditability-paradox fix). Both fields
    default ``None``; CAC-04 never populates them — the verdict-extraction step
    (CAC-05) does.

    Attributes:
        model_id: Resolved provider-prefixed id of the extractor model, or
            ``None`` until CAC-05 records it.
        prompt_version: The extractor prompt version tag, or ``None`` until
            CAC-05 records it.
    """

    model_id: str | None = None
    prompt_version: str | None = None


class ProviderExecutionReceipt(BaseModel):
    """A per-call execution record for one council member that was CALLED.

    Built from a returned :class:`~conclave.models.ModelAnswer` by
    :func:`receipt_from_answer` (a skipped member has a :class:`ProviderSkip`
    instead, never a receipt). Carries only non-secret, auditable facts: the
    settings actually used, the latency, the token usage, and a redacted error.

    Attributes:
        name: Friendly council member name (e.g. ``"grok"``).
        provider: Provider prefix derived from ``model_id`` (e.g. ``"xai"``).
        model_id: Resolved provider-prefixed model id (e.g. ``"xai/grok-4.3"``).
        generation_settings: The settings actually used for the call
            (``{"temperature": ..., "timeout": ...}``).
        latency_ms: Wall-clock latency of the call in milliseconds.
        usage: Token usage if the provider reported it, else ``None``.
        error: Redacted error message if the call failed, else ``None``. This
            field NEVER holds key material — it is redacted upstream and again on
            construction (belt-and-suspenders).
        schema_valid: Whether the member's structured output validated. ``None``
            until CAC-02 structured output exists; defined now, populated later.
    """

    name: str
    provider: str
    model_id: str
    generation_settings: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    usage: TokenUsage | None = None
    error: str | None = None
    schema_valid: bool | None = None


class ModelHarnessManifest(BaseModel):
    """The auditable execution + provenance receipt for a council run (§3).

    First-class on every :class:`conclave.models.CouncilResult`. Trivially
    constructible: only ``request_id``, ``conclave_version``, and ``mode`` are
    required; every collection/usage/provenance field defaults to an empty or
    ``None`` value so the council can assemble it incrementally and so the
    empty-members path can still attach a complete manifest.

    Attributes:
        request_id: Unique id for this run (``uuid4().hex``), generated by the
            council. Lets a downstream audit correlate logs to this exact run.
        conclave_version: The :data:`conclave.__version__` that produced the run.
        mode: Deliberation mode (``synthesize``/``raw``/``debate``/``adversarial``).
        providers_considered: All requested friendly member names, in order.
        providers_called: Friendly names of members that had a key and were
            actually fanned out.
        providers_skipped: Members skipped before any call, each with a reason.
        model_ids: Concrete resolved model ids of the called members.
        generation_settings: Council-level settings used for member calls
            (``{"temperature": ..., "timeout": ...}``).
        receipts: One :class:`ProviderExecutionReceipt` per called member.
        total_latency_ms: Sum of per-member latencies in milliseconds.
        total_usage: Token usage summed across receipts, or ``None`` when no
            member reported usage.
        estimated_cost: Left ``None`` (Scope Plan §8 — no invented pricing table;
            a wrong number in an audit receipt is worse than none).
        pricing_snapshot_date: The dated-estimate slot a pricing table would
            stamp; ``None`` until such a table exists.
        schema_valid: Overall structured-output validity; ``None`` until CAC-02.
        redacted_errors: Member error strings, already redacted upstream.
        secret_safety: Status literal — :data:`SECRET_SAFETY_UNVERIFIED` by
            default; set to :data:`SECRET_SAFETY_VERIFIED` by the council after
            :func:`scan_for_secret_material` confirms the manifest is clean.
        verdict_extraction: Verdict-extraction provenance (DD-2 ripple). Default
            empty; populated by CAC-05.
        verdict_type: The extractor's ``verdict_type`` classification, or ``None``
            until CAC-05 records it.
        consensus_method: The consensus method literal used, or ``None`` until
            CAC-05 records it (DD-2 ripple).
        verdict_absent_reason: Why ``result.verdict`` is ``None`` (open-ended
            generation, N<2, or structured-extraction failure), or ``None`` when
            a verdict is present / not yet computed (DD-2 ripple, filled by CAC-05).
    """

    # REQUIRED identity.
    request_id: str
    conclave_version: str
    mode: str

    # Providers (considered / called / skipped) + resolved ids.
    providers_considered: list[str] = Field(default_factory=list)
    providers_called: list[str] = Field(default_factory=list)
    providers_skipped: list[ProviderSkip] = Field(default_factory=list)
    model_ids: list[str] = Field(default_factory=list)

    # Settings + execution receipts + aggregate latency/usage.
    generation_settings: dict[str, float] = Field(default_factory=dict)
    receipts: list[ProviderExecutionReceipt] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_usage: TokenUsage | None = None

    # Cost (carefully — §8): usage above is hard data; cost is the optional,
    # dated-estimate slot left unfilled until a pricing table exists.
    estimated_cost: float | None = None
    pricing_snapshot_date: str | None = None

    # Structured-output validity (CAC-02) + redacted member errors.
    schema_valid: bool | None = None
    redacted_errors: list[str] = Field(default_factory=list)

    # Secret-safety status (set VERIFIED by the council after the self-scan).
    secret_safety: str = SECRET_SAFETY_UNVERIFIED

    # Verdict provenance (DD-2 ripple) — defined here, populated by CAC-05.
    verdict_extraction: VerdictExtraction = Field(default_factory=VerdictExtraction)
    verdict_type: str | None = None
    consensus_method: str | None = None
    verdict_absent_reason: str | None = None


def scan_for_secret_material(manifest: ModelHarnessManifest) -> bool:
    """Return True when the serialized manifest is CLEAN of key material.

    Serializes the manifest with ``model_dump_json()`` and case-insensitively
    checks for any forbidden raw pattern (:data:`_FORBIDDEN_SUBSTRINGS`:
    ``sk-``, ``bearer``, ``authorization``, ``api_key``, ``x-api-key``). The
    redaction marker ``[REDACTED]`` is deliberately not forbidden — a redacted
    error string legitimately carries it.

    This is the single source of truth for the secret-absence check (the
    load-bearing CAC-04 acceptance criterion); both the council and the test
    suite call it rather than re-implementing the substring set.

    Args:
        manifest: The assembled manifest to inspect.

    Returns:
        ``True`` if no forbidden substring is present (the manifest is clean),
        ``False`` if any is found (key/header/raw-body material may have leaked).
    """
    serialized = manifest.model_dump_json().lower()
    return not any(token in serialized for token in _FORBIDDEN_SUBSTRINGS)


def verified_secret_safety(manifest: ModelHarnessManifest) -> str:
    """Return the ``secret_safety`` status literal for a manifest.

    The council assigns ``manifest.secret_safety = verified_secret_safety(manifest)``
    after assembly, so the VERIFIED stamp is granted only when the self-scan
    passes. A failing scan leaves the manifest at :data:`SECRET_SAFETY_UNVERIFIED`
    (the safe default) so a downstream auditor can see the manifest was not
    proven clean.

    Args:
        manifest: The assembled manifest to verify.

    Returns:
        :data:`SECRET_SAFETY_VERIFIED` when :func:`scan_for_secret_material`
        passes, else :data:`SECRET_SAFETY_UNVERIFIED`.
    """
    return (
        SECRET_SAFETY_VERIFIED if scan_for_secret_material(manifest) else SECRET_SAFETY_UNVERIFIED
    )
