"""The Council: concurrent multi-model fan-out plus synthesis.

``Council`` is the primary importable entry point. It resolves friendly names to
provider-prefixed model ids, skips any member whose API key is absent, fans the prompt out
concurrently, collects partial results even when some members fail, and (in
synthesize mode) asks a synthesizer model to merge the answers into one.

The deliberation modes (``debate``, ``adversarial``) live in :mod:`conclave.modes`
and reuse this class's :meth:`Council.fan_out` primitive so the partial-failure
handling is written exactly once.

Synthesizer selection and degradation (the "council" value prop)
----------------------------------------------------------------

**Which model synthesizes.** Synthesis is performed by one *synthesizer* model,
separate from the council members (though a member may also be the synthesizer).
Selection precedence, highest first:

1. the ``synthesizer=`` argument to :class:`Council` (the CLI ``--synthesizer/-s``
   flag wires straight through to this);
2. the ``synthesizer:`` key in ``~/.conclave/config.yml``;
3. the built-in default :data:`conclave.registry.DEFAULT_SYNTHESIZER` (``"claude"``,
   i.e. ``anthropic/claude-sonnet-4-6``).

The same model is the **judge** in ``adversarial`` mode and the final
consolidator in ``debate`` mode -- one selection drives all three.

**The fallback / degraded path is OBSERVABLE, never silent.** Synthesis can fail
to run for three reasons, and each one is signaled on the result rather than
silently swallowed:

* *No usable member answers* (every member errored/skipped) -- nothing to merge;
* *The synthesizer has no API key* in the environment;
* *The synthesizer call itself fails* (provider error/timeout).

In all three cases ``CouncilResult.synthesis`` stays ``None``, the member answers
are still returned intact, a warning is logged, and an actionable reason is set
on ``CouncilResult.synthesis_error`` (in ``adversarial`` mode the analogous
``AdversarialResult.verdict_error``, mirrored to ``synthesis_error``). A caller
can therefore always tell synthesis did **not** happen as expected by checking
``synthesis is None and synthesis_error is not None`` -- there is no path where
the council quietly returns concatenated/partial output dressed up as a synthesis.

**The synthesis prompt is a versioned constant.** The synthesize-mode system
prompt is :data:`_SYNTH_SYSTEM` (the debate/judge prompts live in
:mod:`conclave.prompts`); the prompt *set* carries the version tag
:data:`conclave.prompts.SYNTHESIS_PROMPT_VERSION`, stamped onto every
:class:`~conclave.models.CouncilResult` as ``prompt_version`` so a prompt change
is detectable downstream instead of being silently absorbed as model drift.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

from . import cache as cache_mod
from . import prompts, transport
from .adapters.base import redact
from .config import ConclaveConfig, load_config, parse_synthesizer_chain
from .logging import get_logger
from .manifest import (
    AdjudicationAttempt,
    AdjudicationAttemptOutcome,
    AdjudicationRole,
    ModelHarnessManifest,
    ProviderExecutionReceipt,
    ProviderSkip,
    verified_secret_safety,
)
from .models import (
    ELITE_PROTOCOL_VERSION,
    FAILOVER_CATEGORIES,
    CouncilResult,
    ModelAnswer,
    StreamEvent,
    TokenUsage,
)
from .pricing import (
    PriceRates,
    PriceSnapshot,
    SpendCapExceeded,
    SpendUnboundable,
    load_default_price_snapshot,
    reported_usage_cost,
    reserve_cost,
)
from .prompts import ELITE_PROMPT_VERSION, SYNTHESIS_PROMPT_VERSION
from .providers import call_model, receipt_from_answer
from .registry import key_present

if TYPE_CHECKING:  # avoid an import cycle at runtime; only needed for typing
    from .verdict_synthesis import VerdictSynthesisResult

logger = get_logger("council")

# A per-member message-list factory: given a (friendly_name, model_id) member,
# return the OpenAI-style messages to send it. Lets each mode tailor the prompt
# per member while sharing Council.fan_out's concurrency + partial-failure code.
MessagesFor = Callable[[str, str], list[dict[str, str]]]

# The synthesize-mode system prompt. It is a stable module constant -- never
# built per-call -- so the wording the council synthesizes under is auditable and
# diffable. Any change to it (or to the debate/judge prompts in
# :mod:`conclave.prompts`) MUST be paired with a bump of
# :data:`conclave.prompts.SYNTHESIS_PROMPT_VERSION`, which is stamped onto every
# :class:`~conclave.models.CouncilResult` as ``prompt_version`` so a downstream
# eval can detect the change rather than silently absorb it. ``test_synthesizer``
# pins both this text and the version, so editing one without the other fails CI.
_SYNTH_SYSTEM = (
    "You are the synthesizer of a council of AI models. You are given the same "
    "user prompt that was posed to several models, plus each model's answer. "
    "Produce one consolidated, accurate answer. Reconcile agreements, surface "
    "and adjudicate disagreements, and note any answer that is clearly wrong. "
    "Do not invent a model's position; rely only on the answers provided."
)


def _synth_user_content(prompt: str, answers: Sequence[ModelAnswer]) -> str:
    """Build the synthesizer's user-message content from prompt + member answers.

    Extracted from :meth:`Council._synthesize` (unchanged behavior, same
    string) so :meth:`Council._plan_table` can measure the EXACT fixed
    wrapper/label bytes a real synthesis call embeds -- via
    :func:`_placeholder_answers`, empty-text answers with the real names this
    run would use -- instead of a hand-duplicated approximation that could
    silently drift from the real prompt (DSE-1514 review, Fix A).
    """
    blocks = "\n\n".join(
        f"### Answer from {a.name} ({a.model_id})"
        f"{f' (Answer ID: {a.answer_id})' if a.answer_id else ''}\n{a.answer}"
        for a in answers
    )
    return (
        f"Original prompt:\n{prompt}\n\n"
        f"Council answers:\n\n{blocks}\n\n"
        "Now produce the consolidated answer."
    )


# Conservative byte allowance for an optional answer_id in a template probe
# (DSE-1514 review, Fix A). A real id is "ca_" + 24 hex chars (27 bytes), or,
# for a phase-derived artifact (:func:`conclave.models.derive_phase_answer_id`),
# "ca_<phase>_" + 24 hex chars -- a few bytes longer for any phase name
# conclave uses today. A placeholder at least this long makes a template
# probe's label byte count an upper bound of BOTH the with-id and the
# without-id (positional-fallback) real cases, never an under-count.
_ANSWER_ID_PROBE = "ca_" + ("f" * 45)


def _placeholder_answers(members: Sequence[tuple[str, str]]) -> list[ModelAnswer]:
    """Zero-length, worst-case-id answers for measuring a template's real fixed bytes.

    Used only by :meth:`Council._plan_table` to measure a phase's fixed
    per-item label overhead (member name, model id, and/or an optional answer
    id) EXACTLY -- using the real names/ids the run would actually use --
    rather than approximating it with a guessed constant. The answer TEXT is
    deliberately left empty: that variable part is bounded separately by the
    per-call ``upstream_output_call_count`` times the output cap.

    Args:
        members: The ``(friendly_name, model_id)`` pairs a template needs one
            placeholder answer per.

    Returns:
        One placeholder :class:`~conclave.models.ModelAnswer` per member, in
        the same order.
    """
    return [
        ModelAnswer(name=name, model_id=model_id, answer="", answer_id=_ANSWER_ID_PROBE)
        for name, model_id in members
    ]


# The modes Council.plan_calls knows how to enumerate (DSE-1514). Kept as its own
# frozenset (rather than re-deriving from _RENDERERS or similar) so the planner's
# contract is explicit and independent of any CLI-only vocabulary.
_VALID_PLAN_MODES = frozenset({"raw", "synthesize", "vote", "debate", "adversarial", "elite"})

# The exact refusal message when a spend cap is requested but output is
# unbounded (DSE-1514): shared verbatim by Council.plan_calls, the Council
# constructor's max_spend_usd guard, and the CLI so the message a library caller
# sees and the message a CLI user sees are byte-identical.
_NO_OUTPUT_CAP_MESSAGE = (
    "cannot bound spend: no output cap (set --max-output-tokens or config max_output_tokens)"
)

# The closed vocabulary of pricing_warnings identifiers (DSE-1514). Every
# append to the ``warnings`` list -- or the list-literal assignment on the
# missing-snapshot path -- inside :meth:`Council._price_manifest` uses one of
# these exact strings, never interpolated text, a provider name, or a count,
# so a warning can never carry secret-shaped material. Tests in
# tests/test_secret_safety_matrix.py assert both statically (scanning the
# source for every append call) and dynamically (every reachable warning
# shape across every mode) that nothing else is ever appended.
PRICING_WARNING_VOCABULARY = frozenset(
    {
        "price_snapshot_stale",
        "price_snapshot_unavailable",
        "unpriced_models_present",
        "unpriced_receipts_present",
        "no_output_cap_configured",
    }
)

# Re-exported for callers that want the version without importing prompts.
__all__ = ["Council", "SYNTHESIS_PROMPT_VERSION"]


def _usage_is_reported(usage: TokenUsage | None) -> bool:
    """Whether ``usage`` is a trustworthy, non-zero signal worth pricing (DSE-1514 QA I1).

    ``usage is None`` is not proof a call cost nothing: it is the shape of both
    a FAILED call (no usage was ever produced) and a SUCCESSFUL one whose
    provider simply omitted the usage field --
    :func:`conclave.adapters.openai_compat`'s chat-completion path returns
    ``None`` when a provider does this, including for some streamed responses.
    And a provider can report a technically-present ``TokenUsage`` that is all
    zeros, which is the same "nothing to price" signal wearing a non-``None``
    shape. Treating either as ``$0.000000`` would assert a false floor rather
    than an honest bound, so :meth:`Council._price_manifest` calls this helper
    to decide "does this receipt carry a number worth pricing" before ever
    looking at the ceiling math.

    Args:
        usage: The receipt's :class:`~conclave.models.TokenUsage`, or ``None``.

    Returns:
        ``True`` only when ``usage`` is present AND at least one of its three
        counters is non-zero.
    """
    return usage is not None and bool(
        usage.prompt_tokens or usage.completion_tokens or usage.total_tokens
    )


@dataclass
class AdjudicationOutcome:
    """Return value of :meth:`Council.adjudicate`.

    ``answer`` is the successful answer, or the terminal / exhausted failure, or
    ``None`` when no candidate could be called at all (every one unkeyed).
    ``called`` lists every REAL call in order (for receipts); ``attempts`` is the
    full ledger including skipped candidates.
    """

    answer: ModelAnswer | None
    attempts: list[AdjudicationAttempt]
    called: list[ModelAnswer]

    @property
    def name(self) -> str | None:
        return self.answer.name if self.answer is not None else None

    @property
    def model_id(self) -> str | None:
        return self.answer.model_id if self.answer is not None else None


@dataclass(frozen=True)
class PlannedCall:
    """One provider call a mode COULD make, bounded without making it.

    Every field is knowable before the first call: the resolved model, the
    output cap the call will be issued with, and a three-part input bound (exact
    prompt bytes, fixed template bytes, provider framing) plus a count of
    upstream calls whose not-yet-produced output this call's input will embed.

    Attributes:
        phase: The manifest phase this call would be recorded under.
        name: Friendly member / candidate name.
        model_id: Resolved provider-prefixed model id.
        prompt_token_upper_bound: UTF-8 bytes of the exact known content.
        prompt_template_token_allowance: UTF-8 bytes of the fixed system + user
            template wording that will surround it.
        provider_framing_token_allowance: ``64 + 16 * messages`` (+256 with a
            structured-output contract), mirroring the eval runner.
        upstream_output_call_count: How many upstream calls' outputs this call's
            input embeds. Multiplied by the output cap and the snapshot's
            ``max_output_bytes_per_token`` to bound them.
        max_output_tokens: The hard output cap this call would carry.
    """

    phase: str
    name: str
    model_id: str
    prompt_token_upper_bound: int
    prompt_template_token_allowance: int
    provider_framing_token_allowance: int
    upstream_output_call_count: int
    max_output_tokens: int

    def input_bytes_bound(self, *, upstream_output_bytes_per_token: int) -> int:
        """Total planned input-byte bound for one candidate rate (DSE-1514 review, Fix A).

        Mirrors the input-side arithmetic :meth:`Council._reserve_plan` feeds
        into :func:`conclave.pricing.reserve_cost` byte for byte, so a test can
        assert a real, unplanned message list never exceeds what this call was
        priced for without duplicating (and risking drifting from) the pricing
        module's own formula.

        Args:
            upstream_output_bytes_per_token: The priced model's attested
                upper bound on one output token's UTF-8 byte length (a
                :class:`conclave.pricing.PriceRates` field) -- the same value
                :meth:`Council._reserve_plan` reads off the snapshot entry.

        Returns:
            The upper bound, in bytes, on everything this call's input could
            contain: the exact known prompt, the fixed template wording, the
            provider framing allowance, and every upstream call's output cap
            converted to bytes.
        """
        return (
            self.prompt_token_upper_bound
            + self.prompt_template_token_allowance
            + self.provider_framing_token_allowance
            + (
                self.upstream_output_call_count
                * self.max_output_tokens
                * upstream_output_bytes_per_token
            )
        )


@dataclass(frozen=True)
class CallPlan:
    """The complete worst-case call plan for one run of one mode."""

    mode: str
    calls: tuple[PlannedCall, ...]
    member_count: int
    chain_count: int


@dataclass(frozen=True)
class _PhaseSpec:
    """One declarative row of a mode's worst-case call table (DSE-1514 review, Fix B).

    :meth:`Council._plan_table` returns one of these per phase a mode could
    run; :meth:`Council.plan_calls` expands each row into one
    :class:`PlannedCall` per target, using the exact prompt bytes and the
    output cap (the two values every row needs but none of them determine).
    ``targets`` is the literal slice of keyed members or keyed chain
    candidates this row calls -- a plain list, not a count -- so the SAME row
    shape covers both a uniform "every member" phase (``targets`` is every
    keyed member) and adversarial's split single-proposer /
    ``(N-1)``-critics shape (two rows, each a different slice of ``members``).

    Attributes:
        phase: The manifest phase this call would be recorded under.
        targets: The exact ``(name, model_id)`` pairs this row calls.
        template: The fixed system + user template wording that will
            surround the exact prompt bytes.
        upstream: How many upstream calls' not-yet-produced output this
            call's input embeds.
        message_count: Messages this call sends, feeding the provider framing
            allowance (``64 + 16 * message_count``, mirroring the eval
            runner). Member-shaped phases send 2 (system + user); chain-shaped
            phases send 3 (system + user + the assembled upstream material).
        contract: Whether a structured-output contract is attached. Adds a
            flat 256-byte provider framing allowance for the schema
            registration itself; the schema's own byte cost is measured as
            part of ``template`` (built via
            :func:`conclave.verdict_synthesis._build_messages`, real member
            placeholders, in :meth:`Council._plan_table`), never added twice.
    """

    phase: str
    targets: Sequence[tuple[str, str]]
    template: str
    upstream: int
    message_count: int = 2
    contract: bool = False

    def to_planned_call(
        self,
        *,
        name: str,
        model_id: str,
        prompt_token_upper_bound: int,
        max_output_tokens: int,
    ) -> PlannedCall:
        """Expand this row into one :class:`PlannedCall` for ``(name, model_id)`` (DSE-1514).

        The template-bytes/framing-bytes arithmetic (a structured-output
        contract's schema bytes live INSIDE ``template`` already; ``contract``
        only adds the flat provider-side framing allowance for the schema
        registration itself, never a second copy of the schema) is written
        here exactly once and shared by two callers: :meth:`Council.plan_calls`
        (expanding every target in ``self.targets``, BEFORE any call is made)
        and :meth:`Council._price_manifest`'s phase-aware reservation (Round 4,
        QA C1 -- pricing ONE already-made, usage-less receipt from the row
        matching its phase).

        Args:
            name: Friendly member / candidate name for this specific call.
            model_id: Resolved provider-prefixed model id for this call.
            prompt_token_upper_bound: UTF-8 bytes of the exact known prompt
                content this call sends.
            max_output_tokens: The hard output cap this call would carry.

        Returns:
            The expanded :class:`PlannedCall`.
        """
        return PlannedCall(
            phase=self.phase,
            name=name,
            model_id=model_id,
            prompt_token_upper_bound=prompt_token_upper_bound,
            prompt_template_token_allowance=len(self.template.encode("utf-8")),
            provider_framing_token_allowance=(
                64 + (16 * self.message_count) + (256 if self.contract else 0)
            ),
            upstream_output_call_count=self.upstream,
            max_output_tokens=max_output_tokens,
        )


class Council:
    """A council of foundation models with an optional synthesizer.

    Args:
        models: Friendly names (or raw provider-prefixed model ids) of council members.
        synthesizer: Friendly name of the synthesizer model, an ordered chain
            spec (``"claude>grok>gemini"``), or a list of names
            (``["claude", "grok"]``). If ``None``, ``config.synthesizer_chain``
            is used when set, else ``config.synthesizer`` (a chain of one).
            Whichever form resolves, ``self.synthesizer`` keeps its historic
            meaning: the primary (first) candidate. The full ordered ladder is
            ``self.synthesizer_chain`` (DSE-1512); :meth:`adjudicate` walks it
            and advances past a candidate only on an infrastructure failure
            (see :data:`conclave.models.FAILOVER_CATEGORIES`) -- any other
            failure is terminal for the role.
        config: Pre-loaded config; if ``None``, loaded from disk + defaults.
        temperature: Sampling temperature for member calls.
        timeout: Per-call timeout in seconds.
        cache: Opt-in result cache. ``None`` (default) defers to
            ``config.cache`` (off unless enabled in ``~/.conclave/config.yml``);
            ``True``/``False`` overrides it for this council. When enabled, an
            identical repeat run is served from the on-disk cache instead of
            re-calling the providers. The cache never persists API keys --
            see :mod:`conclave.cache`.
        extract_verdict: Whether to run the structured verdict-extraction step
            (CAC-05) after a synthesize-mode run. Defaults to ``True`` -- the
            auditable verdict (consensus score, conflicts, provider votes) is the
            council's product, so it is on by default. **Cost note:** verdict
            extraction makes a SECOND synthesizer call (one extra LLM round-trip
            per ``ask``, plus one more on the single repair retry) distinct from
            the prose ``synthesis`` call. Subsuming both into a single synthesizer
            call is a future optimization; for now this flag is the single opt-out.
            Set ``False`` to skip the verdict entirely (``CouncilResult.verdict``
            stays ``None`` and the manifest's verdict-provenance slots stay at
            their defaults). Verdict extraction never runs in ``raw`` mode
            (``synthesize=False``) regardless of this flag.
        allow_transport_debug_logging: Opt **out** of the transport-logging guard.
            Defaults to ``False``, which means the guard is **ON**: constructing a
            ``Council`` installs :func:`conclave.transport.guard_transport_logging`
            so httpx/httpcore ``DEBUG`` records -- the only band that emits request
            headers, including the live ``Authorization``/``x-api-key`` value -- are
            dropped before any handler formats them (key-leak audit, RANK 6). The
            guard is idempotent, so constructing many councils installs it once. The
            filter is scoped to the ``httpx``/``httpcore`` loggers only; it never
            touches the host application's root logger or any other logger.
            Set ``True`` to skip installation for the rare case where you genuinely
            need httpx/httpcore ``DEBUG`` output in a process that does not hold real
            keys; you remain responsible for that band then. Consumers using the
            provider functions directly (without a ``Council``) can still call
            :func:`conclave.guard_transport_logging` themselves.
        source_bundle_digest: Optional digest of a future source-grounding bundle.
            When supplied, it participates in cache identity so grounded and
            ungrounded Elite runs cannot collide. The value is re-hashed before
            entering the canonical identity document.
        max_output_tokens: Opt-in hard ceiling on output tokens for EVERY call
            this council makes -- members, synthesis, judge, verdict extraction
            and its repair retry, and the streaming paths (DSE-1514). ``None``
            (the default) defers to ``config.max_output_tokens`` (itself
            ``None`` unless set), leaving each provider's own default in place
            exactly as before this flag existed. A cap is the precondition for
            ``--max-spend-usd``: see :meth:`plan_calls`.
        max_spend_usd: Opt-in pre-flight spend cap in USD (DSE-1514). When set,
            every deliberation call (:meth:`ask`/:meth:`ask_stream` and their
            mode wrappers -- :meth:`debate`, :meth:`adversarial`, :meth:`vote`,
            :meth:`elite`, and their ``_sync`` variants) first enumerates
            :meth:`plan_calls`, prices it via :meth:`_reserve_plan`, and raises
            :class:`conclave.pricing.SpendCapExceeded` -- BEFORE any provider
            call -- when the reserved total exceeds this cap. Requires
            ``max_output_tokens`` (explicit or via config): an unbounded output
            cannot be bounded in dollars, so setting this without a cap raises
            :class:`conclave.pricing.SpendUnboundable` at construction time
            rather than at the first call. ``None`` (the default) installs no
            gate at all -- byte-identical to today.

            **The gate has exactly one chokepoint, :meth:`_cached_run` (plus
            :meth:`ask_stream`'s own call to the same :meth:`_gate_live_run`),
            and nothing routes around it implicitly.** A caller that reaches
            into a lower-level primitive DIRECTLY -- :meth:`fan_out`,
            :meth:`synthesize_blocks`, :meth:`adjudicate`,
            :func:`conclave.verdict_synthesis.extract_verdict`, or any
            :mod:`conclave.modes` ``run_*`` function called without going
            through the matching :class:`Council` method -- makes real
            provider calls WITHOUT ever consulting ``max_spend_usd``, however
            large. This is a deliberate seam (those primitives are also used
            to build the gate's own byte-accounting), not an oversight; a
            caller composing a custom flow from them is responsible for its
            own spend discipline.

    Example:
        >>> council = Council(models=["grok", "perplexity"], synthesizer="claude")
        >>> result = council.ask_sync("What is the capital of France?")
        >>> print(result.synthesis)
    """

    def __init__(
        self,
        models: list[str],
        synthesizer: str | Sequence[str] | None = None,
        config: ConclaveConfig | None = None,
        temperature: float = 0.7,
        timeout: float = 120.0,
        cache: bool | None = None,
        extract_verdict: bool = True,
        allow_transport_debug_logging: bool = False,
        source_bundle_digest: str | None = None,
        max_output_tokens: int | None = None,
        max_spend_usd: Decimal | None = None,
    ) -> None:
        self.config = config or load_config()
        self.requested_models = list(models)
        self.synthesizer_chain = self._resolve_chain(synthesizer, self.config)
        # Back-compat: the primary candidate keeps the historic attribute.
        self.synthesizer = self.synthesizer_chain[0]
        self.temperature = temperature
        self.timeout = timeout
        # Explicit override wins; otherwise defer to config (off by default).
        self.cache_enabled = self.config.cache if cache is None else cache
        # Default-on verdict extraction (CAC-06). Named ``*_enabled`` to read
        # unambiguously as a switch, never confused with the imported
        # ``extract_verdict`` engine function. There is no per-call override --
        # this constructor flag is the single resolution path (one opt-out).
        self.extract_verdict_enabled = extract_verdict
        # Horizon-2 placeholder: callers with a grounded source bundle can bind
        # its digest into cache identity now, before source retrieval ships.
        # cache.build_identity re-hashes this value so malformed/raw input never
        # appears in inspectable identity documents or diagnostics.
        self.source_bundle_digest = source_bundle_digest
        # Default-on transport-logging guard (key-leak audit, RANK 6): drop
        # httpx/httpcore DEBUG records (the only band that emits the auth header)
        # so a process holding a real key cannot leak it via verbose transport
        # logging, even if the host enables DEBUG app-wide. Idempotent, so many
        # councils install it once; scoped to the httpx/httpcore loggers only.
        # ``allow_transport_debug_logging=True`` opts out for callers who need
        # that DEBUG band and accept the responsibility.
        if not allow_transport_debug_logging:
            transport.guard_transport_logging()
        # Explicit override wins; otherwise defer to config (off by default).
        # A cap is what makes a run's output -- and therefore its spend --
        # boundable at all; see Council.plan_calls and the --max-spend-usd gate.
        self.max_output_tokens = (
            self.config.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        # A zero/negative cap is not a ceiling at all: it either bypasses the
        # provider call entirely (max_tokens=0) or crashes downstream in
        # pricing.py's token-bound arithmetic. Reject it here so every caller
        # (library or CLI) gets the same clean failure at construction (DSE-1514
        # review, F2). The CLI additionally enforces this with `min=1` on the
        # typer option for a usage-error exit before Council is even reached.
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        # A spend cap without an output cap is not enforceable: output is the
        # unbounded term. Refuse at construction rather than at the first call,
        # so a library caller cannot get halfway into a run before finding out.
        self.max_spend_usd = max_spend_usd
        if max_spend_usd is not None:
            # Reject BEFORE any ordering comparison downstream (_reserve_plan's
            # `reserved > self.max_spend_usd`): Decimal("NaN") raises
            # InvalidOperation on any ordering comparison (an uncaught crash),
            # and Decimal("Infinity") is finite-comparison-safe but silently
            # disables the gate -- nothing can ever exceed it (DSE-1514 review,
            # F1). `is_finite()` alone rejects both NaN and +/-Infinity without
            # ever evaluating `<= 0` against a NaN, which would itself raise.
            # This guards every library caller; the CLI applies its own
            # stricter format check (rejecting non-finite spellings, signs, and
            # underscore literals) before `Decimal(...)` is ever constructed.
            if not max_spend_usd.is_finite() or max_spend_usd <= 0:
                raise ValueError("max_spend_usd must be a finite positive Decimal")
            if self.max_output_tokens is None:
                raise SpendUnboundable(_NO_OUTPUT_CAP_MESSAGE)

    @staticmethod
    def _resolve_chain(spec: str | Sequence[str] | None, config: ConclaveConfig) -> list[str]:
        """Resolve the synthesizer ladder (DSE-1512).

        Precedence, highest first: the constructor ``synthesizer=`` arg (string
        chain spec or list of names) -> ``config.synthesizer_chain`` -> a chain
        of one built from ``config.synthesizer``. This mirrors the pre-existing
        scalar-``synthesizer`` precedence documented in the class docstring, just
        extended to an ordered list -- a chain of one behaves exactly like today.

        Args:
            spec: The constructor's ``synthesizer`` argument, or ``None``.
            config: The resolved council config.

        Returns:
            A non-empty ordered list of candidate friendly names.
        """
        if isinstance(spec, str):
            chain = parse_synthesizer_chain(spec)
        elif spec is not None:
            chain = parse_synthesizer_chain(">".join(spec))
        else:
            chain = list(config.synthesizer_chain)
        return chain or [config.synthesizer]

    def _available_members(self) -> tuple[list[tuple[str, str]], list[str]]:
        """Partition requested members into (available, skipped-for-no-key).

        Returns:
            A tuple ``(members, skipped)`` where ``members`` is a list of
            ``(friendly_name, model_id)`` pairs that have a key present, and
            ``skipped`` is the list of friendly names with no key available.
        """
        members: list[tuple[str, str]] = []
        skipped: list[str] = []
        for name in self.requested_models:
            model_id = self.config.resolve_model_id(name)
            if key_present(model_id):
                members.append((name, model_id))
            else:
                logger.warning("skipping %s (%s): no API key in environment", name, model_id)
                skipped.append(name)
        return members, skipped

    def _generation_settings(self) -> dict[str, float | int]:
        """The generation settings actually used, for the manifest and receipts.

        ``max_output_tokens`` appears ONLY when a cap is configured, so an
        uncapped run's manifest is byte-identical to v1.3.0's.
        """
        settings: dict[str, float | int] = {
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if self.max_output_tokens is not None:
            settings["max_output_tokens"] = self.max_output_tokens
        return settings

    def _keyed_chain(self) -> list[tuple[str, str]]:
        """Resolve the synthesizer chain to the candidates that could be CALLED.

        :meth:`adjudicate` skips an unkeyed candidate without making a call, so
        an unkeyed candidate cannot cost anything and is excluded from the plan.
        ``registry.key_present`` returns ``True`` for an unknown provider prefix,
        which errs toward INCLUDING the call -- the safe direction for a ceiling.
        """
        pairs = [(name, self.config.resolve_model_id(name)) for name in self.synthesizer_chain]
        return [(name, model_id) for name, model_id in pairs if key_present(model_id)]

    def _plan_table(
        self,
        mode: str,
        *,
        members: list[tuple[str, str]],
        chain: list[tuple[str, str]],
        rounds: int,
        choices: list[str] | None,
    ) -> list[_PhaseSpec]:
        """Return the declarative worst-case phase table for ``mode`` (DSE-1514 review, Fix B).

        Pure data assembly: no byte arithmetic and no :class:`PlannedCall`
        construction happens here -- see :meth:`plan_calls`, which expands
        every row with the one piece of run-specific data every row shares
        (the exact prompt bytes and the output cap). That split is what lets
        all six modes share a single expansion loop instead of the two
        parallel if/elif chains this table replaces.

        With ``N`` keyed members, ``C`` keyed chain candidates, ``R`` debate
        rounds, and ``V`` = 1 when verdict extraction is on, the row counts
        reproduce exactly:

        * ``raw`` -- ``N``: one member-phase row, no chain row.
        * ``synthesize`` -- ``N + C + 2CV``: the same member-phase row, plus a
          chain synthesis row and (when verdict extraction is on) extraction
          + repair chain rows.
        * ``vote`` -- ``N``: one member-phase row; no chain row.
        * ``debate`` -- ``N*R + C``: a round-1 row plus one row per round
          2..R (each worst case at full membership; drop-out only shrinks a
          real run), plus a chain consolidation row.
        * ``adversarial`` -- ``N + C``: two member-phase rows -- one
          proposer (``upstream=0``) and ``N-1`` critics (``upstream=1``
          each, embedding the proposal) -- whose target counts always sum to
          ``N`` regardless of how many real proposer attempts fail, plus a
          chain judge row whose ``upstream=N`` (it embeds the proposal and
          every critique). This is the DSE-1514 review Fix A shape: byte
          worst case is 1 proposer succeeding immediately, maximizing the
          number of upstream-embedding critic calls.
        * ``elite`` -- ``3N + C + 2CV``: three member-phase rows (initial,
          critique, revision), plus the synthesis/verdict chain rows shared
          with ``synthesize``.

        Args:
            mode: One of ``raw``/``synthesize``/``vote``/``debate``/
                ``adversarial``/``elite``. Already validated by the caller.
            members: The keyed council members.
            chain: The keyed synthesizer-chain candidates.
            rounds: Debate rounds, already normalized to at least 1.
            choices: Vote choices, which enlarge the vote prompt template.

        Returns:
            The ordered phase rows for ``mode``.
        """
        n = len(members)
        table: list[_PhaseSpec] = []
        # Placeholder answers (real names/model ids, empty text) for measuring
        # a downstream phase's EXACT fixed per-item label overhead -- DSE-1514
        # review, Fix A. Built once per call since every downstream phase that
        # embeds the full membership needs the identical N-sized placeholder
        # list; the debate peer block additionally needs per-member letter
        # aliases, computed separately below where it is used.
        member_placeholders = _placeholder_answers(members)

        if mode in ("raw", "synthesize"):
            table.append(_PhaseSpec("member", members, "", 0))
        elif mode == "vote":
            table.append(
                _PhaseSpec(
                    "member",
                    members,
                    prompts.VOTE_SYSTEM + prompts.vote_user("", choices or []),
                    0,
                )
            )
        elif mode == "debate":
            table.append(_PhaseSpec("round-1", members, "", 0))
            # A worst-case peer block: every member's PRIOR answer anonymized
            # by letter, text left empty (bounded separately via ``upstream``)
            # so only the real, N-exact "Model X (peer) previous answer" /
            # "Your previous answer" label overhead is measured here.
            letters = {name: prompts.LETTERS[i] for i, (name, _mid) in enumerate(members)}
            prior = {
                name: answer
                for (name, _mid), answer in zip(members, member_placeholders, strict=True)
            }
            self_name = members[0][0] if members else ""
            peer_block = (
                prompts.anonymized_peer_block(self_name, letters.get(self_name, ""), prior, letters)
                if members
                else ""
            )
            for round_no in range(2, rounds + 1):
                table.append(
                    _PhaseSpec(
                        f"round-{round_no}",
                        members,
                        prompts.DEBATE_SYSTEM
                        + prompts.debate_round_user("", round_no, rounds, peer_block),
                        n,
                    )
                )
        elif mode == "adversarial":
            # Byte-worst-case (DSE-1514 review, Fix A): 1 proposer succeeds
            # immediately (upstream=0); every OTHER member critiques it, each
            # embedding the proposal's answer text (upstream=1). A real run's
            # k proposer attempts + (N-k) critics always equals N; k=1
            # maximizes the number of upstream-embedding critic calls, which
            # is the pessimistic shape a spend ceiling must plan against.
            if members:
                table.append(_PhaseSpec("proposal", members[:1], "", 0))
                table.append(
                    _PhaseSpec(
                        "critique",
                        members[1:],
                        prompts.CRITIC_SYSTEM + prompts.critic_user("", ""),
                        1,
                    )
                )
        elif mode == "elite":
            table.append(_PhaseSpec("initial", members, "", 0))
            table.append(
                _PhaseSpec(
                    "critique",
                    members,
                    prompts.ELITE_CRITIC_SYSTEM
                    + prompts.elite_critic_user("", member_placeholders),
                    n,
                )
            )
            fallback_original = ModelAnswer(
                name="", model_id="", answer="", answer_id=_ANSWER_ID_PROBE
            )
            original = member_placeholders[0] if member_placeholders else fallback_original
            # DSE-1514 review, Fix A: modes._elite_revision_messages_for passes
            # EVERY reviser its OWN initial answer as ``original_answer`` --
            # which is ALSO one of the N entries already inside the initial
            # panel. That answer's text is therefore embedded TWICE in a real
            # revision call (once standalone, once inside the anonymized
            # panel), so the byte-worst-case upstream count is N (initial
            # panel) + N (critique panel) + 1 (the duplicate), not 2N -- an
            # undercount the byte-lower-bound regression test below caught.
            table.append(
                _PhaseSpec(
                    "revision",
                    members,
                    prompts.ELITE_REVISION_SYSTEM
                    + prompts.elite_revision_user(
                        "", original, member_placeholders, member_placeholders
                    ),
                    2 * n + 1,
                )
            )

        if mode == "debate":
            # Mirrors modes._debate_synthesize's real block format exactly
            # ("### Final answer from {name} ({model_id})\n{answer}") using
            # the real member names/model ids, text left empty.
            debate_final_blocks = "\n\n".join(
                f"### Final answer from {a.name} ({a.model_id})\n{a.answer}"
                for a in member_placeholders
            )
            table.append(
                _PhaseSpec(
                    "debate_final",
                    chain,
                    prompts.DEBATE_FINAL_SYSTEM
                    + prompts.debate_final_user("", rounds, debate_final_blocks),
                    n,
                    message_count=3,
                )
            )
        elif mode == "adversarial":
            # Judge upstream is ALWAYS N: it embeds the proposal (1) plus
            # every critique the byte-worst-case shape produces (N-1).
            table.append(
                _PhaseSpec(
                    "judge",
                    chain,
                    prompts.JUDGE_SYSTEM + prompts.judge_user("", "", "", ""),
                    n,
                    message_count=3,
                )
            )
        elif mode in ("synthesize", "elite"):
            table.append(
                _PhaseSpec(
                    "synthesis",
                    chain,
                    _SYNTH_SYSTEM + _synth_user_content("", member_placeholders),
                    n,
                    message_count=3,
                )
            )
            if self.extract_verdict_enabled:
                # DSE-1514 review, Fix A: the two templates below are the REAL
                # extraction/repair message content -- schema included exactly
                # as rendered, and one placeholder label per real member --
                # measured via conclave.verdict_synthesis's own builders rather
                # than a hand-summed approximation. The schema's bytes are
                # therefore already IN the template, so ``contract=True`` here
                # only adds the flat provider-side structured-output framing
                # allowance, never a second copy of the schema on top of it.
                from .verdict_synthesis import VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES
                from .verdict_synthesis import _build_messages as _verdict_build_messages
                from .verdict_synthesis import _repair_instruction as _verdict_repair_instruction

                extraction_messages = _verdict_build_messages("", member_placeholders)
                extraction_template = "".join(m["content"] for m in extraction_messages)
                repair_template = extraction_template + _verdict_repair_instruction(
                    "x" * VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES
                )
                table.append(
                    _PhaseSpec(
                        "verdict_extraction",
                        chain,
                        extraction_template,
                        n,
                        message_count=3,
                        contract=True,
                    )
                )
                table.append(
                    _PhaseSpec(
                        "verdict_repair",
                        chain,
                        repair_template,
                        n + 1,
                        message_count=3,
                        contract=True,
                    )
                )

        return table

    def plan_calls(
        self,
        mode: str,
        prompt: str,
        *,
        rounds: int = 2,
        proposer: str | None = None,
        choices: list[str] | None = None,
    ) -> CallPlan:
        """Enumerate every provider call this mode could make, worst case (DSE-1514).

        The counts are derived from :mod:`conclave.modes` and
        :meth:`_apply_verdict`, not from a remembered formula -- see
        :meth:`_plan_table` for the exact per-mode row table and the
        DSE-1514 review Fix A rationale for the adversarial shape.

        Convergence early-stop, member drop-out, and a proposer succeeding on the
        first try all make a real run CHEAPER than its plan. A plan is never an
        under-count, which is what makes it usable as a spend gate.

        Args:
            mode: One of ``raw``/``synthesize``/``vote``/``debate``/
                ``adversarial``/``elite``.
            prompt: The exact user prompt (bounded by its UTF-8 byte length).
            rounds: Debate rounds; ignored for other modes.
            proposer: Adversarial proposer. It does not change the COUNT (see
                :meth:`_plan_table`) and is accepted only for signature parity
                with the modes.
            choices: Vote choices, which enlarge the vote prompt template.

        Returns:
            The :class:`CallPlan`.

        Raises:
            ValueError: ``mode`` is not a known deliberation mode, or
                ``max_output_tokens`` is not configured (an unbounded output
                cannot be planned).
        """
        if mode not in _VALID_PLAN_MODES:
            raise ValueError(f"unknown mode for call planning: {mode}")
        cap = self.max_output_tokens
        if cap is None:
            raise ValueError(_NO_OUTPUT_CAP_MESSAGE)

        members, _skipped = self._available_members()
        chain = self._keyed_chain()
        n_members = len(members)
        prompt_bytes = len(prompt.encode("utf-8"))

        table = self._plan_table(
            mode, members=members, chain=chain, rounds=max(1, rounds), choices=choices
        )
        calls: list[PlannedCall] = [
            spec.to_planned_call(
                name=name,
                model_id=model_id,
                prompt_token_upper_bound=prompt_bytes,
                max_output_tokens=cap,
            )
            for spec in table
            for name, model_id in spec.targets
        ]

        return CallPlan(
            mode=mode,
            calls=tuple(calls),
            member_count=n_members,
            chain_count=len(chain),
        )

    def _reserve_plan(self, plan: CallPlan) -> Decimal:
        """Price a :class:`CallPlan` pessimistically against the snapshot.

        Args:
            plan: The worst-case plan from :meth:`plan_calls`.

        Returns:
            The reserved total in USD -- an upper bound on what the run can cost.

        Raises:
            SpendUnboundable: No snapshot, or any planned call's model has no
                snapshot entry. Never falls back to a similar model's rate.
        """
        snapshot = load_default_price_snapshot()
        if snapshot is None:
            raise SpendUnboundable("cannot bound spend: price snapshot unavailable")
        total = Decimal("0")
        for call in plan.calls:
            rates = snapshot.rates_for(call.model_id)
            if rates is None:
                raise SpendUnboundable(
                    f"cannot bound spend: no priced rate for {call.model_id} "
                    f"in snapshot {snapshot.digest()} ({snapshot.captured_at.isoformat()})"
                )
            total += self._reserve_call(rates, call)
        return total

    @staticmethod
    def _reserve_call(rates: PriceRates, call: PlannedCall) -> Decimal:
        """Price one :class:`PlannedCall` against ``rates`` (DSE-1514).

        The one place the reservation formula (input bound = prompt +
        template + framing + upstream-output-as-bytes; output bound = the
        call's own cap) is written, shared by :meth:`_reserve_plan` (pricing
        an entire pre-flight :class:`CallPlan`, before any call is made) and
        :meth:`_price_manifest` (Round 4, QA C1 -- pricing a single
        already-made call's reservation when it reported no usable usage,
        from the plan row matching its phase).

        Args:
            rates: The model's exact ceiling rates.
            call: The planned (or reconstructed) call to price.

        Returns:
            The reserved cost in USD, quantized up.
        """
        return reserve_cost(
            rates,
            prompt_token_upper_bound=call.prompt_token_upper_bound,
            prompt_template_token_allowance=call.prompt_template_token_allowance,
            provider_framing_token_allowance=call.provider_framing_token_allowance,
            upstream_output_token_ceilings=(
                (call.max_output_tokens,) * call.upstream_output_call_count
            ),
            upstream_output_bytes_per_token=rates.max_output_bytes_per_token,
            max_output_tokens=call.max_output_tokens,
        ).reserved_cost_usd

    def _enforce_spend_cap(
        self,
        mode: str,
        prompt: str,
        *,
        rounds: int | None = None,
        proposer: str | None = None,
        choices: list[str] | None = None,
    ) -> None:
        """Refuse an over-budget or unboundable run BEFORE any provider call.

        A no-op when ``max_spend_usd`` is unset, so a run with no spend flags is
        byte-identical to today. Deliberately NOT applied to a cache hit: a hit
        makes no provider call and therefore cannot exceed any cap.

        Raises:
            SpendUnboundable: The plan cannot be priced.
            SpendCapExceeded: The priced plan exceeds the cap.
        """
        if self.max_spend_usd is None:
            return
        plan = self.plan_calls(
            mode,
            prompt,
            rounds=2 if rounds is None else rounds,
            proposer=proposer,
            choices=choices,
        )
        reserved = self._reserve_plan(plan)
        if reserved > self.max_spend_usd:
            raise SpendCapExceeded(reserved, self.max_spend_usd, len(plan.calls))
        logger.info(
            "spend gate: reserved %s USD for %d calls, under the %s USD cap",
            reserved,
            len(plan.calls),
            self.max_spend_usd,
        )

    def _gate_live_run(
        self,
        mode: str,
        prompt: str,
        *,
        rounds: int | None = None,
        proposer: str | None = None,
        choices: list[str] | None = None,
    ) -> None:
        """THE single pre-flight spend-gate chokepoint (DSE-1514 review, Fix C).

        :meth:`_cached_run` and :meth:`ask_stream` each call this exactly
        once, always AFTER their cache-hit decision and always BEFORE the
        first provider call -- a cache hit returns before reaching this
        method entirely, since it makes no call and cannot exceed any cap.
        Before this fix the same :meth:`_enforce_spend_cap` call was
        duplicated at one site per cache branch (four call sites total, two
        per entry point); collapsing them here means "does this run get
        gated" has exactly one answer per entry point instead of two branches
        that had to be kept in sync by hand.

        A thin, behavior-preserving wrapper over :meth:`_enforce_spend_cap`,
        which still owns the actual planning/pricing/raising.
        """
        self._enforce_spend_cap(mode, prompt, rounds=rounds, proposer=proposer, choices=choices)

    def _cache_key(
        self,
        prompt: str,
        mode: str,
        *,
        rounds: int | None = None,
        proposer: str | None = None,
        converge_threshold: float | None = None,
        choices: list[str] | None = None,
    ) -> str:
        """Build the cache key for a run from the resolved, secret-free identity.

        Uses the *resolved* member ids and the FULL resolved synthesizer/judge
        chain (DSE-1512, not just the primary) so two runs collide only when
        they would genuinely produce equivalent output: changing any successor
        candidate in the ladder invalidates a prior entry, since a later run
        over the same prompt could fail over to a different model. Members
        that would be skipped for a missing key are excluded -- a cache entry
        reflects the council that actually ran, so a key reappearing later
        produces the same membership. No environment value is read here.
        """
        members, _skipped = self._available_members()
        chain_pairs = [(c, self.config.resolve_model_id(c)) for c in self.synthesizer_chain]
        synth_id = chain_pairs[0][1]
        used_prefixes = {
            model_id.split("/", 1)[0]
            for _name, model_id in [*members, *chain_pairs]
            if "/" in model_id
        }
        snapshot = load_default_price_snapshot()
        return cache_mod.make_key(
            prompt=prompt,
            mode=mode,
            members=members,
            synthesizer=self.synthesizer,
            synthesizer_model_id=synth_id,
            synthesizer_chain=chain_pairs,
            temperature=self.temperature,
            timeout=self.timeout,
            rounds=rounds,
            proposer=proposer,
            converge_threshold=converge_threshold,
            choices=choices,
            extract_verdict=self.extract_verdict_enabled,
            endpoint_urls={
                prefix: endpoint.completions_url
                for prefix, endpoint in self.config.endpoints.items()
                if prefix in used_prefixes
            },
            source_bundle_digest=self.source_bundle_digest,
            price_snapshot_digest=None if snapshot is None else snapshot.digest(),
            max_output_tokens=self.max_output_tokens,
            protocol_version=ELITE_PROTOCOL_VERSION,
            synthesis_prompt_version=SYNTHESIS_PROMPT_VERSION,
            elite_prompt_version=ELITE_PROMPT_VERSION,
        )

    async def _cached_run(
        self,
        prompt: str,
        mode: str,
        run: Callable[[], Awaitable[CouncilResult]],
        *,
        rounds: int | None = None,
        proposer: str | None = None,
        converge_threshold: float | None = None,
        choices: list[str] | None = None,
    ) -> CouncilResult:
        """Serve ``run`` from the result cache when caching is enabled.

        On a hit the cached :class:`CouncilResult` is returned with ``cached=True``
        and the providers are not called. On a miss (or when caching is off) the
        live ``run`` executes; a successful live run is stored best-effort, EXCEPT
        for the no-store rule below. Cache read/write failures never propagate --
        they degrade to a normal live run.

        This is the single chokepoint every mode funnels through, so it is also
        where the manifest-on-every-result invariant is enforced: each returned
        result passes through :meth:`_ensure_manifest` (a no-op for the
        synthesize/raw path, which builds its own richer manifest in
        :meth:`_ask_uncached`; a fill for ``debate``/``adversarial``/``vote`` and
        for a cache hit stored before the manifest existed).

        **No-store when the primary did not adjudicate (DSE-1512).** A cache
        hit must never pin a result the chain's primary adjudicator did not
        produce, and it must never replay an infrastructure outage -- including
        a still-missing key -- after it has cleared. So a live result is NOT
        written to the cache when
        :attr:`~conclave.models.CouncilResult.primary_failed_over` is ``True``.
        See that property for the exact rule; in one sentence: the primary
        adjudicator of some role did not itself produce the answer, for an
        infrastructure reason (no key, auth, quota, 5xx, timeout, network) or
        because the whole ladder was exhausted. The uncached result is still
        returned to THIS caller unchanged; only the write to disk is skipped,
        so the next identical ``ask`` gets a fresh chance at a healthy primary
        rather than a pinned failure or a successor's answer served under the
        primary's name. This is a deliberate, narrow behavior change from
        v1.3.0: a chain-of-one run whose sole synthesizer had no key or
        errored for an infrastructure reason used to be cached and now is not.
        A ``"terminal_failure"`` ledger entry (the model answered, just not
        usably) is unaffected and remains cacheable exactly as before --
        re-running would not produce a different, better answer.

        **One gate chokepoint (DSE-1514 review, Fix C).** :meth:`_gate_live_run`
        is called exactly once here, after the cache-hit decision above (a hit
        returns before reaching it) and before ``run()`` -- regardless of
        whether caching is enabled at all. This replaced two separate call
        sites (one per cache branch) that had to make the identical call.
        """
        key: str | None = None
        if self.cache_enabled:
            key = self._cache_key(
                prompt,
                mode,
                rounds=rounds,
                proposer=proposer,
                converge_threshold=converge_threshold,
                choices=choices,
            )
            hit = cache_mod.load(key)
            if hit is not None:
                logger.info("cache hit for %s run (%s)", mode, key[:12])
                self._ensure_manifest(hit, mode)
                self._price_manifest(hit)
                return hit

        self._gate_live_run(mode, prompt, rounds=rounds, proposer=proposer, choices=choices)
        result = await run()
        self._ensure_manifest(result, mode)
        self._price_manifest(result)
        if not self.cache_enabled:
            return result
        if result.primary_failed_over:
            logger.info(
                "not caching %s run (%s): primary adjudicator failed for an infrastructure reason",
                mode,
                key[:12],
            )
            return result
        cache_mod.store(key, result)
        return result

    def _ensure_manifest(self, result: CouncilResult, mode: str) -> None:
        """Guarantee the manifest-on-every-result invariant at the single chokepoint.

        Every mode funnels through :meth:`_cached_run`, so attaching the auditable
        :class:`ModelHarnessManifest` here makes it a true invariant rather than a
        per-mode responsibility that can silently drift (the drift this method
        fixes: ``debate``/``adversarial``/``vote`` used to return with
        ``manifest is None``).

        A no-op when ``result.manifest`` already exists: the synthesize/raw path
        builds its own manifest inside :meth:`_ask_uncached` so it can populate the
        verdict-provenance slots and re-stamp secret-safety over them, and this
        method must not overwrite that richer manifest. The
        ``debate``/``adversarial``/``vote`` wrappers return a result with
        ``manifest is None`` -- this fills it from the resolved membership and the
        collected answers, including the zero-members early-return path and a stale
        cache hit stored before this invariant existed, so no result ever escapes
        without an audit manifest.

        Membership is re-resolved via :meth:`_available_members` (the same
        resolution :meth:`_cache_key` already performs) rather than threaded back
        through every mode's return value; this keeps ``providers_called`` /
        ``model_ids`` reflecting the full resolved membership even for debate rounds
        where a member later dropped out. For ``debate`` with ``result.rounds``
        populated the per-answer receipts are built from EVERY round (see
        :meth:`_build_debate_manifest`, DSE-1514 QA C2); every other mode builds
        them from ``result.answers`` alone. :meth:`_build_manifest` stamps
        ``secret_safety`` VERIFIED when the assembled manifest is provably clean.

        Args:
            result: The result to attach a manifest to. Mutated in place.
            mode: The deliberation mode string recorded on the manifest.
        """
        if result.manifest is not None:
            return
        members, skipped = self._available_members()
        if mode == "elite" and result.elite is not None:
            result.manifest = self._build_elite_manifest(
                members=members,
                skipped=skipped,
                result=result,
            )
        elif mode == "debate" and result.rounds:
            result.manifest = self._build_debate_manifest(
                members=members,
                skipped=skipped,
                result=result,
            )
        else:
            result.manifest = self._build_manifest(
                mode=mode, members=members, skipped=skipped, answers=result.answers
            )

    async def fan_out(
        self,
        members: list[tuple[str, str]],
        messages_for: MessagesFor,
    ) -> list[ModelAnswer]:
        """Fan a per-member message list out concurrently and collect results.

        This is the single concurrency primitive reused by every mode (synthesize,
        raw, debate, adversarial). It never raises for a member failure: each
        member yields a :class:`ModelAnswer` carrying either an answer or an error.

        Args:
            members: ``(friendly_name, model_id)`` pairs to call.
            messages_for: Callable mapping a ``(name, model_id)`` member to the
                OpenAI-style message list to send it. Lets each mode tailor the
                prompt per member (e.g. inject peers' prior-round answers) while
                sharing the gather/partial-failure logic.

        Returns:
            One :class:`ModelAnswer` per member, in the same order as ``members``.
        """
        tasks = [
            call_model(
                name,
                model_id,
                messages_for(name, model_id),
                config=self.config,
                temperature=self.temperature,
                timeout=self.timeout,
                max_output_tokens=self.max_output_tokens,
            )
            for name, model_id in members
        ]
        # return_exceptions=True is belt-and-suspenders; call_model already
        # converts provider failures into ModelAnswer.error, but this guards
        # against any unexpected raise so one bad member can't abort the gather.
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        answers: list[ModelAnswer] = []
        for (name, model_id), outcome in zip(members, gathered, strict=True):
            if isinstance(outcome, ModelAnswer):
                answers.append(outcome)
            else:
                # call_model already redacts and never raises, so this arm only
                # fires on an UNEXPECTED escape. Redact the exception text anyway:
                # the invariant "every error string conclave surfaces is scrubbed"
                # must hold even on this defense-in-depth path (key-leak audit).
                message = redact(f"{type(outcome).__name__}: {outcome}")
                logger.warning("%s raised unexpectedly: %s", name, message)
                answers.append(ModelAnswer(name=name, model_id=model_id, error=message))
        return answers

    async def ask(self, prompt: str, synthesize: bool = True) -> CouncilResult:
        """Run the council asynchronously.

        When the result cache is enabled, an identical prior run is returned from
        cache (``CouncilResult.cached is True``) without calling the providers.

        Args:
            prompt: The user prompt to fan out.
            synthesize: When True (default), merge answers via the synthesizer.

        Returns:
            A :class:`CouncilResult` with per-member answers and (optionally) the
            synthesis. A run with zero available members returns an empty-answer
            result rather than raising.
        """
        mode = "synthesize" if synthesize else "raw"
        return await self._cached_run(
            prompt, mode, lambda: self._ask_uncached(prompt, synthesize=synthesize)
        )

    def _build_manifest(
        self,
        *,
        mode: str,
        members: list[tuple[str, str]],
        skipped: list[str],
        answers: list[ModelAnswer],
    ) -> ModelHarnessManifest:
        """Assemble the auditable :class:`ModelHarnessManifest` for a run (CAC-04).

        Builds the manifest from the resolved membership plus the collected
        member ``answers`` (one execution receipt per answer via
        :func:`conclave.providers.receipt_from_answer`). It works for both the
        normal path (``answers`` populated) and the empty-members path
        (``members``/``answers`` empty, ``skipped`` listing every requested name)
        so every live run returns a manifest. It is called for the synthesize/raw
        path directly in :meth:`_ask_uncached` and for ``debate``/``adversarial``/
        ``vote`` via :meth:`_ensure_manifest`, so the ``mode`` argument spans every
        deliberation mode.

        The ``conclave_version`` is read via a deferred import: ``conclave``
        imports this module at package init *before* it assigns ``__version__``,
        so a top-level import would resolve too early. Deferring it into this
        method (run only when a result is produced, by which point the package is
        fully initialized) mirrors the ``models._default_prompt_version`` factory.

        After assembly the manifest is scanned for secret material and its
        ``secret_safety`` stamped VERIFIED only when provably clean
        (:func:`conclave.manifest.verified_secret_safety`).

        Args:
            mode: Deliberation mode (``"synthesize"``/``"raw"``/``"debate"``/
                ``"adversarial"``/``"vote"``).
            members: ``(friendly_name, model_id)`` pairs that were called.
            skipped: Friendly names skipped for a missing key.
            answers: The collected per-member answers (empty on the no-members path).

        Returns:
            A fully-assembled, secret-safety-stamped manifest.
        """
        from . import __version__

        receipts = [
            receipt_from_answer(
                a,
                temperature=self.temperature,
                timeout=self.timeout,
                max_output_tokens=self.max_output_tokens,
            )
            for a in answers
        ]
        manifest = ModelHarnessManifest(
            request_id=uuid4().hex,
            conclave_version=__version__,
            mode=mode,
            providers_considered=list(self.requested_models),
            providers_called=[name for name, _model_id in members],
            providers_skipped=[
                ProviderSkip(name=name, reason="no API key in environment") for name in skipped
            ],
            model_ids=[model_id for _name, model_id in members],
            generation_settings=self._generation_settings(),
            receipts=receipts,
        )
        self._recompute_manifest_accounting(manifest)
        return manifest

    def _build_debate_manifest(
        self,
        *,
        members: list[tuple[str, str]],
        skipped: list[str],
        result: CouncilResult,
    ) -> ModelHarnessManifest:
        """Assemble a debate manifest with one receipt per answer per ROUND (DSE-1514 QA C2).

        :meth:`_build_manifest` builds receipts only from ``result.answers``,
        which :func:`conclave.modes.run_debate` mirrors from the FINAL round
        only. For a multi-round debate that silently under-counts every call:
        rounds ``1..R-1`` and any member that dropped out mid-debate never get
        a receipt, so the manifest -- and therefore
        :meth:`_price_manifest`'s run-level cost ceiling, which sums exactly
        the receipts present -- covers a strict subset of the calls a real
        debate makes (4 of 7 at 3 members / 2 rounds, 4 of 10 at 3 members / 3
        rounds) while still reporting a complete, non-``None`` ceiling.

        This builds one receipt per answer for EVERY round in
        ``result.rounds``, in round order, phase-stamped
        ``f"round-{round_number}"`` -- the exact same ``"round-N"`` string
        :meth:`_plan_table` gives its debate-round plan rows (DSE-1514 Round
        4), so a usage-less receipt's phase maps onto its plan row by plain
        equality, with no pattern-parsing needed. The ``debate_final``
        adjudication receipt is appended AFTERWARDS by
        :func:`conclave.modes._debate_synthesize` via
        :meth:`_adjudicate_and_record`, so it is deliberately not built here
        -- :func:`conclave.modes.run_debate` calls this (via
        :meth:`_ensure_manifest`) BEFORE that consolidation step runs.

        ``providers_called``/``model_ids`` still reflect the full resolved
        membership (unchanged from :meth:`_build_manifest`), not just the
        per-round callees, so a member that drops out mid-debate stays
        visible as "called" even though a later round has no receipt for it.

        Args:
            members: ``(friendly_name, model_id)`` pairs resolvable (keyed)
                for this run.
            skipped: Friendly names skipped for a missing key.
            result: The in-flight debate result. ``result.rounds`` is already
                fully populated -- :func:`conclave.modes.run_debate` calls
                this only after its round loop completes.

        Returns:
            A fully-assembled, secret-safety-stamped manifest whose receipts
            cover every round's calls, in round order.
        """
        from . import __version__

        receipts = [
            receipt_from_answer(
                answer,
                temperature=self.temperature,
                timeout=self.timeout,
                phase=f"round-{debate_round.round_number}",
            )
            for debate_round in result.rounds
            for answer in debate_round.answers
        ]
        manifest = ModelHarnessManifest(
            request_id=uuid4().hex,
            conclave_version=__version__,
            mode="debate",
            providers_considered=list(self.requested_models),
            providers_called=[name for name, _model_id in members],
            providers_skipped=[
                ProviderSkip(name=name, reason="no API key in environment") for name in skipped
            ],
            model_ids=[model_id for _name, model_id in members],
            generation_settings={"temperature": self.temperature, "timeout": self.timeout},
            receipts=receipts,
        )
        self._recompute_manifest_accounting(manifest)
        return manifest

    def _build_elite_manifest(
        self,
        *,
        members: list[tuple[str, str]],
        skipped: list[str],
        result: CouncilResult,
    ) -> ModelHarnessManifest:
        """Assemble a phase-complete manifest for an elite protocol run.

        Elite keeps every attempted member call in phase-specific artifact
        collections. Flattening those collections here preserves repeated calls
        as distinct receipts while provider membership remains unique. Failed
        gates therefore retain the completed and attempted phases without
        inventing receipts for phases that never ran.
        """
        from . import __version__

        elite = result.elite
        phase_artifacts = (
            []
            if elite is None
            else [
                ("initial", elite.initial_answers),
                ("critique", elite.critiques),
                ("revision", elite.revisions),
            ]
        )
        receipts = [
            receipt_from_answer(
                answer,
                temperature=self.temperature,
                timeout=self.timeout,
                phase=phase,
                protocol_version=ELITE_PROTOCOL_VERSION,
                prompt_version=None if phase == "initial" else ELITE_PROMPT_VERSION,
                max_output_tokens=self.max_output_tokens,
            )
            for phase, answers in phase_artifacts
            for answer in answers
        ]
        manifest = ModelHarnessManifest(
            request_id=uuid4().hex,
            conclave_version=__version__,
            mode="elite",
            providers_considered=list(self.requested_models),
            providers_called=list(dict.fromkeys(name for name, _model_id in members)),
            providers_skipped=[
                ProviderSkip(name=name, reason="no API key in environment") for name in skipped
            ],
            model_ids=list(dict.fromkeys(model_id for _name, model_id in members)),
            generation_settings=self._generation_settings(),
            receipts=receipts,
        )
        self._recompute_manifest_accounting(manifest)
        return manifest

    @staticmethod
    def _recompute_manifest_accounting(manifest: ModelHarnessManifest) -> None:
        """Recompute every manifest aggregate from its complete receipt ledger."""
        manifest.total_latency_ms = sum(receipt.latency_ms for receipt in manifest.receipts)
        usages = [receipt.usage for receipt in manifest.receipts if receipt.usage is not None]
        manifest.total_usage = (
            TokenUsage(
                prompt_tokens=sum(usage.prompt_tokens for usage in usages),
                completion_tokens=sum(usage.completion_tokens for usage in usages),
                total_tokens=sum(usage.total_tokens for usage in usages),
            )
            if usages
            else None
        )
        # Cost is known only when every actual call carries a trustworthy priced
        # value. A partially-known total would be misleading, so unknown remains
        # None rather than being silently coerced to zero.
        costs = [receipt.estimated_cost for receipt in manifest.receipts]
        manifest.estimated_cost = (
            sum(cost for cost in costs if cost is not None)
            if costs and all(cost is not None for cost in costs)
            else None
        )
        manifest.redacted_errors = [
            receipt.error for receipt in manifest.receipts if receipt.error is not None
        ]
        manifest.secret_safety = verified_secret_safety(manifest)

    def _price_manifest(self, result: CouncilResult) -> None:
        """Stamp cost ceilings on the manifest -- the LAST step of a run (DSE-1514).

        Runs after ``_ensure_manifest`` and after every receipt is appended, so
        it sees the complete ledger. It is idempotent: it recomputes every field
        from the receipts each time, so re-pricing a cache hit is harmless.

        The rule, per receipt:

        * the model has no snapshot entry -> unpriced (``None``/``None``), and
          the model id joins ``unpriced_models``;
        * the provider reported a trustworthy, non-zero usage figure
          (:func:`_usage_is_reported`) -> ``reported_usage_cost`` at ceiling
          rates, basis ``"reported_usage"``;
        * usage is not reported (the call FAILED; the call SUCCEEDED but the
          provider omitted the usage field -- :mod:`conclave.adapters.
          openai_compat` returns ``usage=None`` in that case, including for
          some streamed responses, a normal outcome on a clean call, not a
          failure signal; or the provider reported an all-zero ``TokenUsage``,
          QA I1 -- a technically-present but empty usage is the same "nothing
          to price" shape as ``None``, and pricing it at ``$0.000000`` would
          assert a false floor rather than an honest bound) AND an output cap
          is configured AND the receipt's ``phase`` maps onto a row of
          :meth:`_plan_table` for this run's mode (:meth:`_reservation_row_for_phase`,
          DSE-1514 Round 4, QA C1) -> that row's own pessimistic reservation --
          its REAL template bytes and upstream-embedding count, never a flat
          constant that cannot tell a bare member call from a synthesis call
          embedding N upstream answers -- basis ``"reservation"``;
        * no usage, no cap, or the phase has no plan row (currently: an
          adversarial proposal/critique receipt, whose ``phase`` is ``None``
          on both shapes and so cannot be told apart post hoc) -> unpriced.
          Nothing is ever estimated from a guess.

        And at run level, ALL-OR-NOTHING: ``cost_ceiling_usd`` is the sum of
        every receipt ceiling only when ``unpriced_models`` is empty AND
        ``unpriced_receipts`` is zero. A run with no receipts at all (the
        memberless path) has a provable ceiling of ``Decimal("0")`` -- no call
        was made, so nothing was spent -- deliberately distinguished from
        ``None`` ("could not be bounded").

        Never raises: an unusable snapshot, an unreadable file, or a provider
        that reports an impossible token total degrades to "unpriced" with a
        bounded warning. A missing ceiling is an acceptable outcome; a failed
        council run because of pricing is not.
        """
        manifest = result.manifest
        if manifest is None:
            return

        snapshot: PriceSnapshot | None = load_default_price_snapshot()
        if snapshot is None:
            for receipt in manifest.receipts:
                receipt.cost_ceiling_usd = None
                receipt.cost_basis = None
            manifest.cost_ceiling_usd = None
            manifest.price_snapshot_digest = None
            manifest.priced_as_of = None
            manifest.unpriced_models = []
            manifest.unpriced_receipts = len(manifest.receipts)
            manifest.pricing_warnings = ["price_snapshot_unavailable"]
            manifest.secret_safety = verified_secret_safety(manifest)
            return

        cap = self.max_output_tokens
        unpriced_models: set[str] = {
            model_id for model_id in manifest.model_ids if snapshot.rates_for(model_id) is None
        }
        unpriced_receipts = 0
        for receipt in manifest.receipts:
            rates = snapshot.rates_for(receipt.model_id)
            if rates is None:
                unpriced_models.add(receipt.model_id)
                receipt.cost_ceiling_usd = None
                receipt.cost_basis = None
                unpriced_receipts += 1
                continue
            if _usage_is_reported(receipt.usage):
                try:
                    receipt.cost_ceiling_usd = reported_usage_cost(
                        rates,
                        prompt_tokens=receipt.usage.prompt_tokens,
                        completion_tokens=receipt.usage.completion_tokens,
                        total_tokens=receipt.usage.total_tokens,
                    )
                except ValueError:
                    # A provider reported a total below its own attributed usage.
                    # Bounding it would require inventing the missing tokens.
                    logger.warning(
                        "unusable reported usage for %s; leaving the call unpriced",
                        receipt.model_id,
                    )
                    receipt.cost_ceiling_usd = None
                    receipt.cost_basis = None
                    unpriced_receipts += 1
                    continue
                receipt.cost_basis = "reported_usage"
                continue
            # No trustworthy usage: failed, successful-but-unreported, or
            # all-zero. Reserve from the plan row matching this receipt's
            # phase when a cap exists and one matches (DSE-1514 Round 4, QA
            # C1); otherwise unpriced -- see the docstring, never estimated.
            row = None if cap is None else self._reservation_row_for_phase(receipt.phase, result)
            if row is None:
                receipt.cost_ceiling_usd = None
                receipt.cost_basis = None
                unpriced_receipts += 1
                continue
            planned = row.to_planned_call(
                name=receipt.name,
                model_id=receipt.model_id,
                prompt_token_upper_bound=len(result.prompt.encode("utf-8")),
                max_output_tokens=cap,
            )
            receipt.cost_ceiling_usd = self._reserve_call(rates, planned)
            receipt.cost_basis = "reservation"

        warnings: list[str] = []
        if unpriced_models:
            warnings.append("unpriced_models_present")
        if unpriced_receipts:
            warnings.append("unpriced_receipts_present")
            if cap is None:
                warnings.append("no_output_cap_configured")
        if snapshot.is_stale():
            warnings.append("price_snapshot_stale")

        manifest.price_snapshot_digest = snapshot.digest()
        manifest.priced_as_of = snapshot.captured_at.isoformat()
        manifest.unpriced_models = sorted(unpriced_models)
        manifest.unpriced_receipts = unpriced_receipts
        manifest.pricing_warnings = warnings
        manifest.cost_ceiling_usd = (
            sum((receipt.cost_ceiling_usd for receipt in manifest.receipts), Decimal("0"))
            if not unpriced_models and not unpriced_receipts
            else None
        )
        # Re-stamp: the ceiling fields were written after _build_manifest's scan.
        manifest.secret_safety = verified_secret_safety(manifest)

    def _reservation_row_for_phase(
        self, phase: str | None, result: CouncilResult
    ) -> _PhaseSpec | None:
        """Map a usage-less receipt's ``phase`` onto its :meth:`_plan_table` row (DSE-1514 Round 4, QA C1).

        Rebuilds the SAME declarative phase table :meth:`plan_calls` would
        enumerate for ``result.mode`` (down to the real vote choices and the
        real round count, so the reconstructed row's template bytes match the
        real call as closely as a post-hoc reconstruction can) and returns
        the one row whose ``phase`` matches. The mapping is exact-string
        equality, never a guess:

        * ``None`` (an untagged member-shaped call -- raw/synthesize/vote
          members, or an adversarial proposal/critique, which are NOT
          distinguishable post hoc since both share ``phase=None``) is
          looked up as ``"member"``: matches raw/synthesize/vote's single
          member row; adversarial has no row named ``"member"`` (its
          member-shaped calls split into ``"proposal"``/``"critique"`` rows a
          receipt's phase cannot currently pick between), so an adversarial
          proposal/critique receipt deliberately finds no row here and stays
          unpriced -- never silently priced as the wrong shape;
        * ``"initial"``/``"critique"``/``"revision"`` (elite) match the
          identically named elite row;
        * ``"synthesis"``/``"judge"``/``"debate_final"`` match the
          identically named adjudication row;
        * ``"verdict_extraction"``/``"verdict_repair"`` match the identically
          named verdict row;
        * ``"round-{n}"`` (debate) rebuilds the table with ``rounds=n`` (or
          the number of rounds actually run, whichever is larger) so the
          row for round ``n`` specifically exists, with its real ``upstream``
          count (0 for round 1, every member for round 2+).

        Args:
            phase: The receipt's ``phase`` (``None`` for an untagged call).
            result: The in-flight result -- read for ``mode`` (which table to
                build), ``rounds`` (debate's real round count), and
                ``vote.choices`` (the real vote template).

        Returns:
            The matching :class:`_PhaseSpec`, or ``None`` when this mode has
            no row for ``phase`` -- the caller leaves the receipt unpriced.
        """
        key = phase or "member"
        members, _skipped = self._available_members()
        chain = self._keyed_chain()

        if key.startswith("round-"):
            try:
                round_no = int(key.removeprefix("round-"))
            except ValueError:
                return None
            rounds = max(round_no, len(result.rounds))
            table = self._plan_table(
                "debate", members=members, chain=chain, rounds=rounds, choices=None
            )
        elif result.mode not in _VALID_PLAN_MODES:
            return None
        else:
            choices = result.vote.choices if result.vote is not None else None
            rounds = max(1, len(result.rounds))
            table = self._plan_table(
                result.mode, members=members, chain=chain, rounds=rounds, choices=choices
            )

        return next((spec for spec in table if spec.phase == key), None)

    def _append_manifest_receipts(
        self,
        result: CouncilResult,
        receipts: list[ProviderExecutionReceipt],
    ) -> None:
        """Append explicit call receipts and refresh all derived manifest fields."""
        if result.manifest is None or not receipts:
            return
        result.manifest.receipts.extend(receipts)
        self._recompute_manifest_accounting(result.manifest)

    async def _ask_uncached(self, prompt: str, synthesize: bool = True) -> CouncilResult:
        """The live ask path (no cache consultation). See :meth:`ask`.

        A :class:`ModelHarnessManifest` is attached on **every** return, including
        the zero-members early return, so a consumer can always audit what ran
        (CAC-04). The empty-members manifest carries no receipts, the full skip
        list, and a VERIFIED ``secret_safety`` stamp.
        """
        mode = "synthesize" if synthesize else "raw"
        members, skipped = self._available_members()
        result = CouncilResult(prompt=prompt, mode=mode, skipped=skipped)

        if not members:
            logger.warning("no council members have keys available; nothing to run")
            result.manifest = self._build_manifest(
                mode=mode, members=members, skipped=skipped, answers=[]
            )
            return result

        base_messages = [{"role": "user", "content": prompt}]
        result.answers = await self.fan_out(members, lambda _name, _model_id: base_messages)
        result.manifest = self._build_manifest(
            mode=mode, members=members, skipped=skipped, answers=result.answers
        )

        if synthesize:
            # Prose synthesis first, then the structured verdict over the SAME
            # answers. ``_synthesize`` records its own succession ledger + one
            # receipt per real call via ``_record_adjudication`` (DSE-1512), so
            # no manual receipt append is needed here. ``_apply_verdict`` runs
            # after the manifest exists so it can populate the manifest's
            # verdict-provenance slots; it is skipped in raw mode (no synthesizer
            # call) and is opt-out via the constructor flag (resolved inside the
            # helper). The no-members early return above never reaches here, so
            # a memberless run carries no verdict.
            await self._synthesize(result)
            await self._apply_verdict(result)
        return result

    async def ask_stream(self, prompt: str, synthesize: bool = True) -> AsyncIterator[StreamEvent]:
        """Stream a synthesize/raw run, yielding incremental :class:`StreamEvent`s.

        The streaming counterpart of :meth:`ask` (issue #7). Members are fanned
        out concurrently and their tokens are interleaved as ``member_delta`` /
        ``member_done`` events; when ``synthesize`` is ``True`` the synthesizer's
        tokens follow as ``synthesis_delta`` / ``synthesis_done``; a terminal
        ``done`` event carries the fully-assembled :class:`CouncilResult`, whose
        shape matches the non-streaming path exactly. Streaming applies to the
        synthesize/raw path only -- ``debate``/``adversarial`` are not streamed.

        **Cache interaction.** When the result cache is enabled and an identical
        prior run is cached, there are no live provider tokens to stream: the
        cached final text is rendered in **one shot** -- a single
        ``member_delta`` per member (and a single ``synthesis_delta`` if a
        synthesis was cached) followed by the matching ``*_done`` events and the
        terminal ``done`` (with ``result.cached is True``). The providers are not
        called. On a cache **miss**, the live stream runs and, on completion, the
        assembled result is stored so a later ``--stream`` or buffered run hits
        -- UNLESS :attr:`~conclave.models.CouncilResult.primary_failed_over` is
        ``True``: the primary adjudicator of some role did not itself produce
        the answer (see that property for the exact rule, including a missing
        key), mirroring :meth:`_cached_run`'s no-store rule exactly -- a cache
        hit must never pin a result the primary did not produce, nor replay an
        infrastructure outage after it has cleared. The run is still returned
        to this caller unchanged; only the write to disk is skipped.

        Args:
            prompt: The user prompt to fan out.
            synthesize: When ``True`` (default), stream the synthesizer too.

        Yields:
            :class:`StreamEvent` objects; the last one is always ``type="done"``.
        """
        from .streaming import stream_ask

        mode = "synthesize" if synthesize else "raw"

        # One gate chokepoint (DSE-1514 review, Fix C): a cache hit returns
        # below before ``_gate_live_run`` is ever reached, since it makes no
        # provider call and cannot exceed any cap. Everything past the hit
        # check is a live run, so the gate call sits exactly once, right
        # before the streaming driver starts -- previously duplicated at one
        # site per cache branch.
        key: str | None = None
        if self.cache_enabled:
            key = self._cache_key(prompt, mode)
            hit = cache_mod.load(key)
            if hit is not None:
                logger.info("cache hit for %s stream (%s)", mode, key[:12])
                # Re-price on every hit, exactly like _cached_run (DSE-1514
                # review, F3): the manifest was priced at STORE time, so a
                # replay without this call would report a stale
                # `priced_as_of` / `price_snapshot_stale` verdict rather than
                # today's -- a wrong number in a receipt.
                self._price_manifest(hit)
                for event in self._replay_cached(hit):
                    yield event
                return

        self._gate_live_run(mode, prompt)

        if not self.cache_enabled:
            async for event in stream_ask(self, prompt, synthesize=synthesize):
                yield event
            return

        # Live miss: stream, capture the terminal result, then store it
        # (no-store on primary infrastructure failure -- see the docstring).
        final: CouncilResult | None = None
        async for event in stream_ask(self, prompt, synthesize=synthesize):
            if event.type == "done" and event.result is not None:
                final = event.result
            yield event
        if final is not None:
            if final.primary_failed_over:
                logger.info(
                    "not caching %s run (%s): primary adjudicator failed for an "
                    "infrastructure reason",
                    mode,
                    key[:12],
                )
            else:
                cache_mod.store(key, final)

    @staticmethod
    def _replay_cached(result: CouncilResult) -> list[StreamEvent]:
        """Render a cached :class:`CouncilResult` as one-shot stream events.

        A cache hit has no live tokens, so each member's full cached answer is
        emitted as a single ``member_delta`` + ``member_done`` (errors emit only
        ``member_done``), the cached synthesis as a single ``synthesis_delta`` +
        ``synthesis_done``, and finally the terminal ``done`` carrying the cached
        result verbatim (``cached is True``). This keeps the streaming consumer's
        event contract intact without fabricating a fake token-by-token stream.
        """
        events: list[StreamEvent] = []
        for ans in result.answers:
            if ans.answer:
                events.append(
                    StreamEvent(
                        type="member_delta",
                        name=ans.name,
                        model_id=ans.model_id,
                        text=ans.answer,
                    )
                )
            events.append(
                StreamEvent(type="member_done", name=ans.name, model_id=ans.model_id, answer=ans)
            )
        if result.synthesis is not None:
            events.append(
                StreamEvent(
                    type="synthesis_delta",
                    name=result.synthesizer,
                    model_id=result.synthesizer_model_id,
                    text=result.synthesis,
                )
            )
            events.append(
                StreamEvent(
                    type="synthesis_done",
                    name=result.synthesizer,
                    model_id=result.synthesizer_model_id,
                    answer=ModelAnswer(
                        name=result.synthesizer or "synthesizer",
                        model_id=result.synthesizer_model_id or "",
                        answer=result.synthesis,
                    ),
                )
            )
        events.append(StreamEvent(type="done", result=result))
        return events

    async def _synthesize(self, result: CouncilResult) -> ModelAnswer | None:
        """Run the synthesizer chain over the successful answers, mutating ``result``.

        This is the buffered (non-streaming) synthesize path; the streaming
        counterpart :func:`conclave.streaming._stream_synthesis` mirrors it
        short-circuit for short-circuit (unchanged by DSE-1512 -- streaming keeps
        the single-candidate synthesizer path). The synthesizer identity is
        ``self.synthesizer`` (the chain's primary, resolved per the precedence
        documented in the module docstring); the full ordered ladder tried is
        ``self.synthesizer_chain`` (:meth:`adjudicate` walks it).

        Every degraded outcome is made observable on ``result`` -- none is
        silent. On success ``result.synthesis`` holds the merged answer; on any
        of the three short-circuits ``result.synthesis`` stays ``None`` and
        ``result.synthesis_error`` carries the reason:

        * **no usable answers** -- every member failed/was skipped, so there is
          nothing to merge (no adjudication attempt is made; the ledger stays
          untouched for this run);
        * **no chain candidate keyed** -- every candidate in
          ``self.synthesizer_chain`` has no API key, so the raw member answers
          are returned with an explanatory error (a chain of one keeps the
          historic single-model wording byte-for-byte); the ledger records one
          ``skipped_unkeyed`` attempt per candidate via
          :meth:`_skipped_attempts`;
        * **chain exhausted or terminal** -- every keyed candidate failed
          (:meth:`adjudicate`'s succession rule), or the answering candidate's
          failure is terminal for the role; the error text of the resolved
          answer is surfaced verbatim.

        The synthesizer identity (``synthesizer`` / ``synthesizer_model_id``) is
        recorded on ``result`` before the key check (as the chain's primary) and
        again after adjudication (as whichever candidate actually answered), so
        a consumer can see which model was selected even when it could not run,
        and which one actually produced the synthesis when the chain failed
        over. The prompt used is the versioned :data:`_SYNTH_SYSTEM`; the
        version tag already lives on ``result.prompt_version``. Every real call
        this method makes is recorded via :meth:`_record_adjudication`, which
        appends the succession ledger and one execution receipt per call.
        """
        usable = result.successful_answers
        if not usable:
            result.synthesis_error = "no successful member answers to synthesize"
            logger.warning(result.synthesis_error)
            return None

        primary_id = self.config.resolve_model_id(self.synthesizer)
        result.synthesizer = self.synthesizer
        result.synthesizer_model_id = primary_id

        err = self._chain_unkeyed_error("synthesizer", "returning raw answers only")
        if err is not None:
            result.synthesis_error = err
            logger.warning(err)
            self._record_adjudication(
                result, self._skipped_attempts("synthesis"), [], phase="synthesis"
            )
            return None

        user_content = _synth_user_content(result.prompt, usable)
        outcome = await self._adjudicate_and_record(
            result,
            "synthesis",
            _SYNTH_SYSTEM,
            user_content,
            phase="synthesis",
            protocol_version=(ELITE_PROTOCOL_VERSION if result.mode == "elite" else None),
        )
        answer = outcome.answer
        if answer is not None:
            result.synthesizer, result.synthesizer_model_id = outcome.name, outcome.model_id
            if answer.ok:
                result.synthesis = answer.answer
            else:
                result.synthesis_error = answer.error
        return answer

    async def _apply_verdict(
        self,
        result: CouncilResult,
        *,
        record_receipts: bool = True,
    ) -> None:
        """Run verdict extraction over the answers and hoist it onto ``result``.

        The SINGLE shared verdict-resolution path. Both the buffered
        ``ask``/:meth:`_ask_uncached` path (here) and the streaming path
        (CAC-06-STREAM) call this one method, so the rule "verdict object is
        canonical, top-level fields are mirrors" is written exactly once and the
        two paths cannot drift. It mutates ``result`` in place and returns
        ``None``.

        The verdict object (``result.verdict``) is the canonical adjudication; the
        convenience fields (``consensus_score``/``method``/``label``,
        ``conflicts``, ``provider_votes``, ``minority_reports``) are HOISTED
        mirrors of the same values for callers that don't want to reach through
        ``result.verdict``. They are populated only when a verdict is present;
        when it is absent they stay at their ``None``/empty defaults.

        Consensus is NEVER recomputed here: it is carried verbatim from
        :func:`conclave.verdict_synthesis.extract_verdict`, which computes it
        deterministically from the model's clustering (DD-1). This method only
        delegates and copies fields.

        **Opt-out & cost.** When ``self.extract_verdict_enabled`` is ``False`` this
        is a no-op and every verdict field is left at its default. When enabled it
        makes a SECOND synthesizer call (the extraction round-trip, plus one repair
        retry on a malformed response) distinct from the prose synthesis call --
        the documented cost of the default-on verdict.

        **Chain walk (DSE-1512).** Unlike every other adjudication role, this one
        cannot simply call :meth:`adjudicate`: verdict extraction is not a single
        call -- :func:`conclave.verdict_synthesis.extract_verdict` makes the
        initial structured call PLUS one same-model repair retry, validates the
        JSON, and computes consensus, all in one invocation. So the chain walk for
        this role lives here: ``extract_verdict`` is called once per
        ``self.synthesizer_chain`` candidate, and the outcome is classified from
        what it reports (``vsr.verdict_absent_reason`` /
        ``vsr.failure_category`` / ``vsr.http_status``) rather than from a raw
        :class:`~conclave.models.ModelAnswer`. A candidate whose failure category
        is in :data:`~conclave.models.FAILOVER_CATEGORIES` advances the chain
        (``"failed_over"``, or ``"exhausted"`` on the last candidate); any other
        failure -- including ``"malformed_response"`` (the candidate answered, but
        not usably) -- is terminal for the role, exactly like every other
        adjudication role's rule: a model that answered is never second-guessed by
        another vendor.

        **Unkeyed candidates are NOT pre-skipped for this role** -- a deliberate
        difference from :meth:`adjudicate`. Today, with a chain of one and an
        unkeyed synthesizer, ``_apply_verdict`` still calls ``extract_verdict``,
        which makes two ``call_model`` invocations that return instantly with the
        "no API key" error (no network), yielding two failed
        ``verdict_extraction``/``verdict_repair`` receipts and
        ``verdict_absent_reason == "verdict extraction failed schema
        validation"``. Preserving that byte-for-byte (the chain-of-one rule) means
        letting ``extract_verdict`` run for every candidate rather than
        pre-filtering by :func:`conclave.registry.key_present` first: an unkeyed
        candidate's ``vsr.failure_category`` comes back ``"unkeyed"``, which IS in
        ``FAILOVER_CATEGORIES``, so it becomes ``"failed_over"`` (or
        ``"exhausted"`` on the last candidate) -- matching the receipts exactly,
        since the calls did happen and did fail. The ``"skipped_unkeyed"`` outcome
        (used by every other role via :meth:`_skipped_attempts`) is therefore
        never produced for this role.

        This distinction is invisible to the cache. Whether an unkeyed primary
        lands as ``"skipped_unkeyed"`` (every other role) or as
        ``"failed_over"``/``"exhausted"`` (this role) does not change the
        no-store outcome, because
        :attr:`~conclave.models.CouncilResult.primary_failed_over` treats all
        three the same at the primary's ``attempt_index == 1`` -- see that
        property for the uniform rule.

        The N<2 responder gate is unaffected: ``extract_verdict`` returns
        immediately with NO call made (``vsr.attempt_receipts == []``) regardless
        of which candidate is asked, since the gate counts responding MEMBERS, not
        chain candidates. The chain walk stops after the first candidate in that
        case (asking a second candidate would just repeat the same no-call
        no-op), so N<2 never contributes a ``verdict_extraction`` ledger entry.

        ``extract_verdict`` never raises, and this method only assigns
        already-secret-free objects afterward, so no defensive try/except is
        needed.

        When ``result.manifest`` exists its verdict-provenance slots are populated
        from the LAST candidate consulted (extractor identity + prompt version,
        absent reason, consensus method, verdict type), the full succession ledger
        is appended, and the manifest's ``secret_safety`` stamp is RE-RUN over the
        final content: the stamp was first computed in :meth:`_build_manifest`
        before these fields existed, so re-stamping keeps the VERIFIED claim honest
        over the manifest a consumer actually receives. The new fields (a resolved
        model id, a prompt-version string, the ``verdict_type``/``consensus_method``
        literals, and the ledger's bounded categories) are provably key-free, so
        the stamp stays VERIFIED.

        Args:
            result: The in-progress :class:`CouncilResult` (answers + manifest
                already attached). Mutated in place.
        """
        if not self.extract_verdict_enabled:
            return

        # Lazy import mirrors this module's deferred-import style (``modes`` /
        # ``streaming`` are imported inside methods) and sidesteps any import-cycle
        # risk between council and the verdict engine.
        from .verdict_synthesis import REASON_EXTRACTION_FAILED
        from .verdict_synthesis import extract_verdict as extract_verdict_fn

        protocol_version = ELITE_PROTOCOL_VERSION if result.mode == "elite" else None
        chain = self.synthesizer_chain
        attempts: list[AdjudicationAttempt] = []
        vsr: VerdictSynthesisResult | None = None
        # Running count of verdict receipts already appended across PRIOR chain
        # candidates in this call (QA review M2): each candidate's own
        # ``attempt_receipts`` restarts at 1 (extract_verdict has no visibility
        # into the chain), so without an offset a successor's receipts collide
        # with the primary's ("attempt=1" appears twice). Renumbering here keeps
        # ``attempt`` monotonically increasing across the whole verdict-
        # extraction sequence on the manifest; a chain of one has offset 0 and
        # is byte-for-byte unchanged.
        verdict_receipts_so_far = 0
        for index, candidate in enumerate(chain, start=1):
            model_id = self.config.resolve_model_id(candidate)
            vsr = await extract_verdict_fn(
                result.prompt,
                result.answers,
                synthesizer_name=candidate,
                synthesizer_model_id=model_id,
                config=self.config,
                temperature=self.temperature,
                timeout=self.timeout,
                protocol_version=protocol_version,
                max_output_tokens=self.max_output_tokens,
            )
            renumbered_receipts = [
                receipt.model_copy(update={"attempt": verdict_receipts_so_far + offset})
                for offset, receipt in enumerate(vsr.attempt_receipts, start=1)
            ]
            if record_receipts:
                self._append_manifest_receipts(result, renumbered_receipts)
            verdict_receipts_so_far += len(renumbered_receipts)
            if not vsr.attempt_receipts:
                # N<2 gate: no call was made for this candidate (or any other --
                # the responder count does not depend on which candidate is
                # asked), so there is nothing to adjudicate. Stop here rather
                # than repeating the same no-op for every remaining candidate.
                break

            def _attempt(
                outcome: str,
                *,
                failure_category: str | None = None,
                http_status: int | None = None,
                _candidate: str = candidate,
                _model_id: str = model_id,
                _index: int = index,
            ) -> AdjudicationAttempt:
                """Build one ledger entry for the candidate/index of this iteration.

                The loop variables are bound as default-argument values so the
                closure captures THIS iteration's ``candidate``/``model_id``/
                ``index`` rather than whatever they are when the loop ends
                (flake8-bugbear B023) -- mirrors :meth:`adjudicate`'s ``_attempt``.
                """
                return AdjudicationAttempt(
                    role="verdict_extraction",
                    candidate=_candidate,
                    model_id=_model_id,
                    attempt_index=_index,
                    outcome=outcome,
                    failure_category=failure_category,
                    http_status=http_status,
                )

            failed = vsr.verdict_absent_reason == REASON_EXTRACTION_FAILED
            is_last = index == len(chain)
            outcome = self._classify_outcome(
                failed=failed, category=vsr.failure_category, is_last=is_last
            )
            attempts.append(
                _attempt(
                    outcome,
                    failure_category=None if outcome == "success" else vsr.failure_category,
                    http_status=None if outcome == "success" else vsr.http_status,
                )
            )
            if outcome == "failed_over":
                logger.warning(
                    "verdict_extraction: '%s' failed (%s); trying next candidate",
                    candidate,
                    vsr.failure_category,
                )
                continue
            if outcome == "exhausted":
                continue
            break

        result.verdict = vsr.verdict
        if vsr.verdict is not None:
            # Hoist the canonical verdict's values to the top-level mirrors.
            result.consensus_score = vsr.verdict.consensus_score
            result.consensus_method = vsr.verdict.consensus_method
            result.consensus_label = vsr.verdict.consensus_label
            result.conflicts = vsr.verdict.conflicts
            result.provider_votes = vsr.verdict.provider_votes
            result.minority_reports = vsr.verdict.minority_reports

        if result.manifest is not None:
            result.manifest.adjudication_succession.extend(attempts)
            result.manifest.verdict_extraction = vsr.extraction
            result.manifest.verdict_absent_reason = vsr.verdict_absent_reason
            result.manifest.consensus_method = vsr.verdict.consensus_method if vsr.verdict else None
            result.manifest.verdict_type = vsr.verdict.verdict_type if vsr.verdict else None
            # Re-stamp over the now-complete manifest so the VERIFIED claim covers
            # the verdict-provenance fields (including the new ledger entries)
            # just written (they are key-free).
            result.manifest.secret_safety = verified_secret_safety(result.manifest)

    @staticmethod
    def _classify_outcome(
        *, failed: bool, category: str | None, is_last: bool
    ) -> AdjudicationAttemptOutcome:
        """The single failover rule, shared by every adjudication role (DSE-1512 review).

        An infrastructure failure (``category in FAILOVER_CATEGORIES``) advances
        the chain -- ``"failed_over"``, or ``"exhausted"`` when there is no next
        candidate to try. Any other failure is terminal for the role: a model
        that answered (even unusably) is never second-guessed by another vendor.
        A non-failure is always ``"success"``.

        Args:
            failed: Whether this attempt failed.
            category: The attempt's typed failure category, or ``None``. Callers
                that must force a terminal outcome regardless of the real
                category (e.g. streaming's post-first-delta rule) pass ``None``
                here while still recording the true category on the ledger entry.
            is_last: Whether this is the last candidate in the chain.

        Returns:
            The bounded :data:`~conclave.manifest.AdjudicationAttemptOutcome`.
        """
        if not failed:
            return "success"
        if category in FAILOVER_CATEGORIES:
            return "exhausted" if is_last else "failed_over"
        return "terminal_failure"

    async def adjudicate(
        self, role: AdjudicationRole, system_prompt: str, user_content: str
    ) -> AdjudicationOutcome:
        """Walk ``synthesizer_chain`` for one adjudication role (DSE-1512).

        Rule: candidates are tried in declared order; an unkeyed candidate is
        skipped without a call; a call that fails with a category in
        :data:`FAILOVER_CATEGORIES` advances to the next candidate; ANY other
        failure is terminal for the role -- a model that answered is never
        second-guessed by another vendor, which would let adjudication shop for
        a result. No scoring, no health tracking: the order is the operator's.

        Args:
            role: Which adjudication role this call serves (recorded on every
                :class:`~conclave.manifest.AdjudicationAttempt`).
            system_prompt: System instruction for the synthesizer/judge.
            user_content: The user-role content (prompt + answers/critiques).

        Returns:
            An :class:`AdjudicationOutcome` carrying the resolved answer (or
            ``None`` when every candidate was unkeyed), the full attempt
            ledger, and the list of real calls made.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        attempts: list[AdjudicationAttempt] = []
        called: list[ModelAnswer] = []
        last_failure: ModelAnswer | None = None
        chain = self.synthesizer_chain
        for index, candidate in enumerate(chain, start=1):
            model_id = self.config.resolve_model_id(candidate)

            def _attempt(
                outcome: str,
                *,
                failure_category: str | None = None,
                http_status: int | None = None,
                _candidate: str = candidate,
                _model_id: str = model_id,
                _index: int = index,
            ) -> AdjudicationAttempt:
                """Build one ledger entry for the candidate/index of this iteration.

                The loop variables are bound as default-argument values so the
                closure captures THIS iteration's ``candidate``/``model_id``/
                ``index`` rather than whatever they are when the loop ends
                (flake8-bugbear B023).
                """
                return AdjudicationAttempt(
                    role=role,
                    candidate=_candidate,
                    model_id=_model_id,
                    attempt_index=_index,
                    outcome=outcome,
                    failure_category=failure_category,
                    http_status=http_status,
                )

            if not key_present(model_id):
                attempts.append(_attempt("skipped_unkeyed", failure_category="unkeyed"))
                continue
            answer = await call_model(
                candidate,
                model_id,
                messages,
                config=self.config,
                temperature=self.temperature,
                timeout=self.timeout,
                max_output_tokens=self.max_output_tokens,
            )
            called.append(answer)
            is_last = index == len(chain)
            category = answer.failure_category
            outcome = self._classify_outcome(
                failed=not answer.ok, category=category, is_last=is_last
            )
            attempts.append(
                _attempt(
                    outcome,
                    failure_category=None if outcome == "success" else category,
                    http_status=None if outcome == "success" else answer.http_status,
                )
            )
            if outcome == "success":
                return AdjudicationOutcome(answer=answer, attempts=attempts, called=called)
            if outcome == "terminal_failure":
                return AdjudicationOutcome(answer=answer, attempts=attempts, called=called)
            # "failed_over" or "exhausted": advance the chain (or fall through to
            # the final return below when this was the last candidate).
            last_failure = answer
            if outcome == "failed_over":
                logger.warning(
                    "%s: '%s' failed (%s); trying next candidate", role, candidate, category
                )
        return AdjudicationOutcome(answer=last_failure, attempts=attempts, called=called)

    def _skipped_attempts(self, role: AdjudicationRole) -> list[AdjudicationAttempt]:
        """Build the succession ledger for a chain where NO candidate is keyed.

        Used by every adjudication call site (synthesis, debate's final
        consolidation, the adversarial judge) when it short-circuits on the
        "no candidate has an API key" branch -- :meth:`adjudicate` is never
        called (there is nothing to call), but the ledger must still record
        that every chain candidate was considered and skipped, one
        ``skipped_unkeyed`` :class:`~conclave.manifest.AdjudicationAttempt` per
        candidate in chain order.

        Args:
            role: Which adjudication role this skip ledger serves.

        Returns:
            One ``skipped_unkeyed`` attempt per candidate in
            ``self.synthesizer_chain``, in chain order.
        """
        return [
            AdjudicationAttempt(
                role=role,
                candidate=candidate,
                model_id=self.config.resolve_model_id(candidate),
                attempt_index=index,
                outcome="skipped_unkeyed",
                failure_category="unkeyed",
            )
            for index, candidate in enumerate(self.synthesizer_chain, start=1)
        ]

    def _chain_unkeyed_error(self, actor: str, suffix: str) -> str | None:
        """Return the no-key error for the whole chain, or ``None`` if any candidate is keyed.

        The shared keyed-check for every adjudication call site (synthesis,
        debate's final consolidation, the adversarial judge) that needs a
        distinct "no candidate has a key" message before calling
        :meth:`adjudicate` (DSE-1512 review, Unit C). ``actor`` is the role
        noun used in the message (``"synthesizer"`` / ``"judge"``); ``suffix``
        is the mode-specific tail (``"returning raw answers only"``,
        ``"returning final-round answers only"``, ``"returning proposal and
        critiques only"``). A chain of one keeps the historic single-candidate
        wording verbatim byte-for-byte.

        Args:
            actor: The role noun for the message.
            suffix: The mode-specific tail clause.

        Returns:
            The formatted error string when every chain candidate is unkeyed,
            else ``None``.
        """
        keyed = [c for c in self.synthesizer_chain if key_present(self.config.resolve_model_id(c))]
        if keyed:
            return None
        if len(self.synthesizer_chain) == 1:
            primary_id = self.config.resolve_model_id(self.synthesizer)
            return f"{actor} '{self.synthesizer}' ({primary_id}) has no API key; {suffix}"
        names = ", ".join(self.synthesizer_chain)
        return f"{actor} chain [{names}] has no API key for any candidate; {suffix}"

    def _record_adjudication(
        self,
        result: CouncilResult,
        attempts: list[AdjudicationAttempt],
        called: list[ModelAnswer],
        *,
        phase: str,
        record_receipts: bool = True,
        protocol_version: str | None = None,
        prompt_version: str | None = SYNTHESIS_PROMPT_VERSION,
    ) -> None:
        """Append the succession ledger (always) and one receipt per real call.

        No-op when the result has no manifest yet. ``_recompute_manifest_accounting``
        re-derives totals and re-stamps ``secret_safety`` over the new content.

        Args:
            result: The in-progress result. Mutated in place.
            attempts: The full attempt ledger from :meth:`adjudicate`, including
                skipped-unkeyed candidates.
            called: The real calls made, in order (from
                :attr:`AdjudicationOutcome.called`).
            phase: The manifest receipt phase to stamp on each call.
            record_receipts: When ``False``, skip appending receipts (the ledger
                is still recorded); used by callers that record receipts
                elsewhere.
            protocol_version: Optional protocol version to stamp on receipts.
            prompt_version: Prompt version to stamp on receipts; defaults to the
                synthesis prompt version.
        """
        if result.manifest is None:
            return
        result.manifest.adjudication_succession.extend(attempts)
        if record_receipts:
            result.manifest.receipts.extend(
                receipt_from_answer(
                    answer,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    phase=phase,
                    attempt=index,
                    protocol_version=protocol_version,
                    prompt_version=prompt_version,
                    max_output_tokens=self.max_output_tokens,
                )
                for index, answer in enumerate(called, start=1)
            )
        self._recompute_manifest_accounting(result.manifest)

    async def _adjudicate_and_record(
        self,
        result: CouncilResult,
        role: AdjudicationRole,
        system_prompt: str,
        user_content: str,
        *,
        phase: str,
        protocol_version: str | None = None,
    ) -> AdjudicationOutcome:
        """``adjudicate`` then ``_record_adjudication`` -- the common tail of every role.

        Shared by :meth:`_synthesize`, :func:`conclave.modes._debate_synthesize`,
        and :func:`conclave.modes._adversarial_judge` (DSE-1512 review, Unit C)
        so the "call the chain, then land the ledger + receipts" sequence is
        written exactly once.

        Args:
            result: The in-progress result. Mutated in place via
                :meth:`_record_adjudication`.
            role: Which adjudication role this call serves.
            system_prompt: System instruction for the synthesizer/judge.
            user_content: The user-role content (prompt + answers/critiques).
            phase: The manifest receipt phase to stamp on each real call.
            protocol_version: Optional protocol version to stamp on receipts.

        Returns:
            The :class:`AdjudicationOutcome` from :meth:`adjudicate`.
        """
        outcome = await self.adjudicate(role, system_prompt, user_content)
        self._record_adjudication(
            result, outcome.attempts, outcome.called, phase=phase, protocol_version=protocol_version
        )
        return outcome

    async def synthesize_blocks(self, system_prompt: str, user_content: str) -> ModelAnswer:
        """Call the synthesizer chain with an arbitrary system + user message.

        A compatibility wrapper kept for external/library callers that want one
        synthesizer answer without building a full :class:`CouncilResult` (QA
        review M4). It walks ``synthesizer_chain`` via :meth:`adjudicate` --
        with a chain of one (the default) this is byte-for-byte the historic
        single-call behavior -- and returns whichever :class:`ModelAnswer` the
        walk resolves to, or a synthetic ``ModelAnswer.error`` when every
        candidate was unkeyed. Callers are responsible for checking
        ``key_present`` beforehand when they need a distinct no-key message.

        **No caller inside this package uses this method any more.**
        :meth:`_synthesize`, :func:`conclave.modes._debate_synthesize`, and
        :func:`conclave.modes._adversarial_judge` all go through
        :meth:`_adjudicate_and_record` instead, because THIS method records
        neither the succession ledger nor any receipts -- it has no
        :class:`CouncilResult` to attach them to. A caller that needs the audit
        trail (the manifest's ``adjudication_succession`` + per-call receipts)
        must go through :meth:`Council.ask` / :meth:`debate` / :meth:`adversarial`
        instead; this method remains for a caller that genuinely only wants the
        bare answer. No behavior change from calling :meth:`adjudicate` directly.

        Args:
            system_prompt: System instruction for the synthesizer/judge.
            user_content: The user-role content (prompt + answers/critiques).

        Returns:
            A :class:`ModelAnswer` from the synthesizer chain.
        """
        outcome = await self.adjudicate("synthesis", system_prompt, user_content)
        if outcome.answer is not None:
            return outcome.answer
        synth_id = self.config.resolve_model_id(self.synthesizer)
        return ModelAnswer(
            name=self.synthesizer,
            model_id=synth_id,
            error="no candidate in synthesizer chain has an API key",
            failure_category="unkeyed",
        )

    async def debate(
        self, prompt: str, rounds: int = 2, converge_threshold: float | None = None
    ) -> CouncilResult:
        """Run a multi-round debate. See :func:`conclave.modes.run_debate`.

        Round 1 is an independent fan-out; rounds 2..N show each member its
        peers' anonymized prior answers; the synthesizer consolidates survivors.
        Cache-served when caching is enabled (``rounds`` and the resolved
        ``converge_threshold`` are part of the key).

        Args:
            prompt: The user prompt.
            rounds: Maximum number of rounds (the historic fixed count).
            converge_threshold: Opt-in early-stop threshold. ``None`` (default)
                defers to ``config.converge_threshold`` (off unless set in
                ``~/.conclave/config.yml``); an explicit value overrides it for
                this call. With early-stop off the debate runs exactly ``rounds``,
                identical to the historic behavior. Mirrors the ``cache``
                None-defers-to-config convention.
        """
        from .modes import run_debate

        # Resolve the opt-in here (mirrors ``cache``: explicit arg wins, else
        # config) so the cache key reflects what will actually run.
        threshold = (
            self.config.converge_threshold if converge_threshold is None else converge_threshold
        )
        return await self._cached_run(
            prompt,
            "debate",
            lambda: run_debate(self, prompt, rounds=rounds, converge_threshold=threshold),
            rounds=rounds,
            converge_threshold=threshold,
        )

    async def adversarial(self, prompt: str, proposer: str | None = None) -> CouncilResult:
        """Run propose -> refute -> verdict. See :func:`conclave.modes.run_adversarial`.

        ``proposer`` (friendly name) defaults to the first requested member.
        Cache-served when caching is enabled (``proposer`` is part of the key).
        """
        from .modes import run_adversarial

        return await self._cached_run(
            prompt,
            "adversarial",
            lambda: run_adversarial(self, prompt, proposer=proposer),
            proposer=proposer,
        )

    async def elite(self, prompt: str) -> CouncilResult:
        """Run the answer/claim-audited Elite Decision Protocol.

        Completed runs synthesize the council's revised answers and apply the
        canonical structured verdict. A run that fails any three-responder gate
        returns its phase artifacts without synthesis or verdict extraction.
        """
        from .modes import run_elite

        async def run() -> CouncilResult:
            result = await run_elite(self, prompt)
            if result.elite is not None and result.elite.completed:
                # Ensure the manifest exists BEFORE synthesizing so
                # ``_synthesize``'s ``_record_adjudication`` call has somewhere
                # to land the succession ledger and per-call receipts
                # (DSE-1512). ``_ensure_manifest`` only flattens the already-
                # collected phase artifacts (initial/critique/revision), so
                # building it before synthesis runs is safe -- synthesis is not
                # one of those phases.
                self._ensure_manifest(result, "elite")
                await self._synthesize(result)
                if result.synthesis is None:
                    result.elite.decision_readiness = "not_ready"
                    result.elite.readiness_reasons = ["synthesis.failed"]
                elif not self.extract_verdict_enabled:
                    result.elite.decision_readiness = "indeterminate"
                    result.elite.readiness_reasons = ["adjudication.disabled"]
                else:
                    await self._apply_verdict(result)
                    self._set_elite_readiness(result)
            return result

        return await self._cached_run(prompt, "elite", run)

    @staticmethod
    def _set_elite_readiness(result: CouncilResult) -> None:
        """Classify Elite readiness after required synthesis and adjudication."""
        elite = result.elite
        if elite is None:
            return
        if result.verdict is not None:
            elite.decision_readiness = "ready"
            elite.readiness_reasons = []
            return

        absent_reason = (
            result.manifest.verdict_absent_reason if result.manifest is not None else None
        )
        if absent_reason == "verdict extraction failed schema validation":
            elite.decision_readiness = "not_ready"
            elite.readiness_reasons = ["adjudication.verdict_extraction_failed"]
        elif absent_reason == "fewer than 2 responding members":
            elite.decision_readiness = "not_ready"
            elite.readiness_reasons = ["adjudication.insufficient_responders"]
        elif absent_reason == "open-ended prompt (no decision/review to adjudicate)":
            elite.decision_readiness = "indeterminate"
            elite.readiness_reasons = ["adjudication.open_ended"]
        else:
            elite.decision_readiness = "indeterminate"
            elite.readiness_reasons = ["adjudication.unknown"]

    async def aclose(self) -> None:
        """Close the shared pooled HTTP client.

        Library users running their own event loop (e.g. a server) should call
        this on shutdown so the process-wide connection pool is released and no
        "unclosed client" warning is emitted under strict asyncio. It is safe to
        call more than once; the pooled client is recreated lazily on next use.

        The synchronous wrappers (:meth:`ask_sync`, :meth:`debate_sync`,
        :meth:`adversarial_sync`) already close the client automatically before
        their event loop ends, so CLI/sync callers do not need to call this.
        """
        await transport.aclose()

    def close_sync(self) -> None:
        """Synchronous wrapper around :meth:`aclose` for non-async callers."""
        self._run_sync(self.aclose, "close_sync", close_client=False)

    @staticmethod
    def _run_sync(
        coro_factory: Callable[[], asyncio.Future | object],
        label: str,
        *,
        close_client: bool = True,
    ):
        """Run an async council method synchronously, guarding nested loops.

        ``asyncio.run`` creates (and tears down) a fresh event loop per call. The
        pooled httpx client is bound to whichever loop first used it, so we close
        it inside that same loop's ``finally`` before the loop is destroyed --
        otherwise the pool leaks and asyncio emits an "unclosed client" warning.
        ``close_client=False`` is used by :meth:`close_sync` itself to avoid
        recursively re-closing.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                f"{label}() called from within a running event loop; await the async method instead"
            )

        async def _runner():
            try:
                return await coro_factory()
            finally:
                if close_client:
                    await transport.aclose()

        return asyncio.run(_runner())

    def ask_sync(self, prompt: str, synthesize: bool = True) -> CouncilResult:
        """Synchronous wrapper around :meth:`ask`.

        Safe to call from non-async code. Raises ``RuntimeError`` if invoked
        from inside a running event loop -- use :meth:`ask` there instead.
        """
        return self._run_sync(lambda: self.ask(prompt, synthesize=synthesize), "ask_sync")

    def stream_sync(
        self,
        prompt: str,
        on_event: Callable[[StreamEvent], None],
        synthesize: bool = True,
    ) -> CouncilResult:
        """Drive :meth:`ask_stream` synchronously, invoking ``on_event`` per event.

        For non-async callers (the CLI ``--stream`` path). Each
        :class:`StreamEvent` is passed to ``on_event`` as it arrives so live
        output can be rendered; the fully-assembled :class:`CouncilResult` (from
        the terminal ``done`` event) is returned. Closes the pooled HTTP client
        when the loop ends, like the other ``*_sync`` wrappers. Raises
        ``RuntimeError`` if invoked from inside a running event loop -- iterate
        :meth:`ask_stream` directly there instead.

        Args:
            prompt: The user prompt to fan out.
            on_event: Callback invoked once per :class:`StreamEvent` in order.
            synthesize: When ``True`` (default), stream the synthesizer too.

        Returns:
            The final :class:`CouncilResult` carried by the ``done`` event.
        """

        async def _consume() -> CouncilResult:
            final: CouncilResult | None = None
            async for event in self.ask_stream(prompt, synthesize=synthesize):
                on_event(event)
                if event.type == "done" and event.result is not None:
                    final = event.result
            # ask_stream always ends with a done event carrying a result; fall
            # back to an empty result only as a defensive guard.
            return final if final is not None else CouncilResult(prompt=prompt)

        return self._run_sync(_consume, "stream_sync")

    def debate_sync(
        self, prompt: str, rounds: int = 2, converge_threshold: float | None = None
    ) -> CouncilResult:
        """Synchronous wrapper around :meth:`debate`."""
        return self._run_sync(
            lambda: self.debate(prompt, rounds=rounds, converge_threshold=converge_threshold),
            "debate_sync",
        )

    def adversarial_sync(self, prompt: str, proposer: str | None = None) -> CouncilResult:
        """Synchronous wrapper around :meth:`adversarial`."""
        return self._run_sync(
            lambda: self.adversarial(prompt, proposer=proposer), "adversarial_sync"
        )

    def elite_sync(self, prompt: str) -> CouncilResult:
        """Synchronous wrapper around :meth:`elite`."""
        return self._run_sync(lambda: self.elite(prompt), "elite_sync")

    async def vote(self, prompt: str, choices: list[str]) -> CouncilResult:
        """Run a constrained-choice vote. See :func:`conclave.modes.run_vote`.

        Each member receives the prompt and a labelled option set (A, B, C, ...)
        and must respond with a single letter. Results are tallied and a winner
        (plurality) or split is reported on ``result.vote``.

        Args:
            prompt: The question to vote on.
            choices: Two or more option strings. At least 2 required.
        """
        from .modes import run_vote

        return await self._cached_run(
            prompt,
            "vote",
            lambda: run_vote(self, prompt, choices=choices),
            choices=choices,
        )

    def vote_sync(self, prompt: str, choices: list[str]) -> CouncilResult:
        """Synchronous wrapper around :meth:`vote`."""
        return self._run_sync(lambda: self.vote(prompt, choices=choices), "vote_sync")
