"""Pydantic data models for conclave configuration and results.

These are the stable, importable contract used by both the CLI and any
downstream library consumer (e.g. mcp-warden). Keep field names stable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from .verdict import (
    CouncilConflict,
    CouncilVerdict,
    MinorityReport,
    ProviderVote,
)

ELITE_PROTOCOL_VERSION = "elite_v1"
ELITE_MIN_RESPONDERS = 3

DecisionReadiness = Literal["ready", "not_ready", "indeterminate"]

_ANSWER_ID_VERSION = "answer_v1"


def stable_answer_id(name: str, model_id: str, answer: str) -> str:
    """Return an opaque, deterministic identity for one successful answer.

    The digest is derived only from immutable, non-credential response facts.
    Raw answer text and model metadata are never exposed in the identifier.
    Latency and usage are intentionally excluded because they can change across
    equivalent calls and cache reconstruction.
    """
    identity = json.dumps(
        {
            "version": _ANSWER_ID_VERSION,
            "name": name,
            "model_id": model_id,
            "answer": answer,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"ca_{hashlib.sha256(identity).hexdigest()[:24]}"


def derive_phase_answer_id(
    answer: ModelAnswer,
    phase: str,
    *,
    parent_answer_ids: Iterable[str] = (),
) -> ModelAnswer:
    """Assign an idempotent phase-specific identity to a successful artifact.

    Elite critiques and revisions transform earlier answers. Their IDs therefore
    include the phase and the sorted IDs of the within-run parent artifacts while
    remaining opaque and secret-free. Failed answers remain unidentified.
    """
    if not answer.ok:
        return answer
    prefix = f"ca_{phase}_"
    if answer.answer_id and answer.answer_id.startswith(prefix):
        return answer
    base_id = answer.answer_id or stable_answer_id(
        answer.name, answer.model_id, answer.answer or ""
    )
    identity = json.dumps(
        {
            "version": _ANSWER_ID_VERSION,
            "phase": phase,
            "base_answer_id": base_id,
            "parent_answer_ids": sorted(parent_answer_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    answer.answer_id = f"{prefix}{hashlib.sha256(identity).hexdigest()[:24]}"
    return answer


# NOTE: ``ModelHarnessManifest`` is imported at the BOTTOM of this module (just
# before the ``model_rebuild()`` calls), not here. ``manifest`` imports
# :class:`TokenUsage` from this module, so importing it before ``TokenUsage`` is
# defined would be a circular import. Deferring the import until after the leaf
# types exist breaks the cycle while still making the name available for the
# ``CouncilResult.manifest`` annotation at rebuild time (the same reason the
# verdict types could be imported eagerly is that ``verdict`` imports nothing
# back from ``models``; ``manifest`` does, so it gets the late-import treatment).


def _default_prompt_version() -> str:
    """Resolve the current synthesis-prompt version without an import cycle.

    ``conclave.prompts`` imports this module, so importing it at module load
    would be circular. The import is deferred into this factory (run only when a
    ``CouncilResult`` is constructed, by which point both modules are loaded), so
    every result defaults to the live :data:`conclave.prompts.SYNTHESIS_PROMPT_VERSION`.
    """
    from .prompts import SYNTHESIS_PROMPT_VERSION

    return SYNTHESIS_PROMPT_VERSION


class TokenUsage(BaseModel):
    """Token accounting for a single model call."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


# DSE-1512 — typed failure categories. Derived at the RAISE SITE from the HTTP
# status or exception type, never by inspecting a rendered error string. The
# adjudication ladder (Council.adjudicate) fails over ONLY on the categories in
# FAILOVER_CATEGORIES: infrastructure failures where no model ever produced an
# answer. A model that answered (even malformed) is terminal for that role.
FailureCategory = Literal[
    "unkeyed",  # env var absent -- no call made
    "unresolved",  # unknown provider prefix -- no call made
    "auth",  # 401 / 403
    "quota",  # 402 / 429
    "unavailable",  # 5xx
    "timeout",  # 408 or transport deadline
    "transport",  # DNS / connection / other httpx network error
    "bad_request",  # other 4xx -- the request was wrong, not the vendor
    "malformed_response",  # 2xx with an unusable payload / empty content
    "unexpected",  # anything else -- never failed over
]

FAILOVER_CATEGORIES: frozenset[str] = frozenset(
    {"unkeyed", "unresolved", "auth", "quota", "unavailable", "timeout", "transport"}
)


def categorize_http_status(status: int) -> FailureCategory:
    """Map a non-2xx HTTP status to a :data:`FailureCategory` (pure, no I/O).

    Every 4xx/5xx status is bounded by one of the categories above. Anything
    OUTSIDE that range (1xx, 2xx, 3xx, or >= 600 -- none of which this
    function is meant to be called with, but a defensive raise site could)
    maps to ``"unexpected"`` (QA review M3), never ``"malformed_response"``:
    that category is reserved for a 2xx response with an unusable PAYLOAD, a
    distinct failure mode this function never sees. Neither ``"unexpected"``
    nor ``"malformed_response"`` is in :data:`FAILOVER_CATEGORIES`, so this
    never affects failover either way.
    """
    if status in (401, 403):
        return "auth"
    if status in (402, 429):
        return "quota"
    if status == 408:
        return "timeout"
    if 500 <= status <= 599:
        return "unavailable"
    if 400 <= status <= 499:
        return "bad_request"
    return "unexpected"


class ModelAnswer(BaseModel):
    """One council member's response (or failure).

    Attributes:
        name: Friendly council member name (e.g. ``"grok"``).
        model_id: Resolved provider-prefixed model id (e.g. ``"xai/grok-4.3"``).
        answer: The raw text answer, or ``None`` if the call failed.
        latency_s: Wall-clock seconds for the call.
        usage: Token usage if reported by the provider.
        error: Error message if the call failed, else ``None``.
        answer_id: Stable opaque id assigned by Conclave (not emitted by the
            model). It provides within-run answer provenance and backs the legacy
            compatibility field ``evidence_answer_ids``; it is not external
            evidence that a claim is true. Failed calls keep ``None``.
        warnings: Non-fatal notes about this answer (e.g. structured-output repair
            applied). Empty by default. Distinct from ``error``, which marks the
            whole call as failed.
        failure_category: Typed classification of ``error`` (DSE-1512), derived
            at the raise site from the HTTP status or exception type -- never by
            inspecting ``error`` text. ``None`` on success and on any answer
            collected before this field existed. See :data:`FailureCategory` and
            :data:`FAILOVER_CATEGORIES`.
        http_status: The HTTP status code that produced ``error``, when the
            failure came from a response (as opposed to a pre-call or transport
            failure). ``None`` on success and whenever no HTTP response was
            received.
    """

    name: str
    model_id: str
    answer: str | None = None
    latency_s: float = 0.0
    usage: TokenUsage | None = None
    error: str | None = None
    answer_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    failure_category: FailureCategory | None = None
    http_status: int | None = None

    @property
    def ok(self) -> bool:
        """True when the member returned a usable answer."""
        return self.error is None and self.answer is not None

    @property
    def latency_ms(self) -> float:
        """Call latency in milliseconds, derived from :attr:`latency_s`."""
        return self.latency_s * 1000.0


class EliteResult(BaseModel):
    """Protocol completion and independently adjudicated decision readiness."""

    protocol_version: str = ELITE_PROTOCOL_VERSION
    required_responders: int = Field(default=ELITE_MIN_RESPONDERS, ge=ELITE_MIN_RESPONDERS)
    completed: bool = False
    failure_reason: str | None = None
    decision_readiness: DecisionReadiness = "indeterminate"
    readiness_reasons: list[str] = Field(default_factory=lambda: ["adjudication.not_evaluated"])
    initial_answers: list[ModelAnswer] = Field(default_factory=list)
    critiques: list[ModelAnswer] = Field(default_factory=list)
    revisions: list[ModelAnswer] = Field(default_factory=list)


class StreamEvent(BaseModel):
    """One incremental event from a streaming council run (issue #7).

    Streaming yields a flat sequence of these so a consumer can render live
    output without knowing the council's internals. The terminal ``done`` event
    carries the fully-assembled :class:`CouncilResult`, so a consumer that only
    wants the final structured result can ignore every chunk and read
    ``done`` -- the result shape is byte-for-byte the same as the
    non-streaming path.

    Attributes:
        type: The event kind:

            * ``"member_delta"`` -- an incremental text chunk from one council
              member. ``name``/``model_id`` identify the member and ``text``
              carries the new tokens.
            * ``"member_done"`` -- a member finished (or failed). ``answer``
              carries that member's final :class:`ModelAnswer` (with ``error``
              set on failure, partial text preserved if any).
            * ``"synthesis_delta"`` -- an incremental text chunk from the
              synthesizer (only when ``synthesize=True`` and synthesis runs).
            * ``"synthesis_done"`` -- the synthesizer finished; ``answer`` holds
              its final :class:`ModelAnswer`.
            * ``"done"`` -- the run is complete; ``result`` holds the full
              :class:`CouncilResult`.
        name: Friendly member/synthesizer name for delta/done events.
        model_id: Resolved model id for delta/done events.
        text: The incremental text for ``*_delta`` events.
        answer: The final :class:`ModelAnswer` for ``member_done`` /
            ``synthesis_done`` events.
        result: The full :class:`CouncilResult` for the terminal ``done`` event.
    """

    type: str
    name: str | None = None
    model_id: str | None = None
    text: str | None = None
    answer: ModelAnswer | None = None
    result: CouncilResult | None = None


class DebateRound(BaseModel):
    """One round of a multi-round debate.

    Attributes:
        round_number: 1-based index of the round.
        answers: One ``ModelAnswer`` per member that participated in this round.
            A member that errored in an earlier round is absent here (it has
            dropped out of the debate).
    """

    round_number: int
    answers: list[ModelAnswer] = Field(default_factory=list)

    @property
    def successful_answers(self) -> list[ModelAnswer]:
        """Members that returned a usable answer in this round."""
        return [a for a in self.answers if a.ok]


class AdversarialResult(BaseModel):
    """The proposal/critique/verdict structure of an adversarial run.

    Attributes:
        proposer: Friendly name of the member that produced the proposal.
        proposal: The proposer's ``ModelAnswer`` (answer or error).
        critiques: One ``ModelAnswer`` per critic member, each prompted to
            refute the proposal.
        verdict: The judge's final strengthened answer, or ``None`` if the
            judge could not run.
        verdict_error: Error message if the judge step failed, else ``None``.
        judge: Friendly name of the judge (synthesizer) model.
        judge_model_id: Resolved provider-prefixed id of the judge.
    """

    proposer: str
    proposal: ModelAnswer
    critiques: list[ModelAnswer] = Field(default_factory=list)
    verdict: str | None = None
    verdict_error: str | None = None
    judge: str | None = None
    judge_model_id: str | None = None

    @property
    def successful_critiques(self) -> list[ModelAnswer]:
        """Critics that returned a usable critique."""
        return [c for c in self.critiques if c.ok]


class VoteResult(BaseModel):
    """The tally from a constrained-choice vote run.

    Attributes:
        choices: The fixed option labels offered to members.
        votes: Map of member friendly name → the label they chose (or ``None``
            when a member's answer could not be parsed to a valid choice).
        tally: Map of label → vote count across all members that returned a
            parseable vote. Members that errored or returned an unrecognised
            response are excluded from the tally.
        winner: The label with the most votes, or ``None`` on a tie.
        split: ``True`` when no single choice won an outright plurality (tie).
    """

    choices: list[str] = Field(default_factory=list)
    votes: dict[str, str | None] = Field(default_factory=dict)
    tally: dict[str, int] = Field(default_factory=dict)
    winner: str | None = None
    split: bool = False


class CouncilResult(BaseModel):
    """The full outcome of a council run.

    Attributes:
        prompt: The original user prompt.
        mode: The run mode that produced this result
            (``"synthesize"`` | ``"raw"`` | ``"debate"`` | ``"adversarial"``
            | ``"vote"``).
        answers: One ``ModelAnswer`` per attempted council member. For
            ``debate`` this mirrors the final round so existing consumers that
            read ``answers``/``synthesis`` keep working unchanged.
        synthesizer: Friendly name of the synthesizer model, if synthesis ran.
        synthesizer_model_id: Resolved provider-prefixed id of the synthesizer.
        synthesis: The merged consolidated answer, or ``None`` if not produced.
            For ``debate`` this holds the final synthesized answer; for
            ``adversarial`` it mirrors the judge's verdict.
        synthesis_error: Error message if synthesis failed, else ``None``.
        skipped: Friendly names skipped because no key was available.
        rounds: Per-round answers for ``debate`` mode (empty otherwise).
        adversarial: The proposal/critique/verdict structure for
            ``adversarial`` mode (``None`` otherwise).
        cached: ``True`` when this result was served from the optional result
            cache rather than produced by a live run. ``False`` for every live
            run and for freshly stored entries. Lets a consumer detect a cache
            hit without re-running. See :mod:`conclave.cache`.
        converged: ``True`` when a ``debate`` run stopped early because answers
            converged (the convergence score crossed the configured threshold)
            before ``rounds`` was exhausted. ``False`` for every other run,
            including a debate that ran its full round count. The actual number
            of rounds run is always ``len(rounds)``. See
            :func:`conclave.modes.run_debate`.
        convergence_score: The convergence score (0.0--1.0) of the round that
            triggered an early stop, or ``None`` when no early stop occurred.
            Higher means more stable round-over-round (more converged).
        prompt_version: The version tag of the synthesizer/judge prompt set used
            for this run (:data:`conclave.prompts.SYNTHESIS_PROMPT_VERSION`).
            Stamped on **every** result regardless of mode or whether synthesis
            actually ran, so a downstream eval/regression suite can detect that
            the synthesis prompt wording changed between two runs instead of
            silently attributing the shift to model drift. Opaque string; only
            equality is meaningful.
        verdict: The synthesized adjudication of the run (CAC-01 result contract
            v2), or ``None`` when no verdict applies — open-ended generation,
            N<2 responding members, or structured extraction failed after one
            repair (DD-2 verdict-absent rule). When absent, ``synthesis`` and
            ``member_answers`` are still populated. Filled by CAC-05.
        consensus_score: Position-cluster ratio in ``[0.0, 1.0]`` for the
            verdict's primary recommendation (DD-1, ``position_cluster_ratio_v1``),
            or ``None`` (N<2 or no positioned members). Distinct from
            ``convergence_score`` (difflib text-stability); never conflated.
            Filled by CAC-05.
        consensus_method: The consensus method literal used
            (``"position_cluster_ratio_v1"``), or ``None`` until computed.
        consensus_label: Deterministic bucket derived from ``consensus_score``
            (``unanimous``/``strong``/``majority``/``split``/``none``), or
            ``None`` until computed.
        conflicts: Disagreements between positions (DD-2); empty when <2 positions
            or not yet computed. Filled by CAC-05.
        provider_votes: Per-provider votes for positions (DD-2, GH #3); empty
            until computed. Filled by CAC-05.
        minority_reports: Dissenting views worth surfacing (DD-2); empty until
            computed. Filled by CAC-05.
        manifest: The auditable execution + provenance receipt for the run
            (CAC-04, :class:`conclave.manifest.ModelHarnessManifest`): what ran
            (providers considered/called/skipped, resolved model ids, generation
            settings, per-member receipts, latency, token usage) and how the
            verdict was made (verdict-extraction provenance, ``verdict_type``,
            ``consensus_method``, verdict-absent reason — the verdict-provenance
            slots are populated later by CAC-05). ``None`` only on a result
            constructed without one (e.g. a bare ``CouncilResult(prompt=...)``);
            a live ``synthesize``/``raw`` council run always attaches one,
            including the zero-members path. No secrets, ever.

    Properties:
        member_answers: Read-only alias for ``answers`` (the per-member raw
            responses), exposed under the contract-v2 name. Returns the same list
            object as ``answers``; there is one underlying field.
        degraded: Computed field (DSE-901) -- ``True`` when the run produced at
            least one usable member answer but the judge/synthesizer step failed,
            so a caller cannot mistake a partial run for a clean pass. See the
            property docstring below for the exact rule and the CLI's exit-code
            contract (:func:`conclave.cli.ask`) that keys off it.
        primary_failed_over: Computed field (DSE-1512) -- ``True`` when the
            chain's primary adjudicator failed for an infrastructure reason,
            whether or not a successor then answered. Independent of
            ``degraded``: a successor adjudication is ``primary_failed_over=True,
            degraded=False``. See the property docstring below for the exact
            rule and why the cache never stores such a run.
    """

    prompt: str
    mode: str = "synthesize"
    answers: list[ModelAnswer] = Field(default_factory=list)
    synthesizer: str | None = None
    synthesizer_model_id: str | None = None
    synthesis: str | None = None
    synthesis_error: str | None = None
    skipped: list[str] = Field(default_factory=list)
    rounds: list[DebateRound] = Field(default_factory=list)
    adversarial: AdversarialResult | None = None
    vote: VoteResult | None = None
    elite: EliteResult | None = None
    cached: bool = False
    converged: bool = False
    convergence_score: float | None = None
    prompt_version: str = Field(default_factory=_default_prompt_version)
    # CAC-01 result contract v2 — adjudication layer. All default to None/empty;
    # CAC-05 fills them (no consensus/disagreement computation happens here).
    verdict: CouncilVerdict | None = None
    consensus_score: float | None = None
    consensus_method: str | None = None
    consensus_label: str | None = None
    conflicts: list[CouncilConflict] = Field(default_factory=list)
    provider_votes: list[ProviderVote] = Field(default_factory=list)
    minority_reports: list[MinorityReport] = Field(default_factory=list)
    # CAC-04 auditable manifest. Optional/backward-compatible: a bare
    # CouncilResult(prompt=...) still validates with manifest=None; a live
    # synthesize/raw council run always attaches one (assembled in council.py).
    manifest: ModelHarnessManifest | None = None

    @property
    def successful_answers(self) -> list[ModelAnswer]:
        """Members that returned a usable answer."""
        return [a for a in self.answers if a.ok]

    @property
    def failed_answers(self) -> list[ModelAnswer]:
        """Members that were attempted but errored."""
        return [a for a in self.answers if not a.ok]

    @property
    def member_answers(self) -> list[ModelAnswer]:
        """Contract-v2 alias for :attr:`answers` (the per-member raw responses)."""
        return self.answers

    @computed_field  # type: ignore[prop-decorator]
    @property
    def degraded(self) -> bool:
        """True when members answered but the judge/synthesizer step failed (DSE-901).

        Distinguishes "partial run: judge/synthesis failed" from a clean pass, so
        a verification gate that only checks ``CouncilResult`` fields (or the
        CLI's exit code -- see :func:`conclave.cli.ask`) cannot read a degraded
        run as a full success. This closed a real gap: on 2026-07-25 an
        Anthropic credit failure meant 4/5 members answered but the ``claude``
        judge/synthesizer failed, ``adversarial.verdict`` was ``null``, and the
        CLI still exited 0.

        ``True`` whenever either of the two fields the judge/synthesizer path
        writes on failure is set:

        * ``synthesis_error`` is non-``None`` (``synthesize``/``debate``/``elite``
          modes, and mirrored into ``adversarial`` mode's top-level fields too); or
        * ``adversarial.verdict_error`` is non-``None`` (checked directly as a
          defense-in-depth fallback in case a future code path sets it without
          mirroring to ``synthesis_error``).

        Always ``False`` for ``raw``/``vote`` runs, where the judge/synthesizer is
        never invoked, and for any run where it ran and succeeded. Because
        ``synthesis_error`` is also set when a run has zero usable member answers
        (nothing to synthesize), that harder failure is *also* ``degraded=True``
        here; the CLI's exit-code contract gives that case its own, more severe
        exit code, so this flag alone does not distinguish "nothing answered"
        from "some members answered, judge/synthesis failed" -- callers wanting
        that distinction should also check ``successful_answers``. Included as a
        top-level key in ``model_dump(mode="json")`` output (a Pydantic
        ``computed_field``), so a scripted consumer can check it directly instead
        of reaching into ``synthesis_error``/``adversarial``.
        """
        if self.synthesis_error:
            return True
        if self.adversarial is not None and self.adversarial.verdict_error:
            return True
        return False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def primary_failed_over(self) -> bool:
        """True when the primary adjudicator was replaced (DSE-1512, review-corrected).

        ``True`` when ``self.manifest`` is not ``None`` and the succession
        ledger for some adjudication role (synthesis, debate's final
        consolidation, the adversarial judge, or verdict extraction) shows the
        primary candidate did NOT itself produce the answer. Three cases, all
        ``True``:

        1. **Infrastructure failure of the primary.** Any
           ``manifest.adjudication_succession`` attempt has ``outcome ==
           "failed_over"`` -- the primary failed for an infrastructure reason
           (auth/quota/5xx/timeout/network, see :data:`FAILOVER_CATEGORIES`)
           and a successor then answered.
        2. **Ladder exhausted.** Any attempt has ``outcome == "exhausted"`` --
           every keyed candidate failed for an infrastructure reason and
           nothing answered.
        3. **Successor after a skipped-unkeyed primary.** A candidate at
           ``attempt_index > 1`` has ``outcome == "success"`` -- the primary
           was skipped for a missing key (``"skipped_unkeyed"``, which is
           itself not a failure category eligible for cases 1/2) and a later,
           keyed candidate adjudicated instead. This closed a real gap: an
           unkeyed primary with a keyed successor used to report ``False``
           (no ``"failed_over"``/``"exhausted"`` entry exists for that shape)
           even though the primary plainly did not adjudicate.

        ``False`` for a chain of one whose sole candidate was skipped for a
        missing key (ledger ``["skipped_unkeyed"]`` only, nothing left to
        succeed to -- unchanged v1.3.0 chain-of-one behavior, still
        cacheable), for a chain of one that never had an infrastructure
        failure, and for any run with no manifest.

        Independent of :attr:`degraded`: a run adjudicated by a successor is a
        clean run (``primary_failed_over=True, degraded=False``), while a run
        where the whole chain was exhausted is both
        (``primary_failed_over=True, degraded=True``). The two flags answer
        different questions -- "did the judge/synthesis step ultimately
        produce a usable result?" (``degraded``) versus "did the primary
        candidate itself need to be replaced or exhausted?"
        (``primary_failed_over``).

        A run for which this is ``True`` is never written to the result cache
        (see :meth:`conclave.council.Council._cached_run`): a cache hit must
        never pin a result the primary did not produce, nor replay an
        infrastructure outage after it has cleared. Included as a top-level
        key in ``model_dump(mode="json")`` output (a Pydantic
        ``computed_field``), so a scripted consumer can check it directly
        instead of walking the manifest's succession ledger.
        """
        if self.manifest is None:
            return False
        attempts = self.manifest.adjudication_succession
        if any(attempt.outcome in ("failed_over", "exhausted") for attempt in attempts):
            return True
        # A successor adjudicated after the primary was skipped for a missing key.
        return any(
            attempt.outcome == "success" and attempt.attempt_index > 1 for attempt in attempts
        )


# Late import (see the note near the top): ``manifest`` imports ``TokenUsage``
# from this module, so it can only be imported once the leaf types above exist.
# Importing it here makes ``ModelHarnessManifest`` available in this module's
# namespace for the ``CouncilResult.manifest`` annotation resolved by
# ``model_rebuild()`` below.
from .manifest import ModelHarnessManifest  # noqa: E402 -- late import breaks an import cycle

# ``StreamEvent.result`` forward-references ``CouncilResult`` (defined after it
# under ``from __future__ import annotations``); resolve that ref now that the
# class exists so ``StreamEvent`` validates correctly.
StreamEvent.model_rebuild()

# ``CouncilResult`` references the verdict types (imported at runtime from
# ``.verdict``) only via string annotations under ``from __future__ import
# annotations``. Rebuild it so Pydantic resolves those references eagerly rather
# than on first validation — belt-and-suspenders for the CAC-01 additions.
CouncilResult.model_rebuild()
