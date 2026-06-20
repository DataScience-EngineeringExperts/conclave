"""Pydantic data models for conclave configuration and results.

These are the stable, importable contract used by both the CLI and any
downstream library consumer (e.g. mcp-warden). Keep field names stable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .verdict import (
    CouncilConflict,
    CouncilVerdict,
    MinorityReport,
    ProviderVote,
)


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

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelAnswer(BaseModel):
    """One council member's response (or failure).

    Attributes:
        name: Friendly council member name (e.g. ``"grok"``).
        model_id: Resolved provider-prefixed model id (e.g. ``"xai/grok-4.3"``).
        answer: The raw text answer, or ``None`` if the call failed.
        latency_s: Wall-clock seconds for the call.
        usage: Token usage if reported by the provider.
        error: Error message if the call failed, else ``None``.
        answer_id: Stable id assigned by conclave (not emitted by the model), used
            to back ``evidence_answer_ids`` on the verdict's positions/conflicts
            (CAC-01 result contract v2). ``None`` until conclave assigns it.
        warnings: Non-fatal notes about this answer (e.g. structured-output repair
            applied). Empty by default. Distinct from ``error``, which marks the
            whole call as failed.
    """

    name: str
    model_id: str
    answer: str | None = None
    latency_s: float = 0.0
    usage: TokenUsage | None = None
    error: str | None = None
    answer_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the member returned a usable answer."""
        return self.error is None and self.answer is not None

    @property
    def latency_ms(self) -> float:
        """Call latency in milliseconds, derived from :attr:`latency_s`."""
        return self.latency_s * 1000.0


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


class CouncilResult(BaseModel):
    """The full outcome of a council run.

    Attributes:
        prompt: The original user prompt.
        mode: The run mode that produced this result
            (``"synthesize"`` | ``"raw"`` | ``"debate"`` | ``"adversarial"``).
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

    Properties:
        member_answers: Read-only alias for ``answers`` (the per-member raw
            responses), exposed under the contract-v2 name. Returns the same list
            object as ``answers``; there is one underlying field.
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
    cached: bool = False
    converged: bool = False
    convergence_score: float | None = None
    prompt_version: str = Field(default_factory=_default_prompt_version)
    # CAC-01 result contract v2 — adjudication layer. All default to None/empty;
    # CAC-05 fills them (no consensus/disagreement computation happens here).
    # NOTE: CAC-04 adds the `manifest: ModelHarnessManifest` field here.
    verdict: CouncilVerdict | None = None
    consensus_score: float | None = None
    consensus_method: str | None = None
    consensus_label: str | None = None
    conflicts: list[CouncilConflict] = Field(default_factory=list)
    provider_votes: list[ProviderVote] = Field(default_factory=list)
    minority_reports: list[MinorityReport] = Field(default_factory=list)

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


# ``StreamEvent.result`` forward-references ``CouncilResult`` (defined after it
# under ``from __future__ import annotations``); resolve that ref now that the
# class exists so ``StreamEvent`` validates correctly.
StreamEvent.model_rebuild()

# ``CouncilResult`` references the verdict types (imported at runtime from
# ``.verdict``) only via string annotations under ``from __future__ import
# annotations``. Rebuild it so Pydantic resolves those references eagerly rather
# than on first validation — belt-and-suspenders for the CAC-01 additions.
CouncilResult.model_rebuild()
