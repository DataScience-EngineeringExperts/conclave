"""Pydantic data models for conclave configuration and results.

These are the stable, importable contract used by both the CLI and any
downstream library consumer (e.g. mcp-warden). Keep field names stable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    """

    name: str
    model_id: str
    answer: str | None = None
    latency_s: float = 0.0
    usage: TokenUsage | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the member returned a usable answer."""
        return self.error is None and self.answer is not None


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

    @property
    def successful_answers(self) -> list[ModelAnswer]:
        """Members that returned a usable answer."""
        return [a for a in self.answers if a.ok]

    @property
    def failed_answers(self) -> list[ModelAnswer]:
        """Members that were attempted but errored."""
        return [a for a in self.answers if not a.ok]
