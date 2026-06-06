"""Pydantic data models for conclave configuration and results.

These are the stable, importable contract used by both the CLI and any
downstream library consumer (e.g. mcp-warden). Keep field names stable.
"""

from __future__ import annotations

from typing import Optional

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
        model_id: Resolved LiteLLM model id (e.g. ``"xai/grok-4.3"``).
        answer: The raw text answer, or ``None`` if the call failed.
        latency_s: Wall-clock seconds for the call.
        usage: Token usage if reported by the provider.
        error: Error message if the call failed, else ``None``.
    """

    name: str
    model_id: str
    answer: Optional[str] = None
    latency_s: float = 0.0
    usage: Optional[TokenUsage] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True when the member returned a usable answer."""
        return self.error is None and self.answer is not None


class CouncilResult(BaseModel):
    """The full outcome of a council run.

    Attributes:
        prompt: The original user prompt.
        answers: One ``ModelAnswer`` per attempted council member.
        synthesizer: Friendly name of the synthesizer model, if synthesis ran.
        synthesizer_model_id: Resolved LiteLLM id of the synthesizer.
        synthesis: The merged consolidated answer, or ``None`` if not produced.
        synthesis_error: Error message if synthesis failed, else ``None``.
        skipped: Friendly names skipped because no key was available.
    """

    prompt: str
    answers: list[ModelAnswer] = Field(default_factory=list)
    synthesizer: Optional[str] = None
    synthesizer_model_id: Optional[str] = None
    synthesis: Optional[str] = None
    synthesis_error: Optional[str] = None
    skipped: list[str] = Field(default_factory=list)

    @property
    def successful_answers(self) -> list[ModelAnswer]:
        """Members that returned a usable answer."""
        return [a for a in self.answers if a.ok]

    @property
    def failed_answers(self) -> list[ModelAnswer]:
        """Members that were attempted but errored."""
        return [a for a in self.answers if not a.ok]
