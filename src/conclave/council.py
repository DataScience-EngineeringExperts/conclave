"""The Council: concurrent multi-model fan-out plus synthesis.

``Council`` is the primary importable entry point. It resolves friendly names to
LiteLLM model ids, skips any member whose API key is absent, fans the prompt out
concurrently, collects partial results even when some members fail, and (in
synthesize mode) asks a synthesizer model to merge the answers into one.
"""

from __future__ import annotations

import asyncio

from .config import ConclaveConfig, load_config
from .logging import get_logger
from .models import CouncilResult, ModelAnswer
from .providers import call_model
from .registry import key_present

logger = get_logger("council")

_SYNTH_SYSTEM = (
    "You are the synthesizer of a council of AI models. You are given the same "
    "user prompt that was posed to several models, plus each model's answer. "
    "Produce one consolidated, accurate answer. Reconcile agreements, surface "
    "and adjudicate disagreements, and note any answer that is clearly wrong. "
    "Do not invent a model's position; rely only on the answers provided."
)


class Council:
    """A council of foundation models with an optional synthesizer.

    Args:
        models: Friendly names (or raw LiteLLM ids) of council members.
        synthesizer: Friendly name of the synthesizer model. If ``None``, the
            config default is used.
        config: Pre-loaded config; if ``None``, loaded from disk + defaults.
        temperature: Sampling temperature for member calls.
        timeout: Per-call timeout in seconds.

    Example:
        >>> council = Council(models=["grok", "perplexity"], synthesizer="claude")
        >>> result = council.ask_sync("What is the capital of France?")
        >>> print(result.synthesis)
    """

    def __init__(
        self,
        models: list[str],
        synthesizer: str | None = None,
        config: ConclaveConfig | None = None,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> None:
        self.config = config or load_config()
        self.requested_models = list(models)
        self.synthesizer = synthesizer or self.config.synthesizer
        self.temperature = temperature
        self.timeout = timeout

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

    async def ask(self, prompt: str, synthesize: bool = True) -> CouncilResult:
        """Run the council asynchronously.

        Args:
            prompt: The user prompt to fan out.
            synthesize: When True (default), merge answers via the synthesizer.

        Returns:
            A :class:`CouncilResult` with per-member answers and (optionally) the
            synthesis. A run with zero available members returns an empty-answer
            result rather than raising.
        """
        members, skipped = self._available_members()
        result = CouncilResult(prompt=prompt, skipped=skipped)

        if not members:
            logger.warning("no council members have keys available; nothing to run")
            return result

        messages = [{"role": "user", "content": prompt}]
        tasks = [
            call_model(
                name,
                model_id,
                messages,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            for name, model_id in members
        ]
        # return_exceptions=True is belt-and-suspenders; call_model already
        # converts provider failures into ModelAnswer.error, but this guards
        # against any unexpected raise so one bad member can't abort the gather.
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for (name, model_id), outcome in zip(members, gathered):
            if isinstance(outcome, ModelAnswer):
                result.answers.append(outcome)
            else:
                logger.warning("%s raised unexpectedly: %s", name, outcome)
                result.answers.append(
                    ModelAnswer(
                        name=name,
                        model_id=model_id,
                        error=f"{type(outcome).__name__}: {outcome}",
                    )
                )

        if synthesize:
            await self._synthesize(result)
        return result

    async def _synthesize(self, result: CouncilResult) -> None:
        """Run the synthesizer over the successful answers, mutating ``result``."""
        usable = result.successful_answers
        if not usable:
            result.synthesis_error = "no successful member answers to synthesize"
            logger.warning(result.synthesis_error)
            return

        synth_id = self.config.resolve_model_id(self.synthesizer)
        result.synthesizer = self.synthesizer
        result.synthesizer_model_id = synth_id

        if not key_present(synth_id):
            result.synthesis_error = (
                f"synthesizer '{self.synthesizer}' ({synth_id}) has no API key; "
                "returning raw answers only"
            )
            logger.warning(result.synthesis_error)
            return

        blocks = "\n\n".join(
            f"### Answer from {a.name} ({a.model_id})\n{a.answer}" for a in usable
        )
        synth_messages = [
            {"role": "system", "content": _SYNTH_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original prompt:\n{result.prompt}\n\n"
                    f"Council answers:\n\n{blocks}\n\n"
                    "Now produce the consolidated answer."
                ),
            },
        ]
        synth_answer = await call_model(
            self.synthesizer,
            synth_id,
            synth_messages,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        if synth_answer.ok:
            result.synthesis = synth_answer.answer
        else:
            result.synthesis_error = synth_answer.error

    def ask_sync(self, prompt: str, synthesize: bool = True) -> CouncilResult:
        """Synchronous wrapper around :meth:`ask`.

        Safe to call from non-async code. Raises ``RuntimeError`` if invoked
        from inside a running event loop -- use :meth:`ask` there instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask(prompt, synthesize=synthesize))
        raise RuntimeError(
            "ask_sync() called from within a running event loop; await ask() instead"
        )
