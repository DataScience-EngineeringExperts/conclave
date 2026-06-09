"""The Council: concurrent multi-model fan-out plus synthesis.

``Council`` is the primary importable entry point. It resolves friendly names to
provider-prefixed model ids, skips any member whose API key is absent, fans the prompt out
concurrently, collects partial results even when some members fail, and (in
synthesize mode) asks a synthesizer model to merge the answers into one.

The deliberation modes (``debate``, ``adversarial``) live in :mod:`conclave.modes`
and reuse this class's :meth:`Council.fan_out` primitive so the partial-failure
handling is written exactly once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from . import cache as cache_mod
from . import transport
from .config import ConclaveConfig, load_config
from .logging import get_logger
from .models import CouncilResult, ModelAnswer
from .providers import call_model
from .registry import key_present

logger = get_logger("council")

# A per-member message-list factory: given a (friendly_name, model_id) member,
# return the OpenAI-style messages to send it. Lets each mode tailor the prompt
# per member while sharing Council.fan_out's concurrency + partial-failure code.
MessagesFor = Callable[[str, str], list[dict[str, str]]]

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
        models: Friendly names (or raw provider-prefixed model ids) of council members.
        synthesizer: Friendly name of the synthesizer model. If ``None``, the
            config default is used.
        config: Pre-loaded config; if ``None``, loaded from disk + defaults.
        temperature: Sampling temperature for member calls.
        timeout: Per-call timeout in seconds.
        cache: Opt-in result cache. ``None`` (default) defers to
            ``config.cache`` (off unless enabled in ``~/.conclave/config.yml``);
            ``True``/``False`` overrides it for this council. When enabled, an
            identical repeat run is served from the on-disk cache instead of
            re-calling the providers. The cache never persists API keys --
            see :mod:`conclave.cache`.

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
        cache: bool | None = None,
    ) -> None:
        self.config = config or load_config()
        self.requested_models = list(models)
        self.synthesizer = synthesizer or self.config.synthesizer
        self.temperature = temperature
        self.timeout = timeout
        # Explicit override wins; otherwise defer to config (off by default).
        self.cache_enabled = self.config.cache if cache is None else cache

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

    def _cache_key(
        self,
        prompt: str,
        mode: str,
        *,
        rounds: int | None = None,
        proposer: str | None = None,
        converge_threshold: float | None = None,
    ) -> str:
        """Build the cache key for a run from the resolved, secret-free identity.

        Uses the *resolved* member ids and the synthesizer/judge identity so two
        runs collide only when they would genuinely produce equivalent output.
        Members that would be skipped for a missing key are excluded -- a cache
        entry reflects the council that actually ran, so a key reappearing later
        produces the same membership. No environment value is read here.
        """
        members, _skipped = self._available_members()
        synth_id = self.config.resolve_model_id(self.synthesizer)
        return cache_mod.make_key(
            prompt=prompt,
            mode=mode,
            members=members,
            synthesizer=self.synthesizer,
            synthesizer_model_id=synth_id,
            temperature=self.temperature,
            rounds=rounds,
            proposer=proposer,
            converge_threshold=converge_threshold,
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
    ) -> CouncilResult:
        """Serve ``run`` from the result cache when caching is enabled.

        On a hit the cached :class:`CouncilResult` is returned with ``cached=True``
        and the providers are not called. On a miss (or when caching is off) the
        live ``run`` executes; a successful live run is stored best-effort. Cache
        read/write failures never propagate -- they degrade to a normal live run.
        """
        if not self.cache_enabled:
            return await run()

        key = self._cache_key(
            prompt,
            mode,
            rounds=rounds,
            proposer=proposer,
            converge_threshold=converge_threshold,
        )
        hit = cache_mod.load(key)
        if hit is not None:
            logger.info("cache hit for %s run (%s)", mode, key[:12])
            return hit

        result = await run()
        cache_mod.store(key, result)
        return result

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
                temperature=self.temperature,
                timeout=self.timeout,
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
                logger.warning("%s raised unexpectedly: %s", name, outcome)
                answers.append(
                    ModelAnswer(
                        name=name,
                        model_id=model_id,
                        error=f"{type(outcome).__name__}: {outcome}",
                    )
                )
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

    async def _ask_uncached(self, prompt: str, synthesize: bool = True) -> CouncilResult:
        """The live ask path (no cache consultation). See :meth:`ask`."""
        members, skipped = self._available_members()
        result = CouncilResult(
            prompt=prompt,
            mode="synthesize" if synthesize else "raw",
            skipped=skipped,
        )

        if not members:
            logger.warning("no council members have keys available; nothing to run")
            return result

        base_messages = [{"role": "user", "content": prompt}]
        result.answers = await self.fan_out(members, lambda _name, _model_id: base_messages)

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

        blocks = "\n\n".join(f"### Answer from {a.name} ({a.model_id})\n{a.answer}" for a in usable)
        user_content = (
            f"Original prompt:\n{result.prompt}\n\n"
            f"Council answers:\n\n{blocks}\n\n"
            "Now produce the consolidated answer."
        )
        answer = await self.synthesize_blocks(_SYNTH_SYSTEM, user_content)
        if answer.ok:
            result.synthesis = answer.answer
        else:
            result.synthesis_error = answer.error

    async def synthesize_blocks(self, system_prompt: str, user_content: str) -> ModelAnswer:
        """Call the synthesizer model with an arbitrary system + user message.

        Shared by synthesize mode, debate's final consolidation, and the
        adversarial judge so the synthesizer call path (and its error capture)
        is written once. Callers are responsible for checking ``key_present``
        on the synthesizer beforehand when they need a distinct no-key message;
        this method still returns a ``ModelAnswer.error`` if the call fails.

        Args:
            system_prompt: System instruction for the synthesizer/judge.
            user_content: The user-role content (prompt + answers/critiques).

        Returns:
            A :class:`ModelAnswer` from the synthesizer model.
        """
        synth_id = self.config.resolve_model_id(self.synthesizer)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await call_model(
            self.synthesizer,
            synth_id,
            messages,
            temperature=self.temperature,
            timeout=self.timeout,
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
