"""Streaming orchestration for the synthesize/raw path (issue #7).

This module owns the council-level streaming logic so :mod:`conclave.council`
stays focused on the buffered modes. :func:`stream_ask` is the single
async-generator engine behind :meth:`conclave.council.Council.ask_stream`: it

* fans the prompt out to every available member **concurrently**, interleaving
  each member's incremental text into one flat :class:`conclave.models.StreamEvent`
  sequence (``member_delta`` / ``member_done``),
* optionally streams the synthesizer over the successful answers
  (``synthesis_delta`` / ``synthesis_done``), and
* emits a terminal ``done`` event carrying the fully-assembled
  :class:`conclave.models.CouncilResult` whose shape is **byte-for-byte
  identical** to the non-streaming :meth:`Council.ask` result -- so downstream
  consumers (and the cache) are unaffected.

Streaming applies to the synthesize/raw path only; ``debate``/``adversarial``
are intentionally out of scope for this issue.

Interleaving is done with an :class:`asyncio.Queue`: each member runs as its own
task draining :func:`conclave.providers.call_model_stream` and pushing events
onto the queue, while the generator yields whatever arrives first. This gives
true concurrency (slow members do not block fast ones) with deterministic
final assembly (the final :class:`CouncilResult.answers` list is ordered by the
members list, not by completion order).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .adapters.base import redact
from .logging import get_logger
from .models import CouncilResult, ModelAnswer, StreamEvent
from .providers import call_model_stream
from .registry import key_present

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from .council import Council

logger = get_logger("streaming")

# Sentinel pushed onto the queue when a single member's stream is exhausted, so
# the consumer knows when all members have finished without polling task state.
_MEMBER_DONE = object()


async def _drive_member(
    council: Council,
    name: str,
    model_id: str,
    messages: list[dict[str, str]],
    queue: asyncio.Queue,
) -> None:
    """Drain one member's token stream onto the shared queue.

    Pushes ``("delta", StreamEvent)`` for each text chunk and a final
    ``("answer", ModelAnswer)`` carrying the assembled answer, then the
    ``_MEMBER_DONE`` sentinel. Never raises: :func:`call_model_stream` already
    captures provider failures onto the final ``ModelAnswer`` (partial text
    preserved), and any unexpected error here is converted to an error answer so
    one bad member can never wedge the queue or abort the run.
    """
    try:
        async for item in call_model_stream(
            name,
            model_id,
            messages,
            temperature=council.temperature,
            timeout=council.timeout,
            config=council.config,
        ):
            if isinstance(item, ModelAnswer):
                await queue.put(("answer", item))
            else:
                await queue.put(
                    (
                        "delta",
                        StreamEvent(type="member_delta", name=name, model_id=model_id, text=item),
                    )
                )
    except Exception as exc:  # noqa: BLE001 -- a member must never wedge the run
        # call_model_stream already redacts and never raises, so this arm only
        # fires on an UNEXPECTED escape. Redact the exception text anyway so the
        # "every surfaced error string is scrubbed" invariant holds even on this
        # defense-in-depth path (key-leak audit, vector 2).
        message = redact(f"{type(exc).__name__}: {exc}")
        logger.warning("%s streaming raised unexpectedly: %s", name, message)
        await queue.put(
            (
                "answer",
                ModelAnswer(name=name, model_id=model_id, error=message),
            )
        )
    finally:
        await queue.put((_MEMBER_DONE, None))


async def stream_ask(
    council: Council, prompt: str, synthesize: bool = True
) -> AsyncIterator[StreamEvent]:
    """Stream a synthesize/raw council run as a flat :class:`StreamEvent` sequence.

    See the module docstring and :meth:`conclave.council.Council.ask_stream`.

    Args:
        council: The :class:`Council` whose members/synthesizer/config drive the
            run.
        prompt: The user prompt to fan out.
        synthesize: When ``True`` (default), stream the synthesizer over the
            successful member answers after every member finishes.

    Yields:
        ``member_delta`` / ``member_done`` events per member (interleaved),
        then ``synthesis_delta`` / ``synthesis_done`` when synthesis runs, then
        a terminal ``done`` event whose ``result`` is the full
        :class:`CouncilResult`.
    """
    members, skipped = council._available_members()
    result = CouncilResult(
        prompt=prompt,
        mode="synthesize" if synthesize else "raw",
        skipped=skipped,
    )

    if not members:
        logger.warning("no council members have keys available; nothing to stream")
        yield StreamEvent(type="done", result=result)
        return

    base_messages = [{"role": "user", "content": prompt}]
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_drive_member(council, name, model_id, base_messages, queue))
        for name, model_id in members
    ]

    # Collect finished answers keyed by name so the final list can be reordered
    # to match the members list (deterministic shape) regardless of arrival.
    by_name: dict[str, ModelAnswer] = {}
    remaining = len(members)
    try:
        while remaining > 0:
            kind, payload = await queue.get()
            if kind == "delta":
                yield payload
            elif kind == "answer":
                by_name[payload.name] = payload
                yield StreamEvent(
                    type="member_done",
                    name=payload.name,
                    model_id=payload.model_id,
                    answer=payload,
                )
            elif kind is _MEMBER_DONE:
                remaining -= 1
    finally:
        # Ensure every member task is awaited even if the consumer stops early.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Reassemble answers in members order for a stable, non-streaming-equivalent
    # CouncilResult. A member that produced no answer event (should not happen --
    # the driver always emits one) is recorded as an explicit error.
    result.answers = [
        by_name.get(name) or ModelAnswer(name=name, model_id=model_id, error="no answer produced")
        for name, model_id in members
    ]

    if synthesize:
        async for event in _stream_synthesis(council, result):
            yield event

    yield StreamEvent(type="done", result=result)


async def _stream_synthesis(council: Council, result: CouncilResult) -> AsyncIterator[StreamEvent]:
    """Stream the synthesizer over ``result``'s successful answers, mutating it.

    Mirrors :meth:`Council._synthesize` (no-usable-answers and no-key short
    circuits set ``synthesis_error`` exactly the same way), but streams the
    synthesizer's tokens as ``synthesis_delta`` events and finishes with a
    ``synthesis_done`` event. On any short circuit nothing is yielded (there is
    no live token stream) -- the reason lands on ``result.synthesis_error`` and
    is visible in the terminal ``done`` event.
    """
    from .council import _SYNTH_SYSTEM

    usable = result.successful_answers
    if not usable:
        result.synthesis_error = "no successful member answers to synthesize"
        logger.warning(result.synthesis_error)
        return

    synth_id = council.config.resolve_model_id(council.synthesizer)
    result.synthesizer = council.synthesizer
    result.synthesizer_model_id = synth_id

    if not key_present(synth_id):
        result.synthesis_error = (
            f"synthesizer '{council.synthesizer}' ({synth_id}) has no API key; "
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
    messages = [
        {"role": "system", "content": _SYNTH_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    final: ModelAnswer | None = None
    async for item in call_model_stream(
        council.synthesizer,
        synth_id,
        messages,
        temperature=council.temperature,
        timeout=council.timeout,
        config=council.config,
    ):
        if isinstance(item, ModelAnswer):
            final = item
        else:
            yield StreamEvent(
                type="synthesis_delta",
                name=council.synthesizer,
                model_id=synth_id,
                text=item,
            )

    if final is not None and final.ok:
        result.synthesis = final.answer
    elif final is not None:
        result.synthesis_error = final.error
    if final is not None:
        yield StreamEvent(
            type="synthesis_done",
            name=council.synthesizer,
            model_id=synth_id,
            answer=final,
        )
