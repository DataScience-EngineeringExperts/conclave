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

The terminal ``done`` result also carries the auditable
:class:`conclave.manifest.ModelHarnessManifest` (CAC-04) and the structured
verdict (CAC-05/CAC-06) -- making the "byte-for-byte identical to non-streaming"
claim literally true rather than aspirational (CAC-06-STREAM). Both are produced
by reusing :meth:`Council._build_manifest` and the single shared
:meth:`Council._apply_verdict` helper, so the streaming and buffered paths cannot
drift: the verdict object stays canonical and the manifest's verdict-provenance
slots are populated exactly once, in one place. The assembly order mirrors
:meth:`Council._ask_uncached` exactly -- build the manifest after the answers are
reassembled, then (synthesize mode only) stream the synthesizer and apply the
verdict -- so manifest -> synthesize -> verdict holds on both paths.

The empty-members early return mirrors :meth:`Council._ask_uncached`'s memberless
return: it attaches a manifest (full skip list, no receipts, VERIFIED stamp) but
deliberately attaches **no verdict** -- a council with zero responders has nothing
to adjudicate, and ``_apply_verdict`` is never reached in the buffered path
either, so the streamed memberless ``done`` carries ``verdict=None`` for parity.

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
        :class:`CouncilResult`. That terminal result carries the auditable
        :class:`conclave.manifest.ModelHarnessManifest` and -- in synthesize mode
        -- the structured verdict, identical to what :meth:`Council.ask` would
        return for the same inputs (CAC-06-STREAM).

    Assembly mirrors :meth:`Council._ask_uncached` exactly so the two paths cannot
    drift:

    * **Empty members.** A manifest is attached (full skip list, no receipts,
      VERIFIED stamp) and a ``done`` event is emitted with **no verdict** -- a
      memberless council has nothing to adjudicate, matching the buffered path,
      which returns before reaching :meth:`Council._apply_verdict`.
    * **Normal path.** After the answers are reassembled in members order the
      manifest is built (on BOTH raw and synthesize, exactly like the buffered
      path builds it before its ``if synthesize`` block). Then, in synthesize
      mode only, the synthesizer is streamed and -- after its stream completes --
      the shared :meth:`Council._apply_verdict` runs, populating
      ``result.verdict`` plus the manifest's verdict-provenance slots. Raw mode
      never extracts a verdict (no synthesizer call), again matching the buffered
      path. Final order: build manifest -> stream synthesis -> apply verdict ->
      emit ``done``.

    Never-raises, secret-safety, and partial-text-on-member-failure behavior are
    unchanged: :meth:`Council._apply_verdict` and :meth:`Council._build_manifest`
    never raise and only ever attach secret-free, VERIFIED-stamped content.
    """
    members, skipped = council._available_members()
    result = CouncilResult(
        prompt=prompt,
        mode="synthesize" if synthesize else "raw",
        skipped=skipped,
    )

    if not members:
        logger.warning("no council members have keys available; nothing to stream")
        # Mirror Council._ask_uncached's memberless return: attach a manifest
        # (full skip list, no receipts, VERIFIED stamp) so the streamed ``done``
        # is auditable, but attach NO verdict. A council with zero responders has
        # nothing to adjudicate, and the buffered path returns before reaching
        # _apply_verdict too -- so a memberless ``done`` carries verdict=None on
        # both paths (CAC-06-STREAM parity).
        result.manifest = council._build_manifest(
            mode=result.mode, members=[], skipped=skipped, answers=[]
        )
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

    # Build the manifest after the answers exist, on BOTH raw and synthesize --
    # exactly like Council._ask_uncached builds it before its ``if synthesize``
    # block. _build_manifest never raises and stamps the manifest secret-safe.
    result.manifest = council._build_manifest(
        mode=result.mode, members=members, skipped=skipped, answers=result.answers
    )

    if synthesize:
        # Prose synthesis streams first, then the structured verdict over the SAME
        # answers -- mirroring _ask_uncached's synthesize -> apply_verdict order.
        # _apply_verdict lives INSIDE this block (raw mode never extracts a
        # verdict, since it makes no synthesizer call) and runs AFTER the synthesis
        # stream completes so it can populate the now-existing manifest's
        # verdict-provenance slots. It is opt-out via the constructor flag and a
        # no-op when disabled, never raises, and only attaches secret-free content.
        async for event in _stream_synthesis(council, result):
            yield event
        # Streaming synthesis has its own token transport and does not yet emit a
        # synthesis receipt. Preserve the established streaming manifest as a
        # member-call ledger rather than appending only half of the downstream
        # call sequence. Complete receipt capture is currently the buffered/Elite
        # contract.
        await council._apply_verdict(result, record_receipts=False)

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
