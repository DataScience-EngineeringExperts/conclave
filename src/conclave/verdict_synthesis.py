"""Disagreement extraction + verdict synthesis engine (CAC-05).

Authoritative spec: ``03_DESIGN_DECISIONS_v1.1.md`` DD-1 (consensus method
``position_cluster_ratio_v1``) and DD-2 (verdict schema + verdict-absent rule),
plus ``00_SCOPE_PLAN.md`` §4 (the three honesty corrections).

This is a self-contained, council-agnostic engine. Given the prompt and the
council members' raw answers, it asks a synthesizer model to produce a structured
*judgment* (the clustering of stances, the conflicts, the votes), with at most one
repair call when the initial output is invalid, then
computes the consensus number **itself, deterministically**, from that clustering
— the model never emits the consensus score. CAC-06 wires :func:`extract_verdict`
into ``council.ask``; this module does not wire itself.

The auditability fix (DD-1, Scope Plan §4.1)
--------------------------------------------
Asking a model "how much do these answers agree?" is circular theatre. Instead
the model is asked only to CLUSTER each member's stance (the one irreducible
LLM-assisted step), and the consensus number is pure arithmetic over that
clustering via :mod:`conclave.agreement` (``|largest cluster| / |positioned
members|``). The number is therefore reproducible and auditable, never a value the
model could fabricate. The extraction schema (:func:`verdict_extraction_json_schema`)
carries no consensus field at all, so a model that smuggles one in is simply
ignored by the Pydantic validator. This module never imports difflib (the debate
``convergence_score`` text-similarity signal is a FORBIDDEN consensus measure,
DD-1) — agreement.py enforces the same rule with a guard test.

Structured-output strategy (native + prompt-level, belt-and-suspenders)
----------------------------------------------------------------------
CAC-06-PLUMB threaded an ``output_contract`` kwarg through
:func:`conclave.providers.call_model` to ``adapter.build_request``, so the engine
now requests structured output on TWO complementary layers:

1. **Native** — :func:`extract_verdict` builds one
   ``OutputContract(schema=verdict_extraction_json_schema(), schema_name=
   "VerdictExtraction", strict=True)`` and passes it to BOTH ``call_model`` calls
   (the initial extraction and the repair retry). Capable providers
   (OpenAI/Anthropic/Gemini) translate it to their provider-native surface
   (OpenAI ``response_format`` ``json_schema`` / Gemini ``responseSchema`` /
   Anthropic tool ``input_schema``) and ENFORCE the schema at decode time.
2. **Prompt-level (retained fallback)** — the same LCD JSON-Schema instruction and
   schema stay embedded in the system/user messages, and the JSON is still parsed
   out of ``ModelAnswer.answer`` (tolerating a ```json code fence) and validated by
   Pydantic (:class:`conclave.verdict.VerdictExtractionModel`) — no ``jsonschema``
   dependency. This is the belt-and-suspenders path for providers WITHOUT strict
   structured-output support: the adapter degrades gracefully (free prose) and
   warns, and the prompt-level parse/validate/repair below still produces a
   conforming verdict.

The native contract is ADDITIVE: it does not replace the parse/validate/repair
fallback, and the engine's failure behavior is unchanged (graceful
``verdict=None``).

Validate → repair-once → fallback (DD-2 verdict-absent rule)
------------------------------------------------------------
The extraction is parsed and validated. On a JSON or schema failure the engine
re-calls the model ONCE with the stringified errors appended, then validates
again. If it still fails (or the model returned an error / empty answer), the
verdict is absent (``verdict=None``) with a recorded reason — it NEVER raises. The
extractor's identity + prompt version is recorded as provenance on EVERY path
(success and all three absent paths).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from . import agreement
from .adapters.base import OutputContract, redact
from .logging import get_logger
from .manifest import ProviderExecutionReceipt, VerdictExtraction
from .models import ModelAnswer
from .providers import call_model, receipt_from_answer
from .verdict import (
    VERDICT_EXTRACTION_PROMPT_VERSION,
    VERDICT_SCHEMA_VERSION,
    CouncilConflict,
    CouncilVerdict,
    VerdictExtractionModel,
    verdict_extraction_json_schema,
)

__all__ = [
    "VERDICT_EXTRACTION_PROMPT_VERSION",
    "VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES",
    "VerdictSynthesisResult",
    "extract_verdict",
    "verdict_extraction_json_schema",
]

logger = get_logger("verdict_synthesis")

# Repair prompts are paid input. Bound validation detail by UTF-8 bytes so a
# malformed structured response cannot amplify into an unbounded second call.
VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES = 2048

# Recorded reasons for an absent verdict (DD-2 verdict-absent rule). Named as
# constants so the three call sites and the test suite share the exact strings.
_REASON_TOO_FEW = "fewer than 2 responding members"
_REASON_OPEN_ENDED = "open-ended prompt (no decision/review to adjudicate)"
_REASON_EXTRACTION_FAILED = "verdict extraction failed schema validation"


def _bounded_repair_error(detail: object) -> str:
    scrubbed = redact(str(detail))
    encoded = scrubbed.encode("utf-8")
    if len(encoded) <= VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES:
        return scrubbed
    return encoded[:VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES].decode("utf-8", errors="ignore")


# The extraction system prompt. Versioned via VERDICT_EXTRACTION_PROMPT_VERSION
# (bump that constant in verdict.py on any wording change here). It instructs the
# model to emit ONLY the structured judgment per the schema and — load-bearing for
# DD-1 — to NOT emit any consensus score (the engine computes it deterministically).
_EXTRACTION_SYSTEM = (
    "You are the verdict extractor for an auditable multi-model council. You are "
    "given the original prompt and each council member's answer, labeled with a "
    "stable answer id. These IDs provide within-run answer provenance only; they "
    "do not prove claims against external sources. Produce ONE JSON object that "
    "conforms exactly to the "
    "provided JSON Schema and nothing else.\n\n"
    "Your job is to ADJUDICATE, not to re-answer:\n"
    "- Set verdict_applies=false when the prompt is open-ended generation (a poem, "
    "a brainstorm, free writing) with no decision or review to settle; otherwise "
    "true.\n"
    "- Set verdict_type to 'decision' (a question with an answer), 'review' (an "
    "accept/revise/reject judgment), or 'synthesis' (open-ended consolidation).\n"
    "- Cluster the members into positions[]; each position lists the providers in "
    "it and the evidence_answer_ids (the compatibility field containing the answer "
    "IDs shown) tracing it, so a human can verify every assignment against the raw "
    "answer.\n"
    "- Record one provider_vote per member that took a stance (position_label must "
    "match a positions[].label); omit a member that took no clean stance.\n"
    "- Add conflicts[] only when there are two or more positions in tension.\n\n"
    "CRITICAL: Do NOT emit any consensus score, percentage, ratio, or agreement "
    "number — the council computes consensus deterministically from your "
    "clustering. Emit only the fields in the schema."
)


class VerdictSynthesisResult(BaseModel):
    """The outcome of one verdict-extraction run (CAC-05 engine return type).

    A small, secret-free carrier returned by :func:`extract_verdict` on every
    path. The verdict is OPTIONAL (DD-2 verdict-absent rule): on the three absent
    paths ``verdict`` is ``None`` and ``verdict_absent_reason`` explains why, while
    ``extraction`` provenance is populated regardless. CAC-06 reads these three
    fields straight onto the :class:`conclave.models.CouncilResult` and the
    manifest.

    Attributes:
        verdict: The assembled :class:`conclave.verdict.CouncilVerdict` with the
            engine's deterministically-computed consensus values, or ``None`` when
            no verdict applies (see ``verdict_absent_reason``).
        extraction: The verdict-extraction provenance
            (:class:`conclave.manifest.VerdictExtraction`: extractor ``model_id``
            + ``prompt_version``). Populated on EVERY path — including the absent
            paths — so an audit can always see which extractor was consulted (or
            would have been). No secrets, ever.
        verdict_absent_reason: The recorded reason ``verdict`` is ``None``
            (``"fewer than 2 responding members"``, ``"open-ended prompt (no
            decision/review to adjudicate)"``, or ``"verdict extraction failed
            schema validation"``), or ``None`` when a verdict is present.
        attempt_receipts: One secret-free receipt for each actual extraction or
            repair call. The N<2 gate makes no call and therefore yields none.
    """

    verdict: CouncilVerdict | None = None
    extraction: VerdictExtraction
    verdict_absent_reason: str | None = Field(default=None)
    attempt_receipts: list[ProviderExecutionReceipt] = Field(default_factory=list)


def _verdict_attempt_receipt(
    answer: ModelAnswer,
    *,
    phase: str,
    attempt: int,
    schema_valid: bool | None,
    temperature: float,
    timeout: float,
    protocol_version: str | None,
) -> ProviderExecutionReceipt:
    """Build one secret-free receipt for an extraction or repair attempt."""
    if answer.error is not None:
        outcome = "failed"
        error_category = None
    elif schema_valid:
        outcome = "success"
        error_category = None
    else:
        outcome = "schema_invalid"
        error_category = "schema_validation"
    return receipt_from_answer(
        answer,
        temperature=temperature,
        timeout=timeout,
        phase=phase,
        attempt=attempt,
        outcome=outcome,
        error_category=error_category,
        schema_valid=schema_valid,
        protocol_version=protocol_version,
        prompt_version=VERDICT_EXTRACTION_PROMPT_VERSION,
        schema_version=VERDICT_SCHEMA_VERSION,
    )


def _responding(member_answers: list[ModelAnswer]) -> list[ModelAnswer]:
    """Return the members that produced a non-empty answer, in order.

    A member counts as responding when it has a non-blank ``answer`` and no
    ``error`` (``ModelAnswer.ok`` plus an explicit blank-text guard — a
    whitespace-only answer is not a real response). Failed/empty members are
    excluded from the N>=2 gate, the extraction prompt, and the consensus
    denominator (DD-1 partial-failure rule).

    Args:
        member_answers: All attempted council-member answers.

    Returns:
        The subset that genuinely responded, preserving input order.
    """
    return [a for a in member_answers if a.ok and a.answer and a.answer.strip()]


def _answer_label(answer: ModelAnswer, index: int) -> str:
    """Return a stable within-run provenance label for one responding member.

    Prefers the conclave-assigned ``answer_id`` (which backs
    ``evidence_answer_ids`` compatibility field on verdict positions). Successful
    answers normally always carry this ID. The positional fallback deliberately
    avoids exposing a provider/model name as a substitute identity.

    Args:
        answer: One responding member answer.
        index: Its 0-based position in the responding list (fallback id source).

    Returns:
        A non-empty label string the model can cite as an answer id.
    """
    return answer.answer_id or f"unidentified-answer-{index + 1}"


def _build_messages(prompt: str, responders: list[ModelAnswer]) -> list[dict[str, str]]:
    """Build the extraction messages from the prompt + every responding answer.

    Each responding member's answer is included verbatim, labeled with its stable
    answer id (:func:`_answer_label`) so the model can populate the compatibility
    field ``evidence_answer_ids``. The LCD extraction schema is embedded in the user
    message (prompt-level structured output — see the module docstring). Messages
    are built from answer TEXT + evidence labels only; the synthesizer call routes
    through :func:`conclave.providers.call_model`, which redacts, so no key
    material can enter the prompt.

    Args:
        prompt: The original user prompt the council answered.
        responders: The responding members (already filtered).

    Returns:
        An OpenAI-style ``[system, user]`` message list.
    """
    blocks = []
    for i, ans in enumerate(responders):
        label = _answer_label(ans, i)
        blocks.append(f"### Member answer (answer id: {label}) — from {ans.name}\n{ans.answer}")
    answers_block = "\n\n".join(blocks)

    schema_json = json.dumps(verdict_extraction_json_schema(), indent=2)
    user = (
        f"Original prompt:\n{prompt}\n\n"
        f"Council member answers:\n\n{answers_block}\n\n"
        "Extract the verdict as a single JSON object conforming exactly to this "
        f"JSON Schema (emit no prose, no consensus number, only the JSON):\n\n"
        f"{schema_json}"
    )
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": user},
    ]


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding Markdown code fence from a model answer, if present.

    Prompt-level structured output (Option A) means a model may wrap its JSON in a
    ```` ```json ... ``` ```` fence. This returns the inner body when a fence is
    detected (the first ``{`` through the last ``}``), else the trimmed text
    unchanged, so the JSON parser sees clean JSON either way.

    Args:
        text: The raw extractor answer text.

    Returns:
        The candidate JSON substring to parse.
    """
    trimmed = text.strip()
    if trimmed.startswith("```"):
        # Drop the opening fence line and any trailing fence.
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1 and end > start:
            return trimmed[start : end + 1]
    return trimmed


def _parse_and_validate(answer: ModelAnswer | None) -> tuple[VerdictExtractionModel | None, str]:
    """Parse + validate one extractor answer into a :class:`VerdictExtractionModel`.

    Returns ``(model, "")`` on success, or ``(None, errors)`` where ``errors`` is a
    short, redacted description of the JSON or schema failure suitable for both the
    repair prompt and a redacted warning log. Never raises: a missing/errored/empty
    answer, malformed JSON, or a Pydantic validation failure all become a
    ``(None, errors)`` pair (DD-2 never-raises contract).

    Args:
        answer: The extractor :class:`ModelAnswer`, or ``None``.

    Returns:
        ``(VerdictExtractionModel, "")`` on success, else ``(None, error_text)``.
    """
    if answer is None or answer.error or not answer.answer or not answer.answer.strip():
        detail = answer.error if (answer and answer.error) else "empty extractor response"
        return None, _bounded_repair_error(detail)

    candidate = _strip_code_fence(answer.answer)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, _bounded_repair_error(f"response was not valid JSON: {exc}")

    try:
        return VerdictExtractionModel.model_validate(data), ""
    except ValidationError as exc:
        return None, _bounded_repair_error(f"JSON did not match the verdict schema: {exc}")


def _member_vote_sequence(
    responders: list[ModelAnswer], extraction: VerdictExtractionModel
) -> list[str | None]:
    """Build the per-member position sequence for the consensus arithmetic (DD-1).

    For each responding member, look up its vote in
    ``extraction.provider_votes`` by provider name and take that vote's
    ``position_label``; a member with no matching vote contributes ``None`` (no
    clean stance → excluded from the denominator by
    :func:`conclave.agreement.consensus_score`). The vote index is keyed on the
    member's ``name`` because that is what the council passes as the
    ``ProviderVote.provider`` identity. This sequence — one entry per responding
    member, in council order — is exactly what ``consensus_score`` expects.

    Args:
        responders: The responding members (the denominator universe).
        extraction: The validated extraction carrying ``provider_votes``.

    Returns:
        One ``position_label`` (or ``None``) per responding member.
    """
    votes = {v.provider: v.position_label for v in extraction.provider_votes}
    return [votes.get(ans.name) for ans in responders]


def _conflict_score(full_sequence: list[str | None], position_labels: list[str]) -> float | None:
    """Recompute one conflict's consensus ratio over just its members (DD-1, DD-2).

    Per-conflict rule (a defensible reading of DD-2's "agreement on a specific
    sub-question"): take the sub-sequence of member votes whose ``position_label``
    is one of this conflict's ``position_labels``, then run the SAME
    largest-cluster ratio (:func:`conclave.agreement.consensus_score`) over that
    sub-population. So a conflict between two evenly-split camps scores ``0.5``
    (a tie on that sub-question), and a lopsided conflict scores higher — the
    number is pure arithmetic over the model's clustering, never model-emitted.
    Labels are matched case/whitespace-insensitively to mirror agreement.py's
    normalization, so a vote and a conflict label that differ only cosmetically
    still pair up.

    Args:
        full_sequence: Every responding member's vote label (or ``None``).
        position_labels: The labels named by this conflict.

    Returns:
        The per-conflict ratio, or ``None`` when fewer than two of the conflict's
        members expressed a position (agreement undefined).
    """
    wanted = {" ".join(label.split()).casefold() for label in position_labels}
    subset = [
        vote
        for vote in full_sequence
        if vote is not None and " ".join(vote.split()).casefold() in wanted
    ]
    return agreement.consensus_score(subset)


def _assemble_verdict(
    extraction: VerdictExtractionModel, responders: list[ModelAnswer]
) -> CouncilVerdict:
    """Assemble the final verdict: model judgment + engine-computed consensus.

    The model's judgment fields (``verdict_type``, ``headline``,
    ``recommendation``, ``positions``, ``provider_votes``, ``minority_reports``,
    ``caveats``, ``dissent_summary``) carry through verbatim. The consensus values
    are computed HERE (never read off the model): the top-level
    ``consensus_score``/``label`` from the full per-member vote sequence, and each
    conflict's ``consensus_score`` from its own sub-population
    (:func:`_conflict_score`). The conflict's other fields (topic, labels, summary)
    come from the model; only its score is recomputed so a model-supplied conflict
    score can never stand.

    Args:
        extraction: The validated model judgment.
        responders: The responding members (the consensus denominator universe).

    Returns:
        The assembled :class:`conclave.verdict.CouncilVerdict`.
    """
    sequence = _member_vote_sequence(responders, extraction)
    score, label = agreement.consensus(sequence)

    # Recompute every conflict's score from its own sub-population (overwriting any
    # value the model may have supplied — the number is never model-emitted).
    conflicts = [
        CouncilConflict(
            topic=c.topic,
            position_labels=c.position_labels,
            summary=c.summary,
            consensus_score=_conflict_score(sequence, c.position_labels),
        )
        for c in extraction.conflicts
    ]

    return CouncilVerdict(
        verdict_type=extraction.verdict_type,
        headline=extraction.headline,
        recommendation=extraction.recommendation,
        # Engine-computed consensus (DD-1) — the model never emitted these.
        consensus_score=score,
        consensus_method=agreement.CONSENSUS_METHOD,
        consensus_label=label,
        positions=extraction.positions,
        conflicts=conflicts,
        provider_votes=extraction.provider_votes,
        minority_reports=extraction.minority_reports,
        caveats=extraction.caveats,
        dissent_summary=extraction.dissent_summary,
    )


async def extract_verdict(
    prompt: str,
    member_answers: list[ModelAnswer],
    *,
    synthesizer_name: str,
    synthesizer_model_id: str,
    config=None,  # noqa: ANN001 -- ConclaveConfig | None; untyped to avoid an import edge
    temperature: float = 0.7,
    timeout: float = 120.0,
    protocol_version: str | None = None,
    call_model_func=None,  # noqa: ANN001 -- injectable async seam for guarded callers
) -> VerdictSynthesisResult:
    """Extract a structured, auditable verdict from a council's member answers.

    The CAC-05 engine. Flow (each step is an acceptance gate — see DD-1/DD-2):

    1. **Gate (N<2).** Count responding members (non-empty answer, no error). With
       fewer than two, return ``verdict=None`` with reason
       ``"fewer than 2 responding members"`` and NO LLM call (consensus is
       undefined for N<2, DD-1 edge case).
    2. **Extract.** Build the ``[system, user]`` messages from the prompt + every
       responding answer (labeled by within-run answer id) with the LCD extraction schema
       embedded, and make the initial :func:`conclave.providers.call_model` call.
    3. **Validate → repair-once → fallback.** Parse JSON + validate via Pydantic.
       On failure, re-call ONCE with the stringified errors appended; if it still
       fails (or the extractor errored / returned empty), return ``verdict=None``
       with reason ``"verdict extraction failed schema validation"``. Never raises.
    4. **Open-ended.** If the validated extraction has ``verdict_applies == False``,
       return ``verdict=None`` with reason
       ``"open-ended prompt (no decision/review to adjudicate)"``.
    5. **Compute + assemble.** Build the per-member vote sequence from
       ``provider_votes``, compute the consensus ratio/label deterministically via
       :mod:`conclave.agreement` (and each conflict's sub-ratio), and assemble the
       :class:`conclave.verdict.CouncilVerdict` (model judgment + computed
       consensus). The consensus number is NEVER emitted by the model.

    Extractor provenance (``synthesizer_model_id`` + the versioned prompt) is
    recorded on EVERY return path, including the absent ones.

    Args:
        prompt: The original user prompt the council answered.
        member_answers: One :class:`conclave.models.ModelAnswer` per attempted
            member (successes and failures); failures are filtered out.
        synthesizer_name: Friendly name of the extractor/synthesizer model.
        synthesizer_model_id: Resolved provider-prefixed id of the extractor (e.g.
            ``"anthropic/claude-sonnet-4"``). Recorded as provenance.
        config: Optional pre-resolved :class:`conclave.config.ConclaveConfig`
            threaded through to ``call_model`` (custom endpoints / no re-read).
        call_model_func: Optional async call seam matching :func:`call_model`.
            ``None`` preserves the normal module-level provider path. Guarded
            eval runners inject their reservation-aware gateway here so both the
            initial extraction and optional repair remain paid-call protected.

    Returns:
        A :class:`VerdictSynthesisResult`. On success ``verdict`` is populated and
        ``verdict_absent_reason`` is ``None``; on any of the three absent paths
        ``verdict`` is ``None`` with the reason recorded. ``extraction`` provenance
        is always populated. Never raises.

    Example:
        >>> # offline: tests patch conclave.verdict_synthesis.call_model
        >>> import asyncio
        >>> from conclave.models import ModelAnswer
        >>> answers = [ModelAnswer(name="a", model_id="a/m", answer="yes",
        ...                        answer_id="a-1")]
        >>> res = asyncio.run(extract_verdict(
        ...     "decide?", answers,
        ...     synthesizer_name="claude",
        ...     synthesizer_model_id="anthropic/claude-sonnet-4"))
        >>> res.verdict is None and res.verdict_absent_reason
        'fewer than 2 responding members'
    """
    # Provenance is recorded identically on every path (success and all absences).
    extraction_provenance = VerdictExtraction(
        model_id=synthesizer_model_id,
        prompt_version=VERDICT_EXTRACTION_PROMPT_VERSION,
    )

    # Step 1 — N<2 gate (no LLM call; consensus undefined for N<2, DD-1).
    responders = _responding(member_answers)
    if len(responders) < 2:
        return VerdictSynthesisResult(
            verdict=None,
            extraction=extraction_provenance,
            verdict_absent_reason=_REASON_TOO_FEW,
        )

    # Step 2 — initial extraction call. Request native structured output (capable
    # providers enforce the schema) AND keep the embedded schema in the messages as
    # the belt-and-suspenders fallback for providers without strict support. Built
    # once and reused for the initial call and the repair retry below; schema_name
    # matches the schema's "VerdictExtraction" title, strict requests native
    # enforcement where available.
    messages = _build_messages(prompt, responders)
    output_contract = OutputContract(
        schema=verdict_extraction_json_schema(),
        schema_name="VerdictExtraction",
        strict=True,
    )
    model_caller = call_model if call_model_func is None else call_model_func
    answer = await model_caller(
        synthesizer_name,
        synthesizer_model_id,
        messages,
        temperature=temperature,
        timeout=timeout,
        config=config,
        output_contract=output_contract,
    )

    # Step 3 — validate, then repair ONCE on failure, then fall back.
    extraction, errors = _parse_and_validate(answer)
    attempt_receipts = [
        _verdict_attempt_receipt(
            answer,
            phase="verdict_extraction",
            attempt=1,
            schema_valid=None if answer.error is not None else extraction is not None,
            temperature=temperature,
            timeout=timeout,
            protocol_version=protocol_version,
        )
    ]
    if extraction is None:
        repair_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous response could not be used. It must be a single "
                    "valid JSON object matching the schema exactly, with no prose "
                    "and no consensus number. The problem was:\n"
                    f"{errors}\n\n"
                    "Return only the corrected JSON object."
                ),
            }
        ]
        retry = await model_caller(
            synthesizer_name,
            synthesizer_model_id,
            repair_messages,
            temperature=temperature,
            timeout=timeout,
            config=config,
            output_contract=output_contract,
        )
        extraction, errors = _parse_and_validate(retry)
        attempt_receipts.append(
            _verdict_attempt_receipt(
                retry,
                phase="verdict_repair",
                attempt=2,
                schema_valid=None if retry.error is not None else extraction is not None,
                temperature=temperature,
                timeout=timeout,
                protocol_version=protocol_version,
            )
        )

    if extraction is None:
        # Repair exhausted — degrade gracefully (DD-2), never raise.
        logger.warning("verdict extraction failed schema validation after repair: %s", errors)
        return VerdictSynthesisResult(
            verdict=None,
            extraction=extraction_provenance,
            verdict_absent_reason=_REASON_EXTRACTION_FAILED,
            attempt_receipts=attempt_receipts,
        )

    # Step 4 — open-ended prompt → synthesis-only, no verdict (DD-2).
    if not extraction.verdict_applies:
        return VerdictSynthesisResult(
            verdict=None,
            extraction=extraction_provenance,
            verdict_absent_reason=_REASON_OPEN_ENDED,
            attempt_receipts=attempt_receipts,
        )

    # Step 5 — compute consensus deterministically + assemble the verdict.
    verdict = _assemble_verdict(extraction, responders)
    return VerdictSynthesisResult(
        verdict=verdict,
        extraction=extraction_provenance,
        verdict_absent_reason=None,
        attempt_receipts=attempt_receipts,
    )
