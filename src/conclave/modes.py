"""Deliberation modes: multi-round debate and adversarial propose/refute/verdict.

Both modes are built on :meth:`conclave.council.Council.fan_out` (the single
concurrency + partial-failure primitive) and :meth:`Council.synthesize_blocks`
(the single synthesizer call path). Keeping the logic here keeps ``council.py``
focused on the v0.1 surface while the deliberation algorithms live on their own.
Prompt wording lives in :mod:`conclave.prompts`.

Design notes:

* **Anonymization (debate).** In rounds 2..N each member is shown its peers'
  prior-round answers relabeled as ``Model A/B/C`` by stable position, *not* by
  brand. This reduces brand-bias (a model deferring to or attacking another by
  name) while keeping the cross-pollination that makes debate useful. A member
  never sees its own answer relabeled -- it is told which letter is "you".
* **Drop-out (debate).** A member that errors in a round drops out of all
  subsequent rounds; the debate continues with the survivors. One model failing
  never aborts the run -- the partial-failure contract from v0.1 is preserved.
* **Adversarial roles.** The proposer answers first; every other available
  member is a critic explicitly prompted to refute; the synthesizer is the judge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import prompts
from .logging import get_logger
from .models import AdversarialResult, CouncilResult, DebateRound, ModelAnswer
from .registry import key_present

if TYPE_CHECKING:  # avoid a circular import at runtime; only needed for typing
    from .council import Council

logger = get_logger("modes")


async def run_debate(council: Council, prompt: str, rounds: int = 2) -> CouncilResult:
    """Run a multi-round debate and return a structured :class:`CouncilResult`.

    Args:
        council: The :class:`Council` providing fan-out, config, and synthesizer.
        prompt: The user prompt.
        rounds: Number of rounds (clamped to ``>= 1``). Round 1 is independent;
            each later round shows members their peers' anonymized prior answers.

    Returns:
        A :class:`CouncilResult` with ``rounds`` (per-round answers), ``answers``
        mirroring the final round, and ``synthesis`` from the final consolidation.
        Survivors are tracked per round: a member that errors drops out of the
        next round. Zero available members yields an empty result, not an error.
    """
    rounds = max(1, rounds)
    members, skipped = council._available_members()
    result = CouncilResult(prompt=prompt, mode="debate", skipped=skipped)

    if not members:
        logger.warning("no council members have keys available; nothing to debate")
        return result

    # Stable letter labels by initial position; survives drop-outs.
    letters = {
        name: prompts.LETTERS[i % len(prompts.LETTERS)] for i, (name, _) in enumerate(members)
    }

    survivors = list(members)  # (name, model_id) pairs still in the debate
    prior: dict[str, ModelAnswer] = {}  # previous round's answers, by name

    for round_no in range(1, rounds + 1):
        if not survivors:
            logger.warning("debate ended early at round %d: no survivors", round_no)
            break

        messages_for = _debate_messages_for(prompt, round_no, rounds, prior, letters)
        answers = await council.fan_out(survivors, messages_for)
        result.rounds.append(DebateRound(round_number=round_no, answers=answers))

        # Survivors for the next round = members that succeeded this round.
        by_name = {a.name: a for a in answers}
        prior = by_name
        next_survivors = [(n, m) for (n, m) in survivors if by_name[n].ok]
        dropped = [n for (n, _m) in survivors if not by_name[n].ok]
        if dropped:
            logger.warning(
                "round %d: dropping failed members from next round: %s",
                round_no,
                ", ".join(dropped),
            )
        survivors = next_survivors

    # Mirror the final round into answers so existing consumers keep working.
    if result.rounds:
        result.answers = list(result.rounds[-1].answers)

    await _debate_synthesize(council, result)
    return result


def _debate_messages_for(
    prompt: str,
    round_no: int,
    rounds: int,
    prior: dict[str, ModelAnswer],
    letters: dict[str, str],
):
    """Build the per-member message factory for one debate round.

    Round 1 sends the bare prompt; later rounds inject each member's own and its
    peers' anonymized prior answers. ``prior``/``letters`` are read at task-build
    time inside ``fan_out``, so binding them here is safe.
    """
    if round_no == 1:
        base = [{"role": "user", "content": prompt}]
        return lambda _name, _model_id: base

    def messages_for(name: str, _model_id: str) -> list[dict[str, str]]:
        peer_block = prompts.anonymized_peer_block(name, letters[name], prior, letters)
        return [
            {"role": "system", "content": prompts.DEBATE_SYSTEM},
            {
                "role": "user",
                "content": prompts.debate_round_user(prompt, round_no, rounds, peer_block),
            },
        ]

    return messages_for


async def _debate_synthesize(council: Council, result: CouncilResult) -> None:
    """Consolidate the final round's surviving answers via the synthesizer."""
    final = result.rounds[-1].successful_answers if result.rounds else []
    if not final:
        result.synthesis_error = "no surviving member answers to synthesize"
        logger.warning(result.synthesis_error)
        return

    synth_id = council.config.resolve_model_id(council.synthesizer)
    result.synthesizer = council.synthesizer
    result.synthesizer_model_id = synth_id
    if not key_present(synth_id):
        result.synthesis_error = (
            f"synthesizer '{council.synthesizer}' ({synth_id}) has no API key; "
            "returning final-round answers only"
        )
        logger.warning(result.synthesis_error)
        return

    blocks = "\n\n".join(
        f"### Final answer from {a.name} ({a.model_id})\n{a.answer}" for a in final
    )
    user_content = prompts.debate_final_user(result.prompt, len(result.rounds), blocks)
    answer = await council.synthesize_blocks(prompts.DEBATE_FINAL_SYSTEM, user_content)
    if answer.ok:
        result.synthesis = answer.answer
    else:
        result.synthesis_error = answer.error


async def run_adversarial(
    council: Council, prompt: str, proposer: str | None = None
) -> CouncilResult:
    """Run a propose -> refute -> verdict pass and return a :class:`CouncilResult`.

    The proposer is a single point of failure, so the run is layered to survive a
    bad one:

    1. **Proposer fallback.** Members are tried as proposer in council order,
       starting with the requested one. A member that returns an unusable answer
       (``ModelAnswer.error`` set / no text) is recorded and the next member is
       tried, until one produces a usable proposal.
    2. **Graceful degrade.** If no member can propose, the run does *not* abort
       with "no verdict". It degrades to a plain synthesize over the surviving
       members and surfaces an actionable warning on ``CouncilResult.synthesis_error``
       (mirrored to the adversarial verdict so consumers reading either field see
       why the adversarial flow was skipped).

    Args:
        council: The :class:`Council` providing fan-out, config, and judge.
        prompt: The user prompt.
        proposer: Friendly name of the proposing member. Defaults to the first
            requested council member. If the named proposer has no key, the run
            falls back to the first available member; if its answer is unusable,
            the run falls back to the next available member as proposer.

    Returns:
        A :class:`CouncilResult` whose ``adversarial`` field carries the proposal,
        critiques, and verdict. ``synthesis`` mirrors the verdict and ``answers``
        contains the proposal plus each critique so existing consumers keep
        working. Zero available members yields an empty result, not an error.
    """
    members, skipped = council._available_members()
    result = CouncilResult(prompt=prompt, mode="adversarial", skipped=skipped)

    if not members:
        logger.warning("no council members have keys available; nothing to propose")
        return result

    requested_proposer = proposer or council.requested_models[0]
    order = _proposer_order(members, requested_proposer)

    # Step 1: find a proposer that produces a usable answer. Each attempt is a
    # single-member fan-out reusing the same partial-failure primitive. Failed
    # attempts are recorded so the judge/degrade path can explain what was tried.
    base = [{"role": "user", "content": prompt}]
    proposal: ModelAnswer | None = None
    p_name = order[0][0]
    failed_proposers: list[ModelAnswer] = []
    for cand_name, cand_model_id in order:
        attempt = (await council.fan_out([(cand_name, cand_model_id)], lambda _n, _m: base))[0]
        result.answers.append(attempt)
        if attempt.ok:
            proposal = attempt
            p_name = cand_name
            break
        failed_proposers.append(attempt)
        logger.warning(
            "proposer '%s' produced no usable answer (%s); trying next member",
            cand_name,
            attempt.error,
        )

    # Step 2: no member could propose -> degrade to synthesize over the survivors
    # instead of aborting the whole run with "no verdict produced".
    if proposal is None:
        await _degrade_to_synthesize(council, result, failed_proposers)
        return result

    adv = AdversarialResult(proposer=p_name, proposal=proposal)

    # Step 3: critics refute the proposal. Every other available member critiques;
    # any failed-proposer attempts are excluded so a member is never both.
    tried = {a.name for a in failed_proposers} | {p_name}
    critics = [(n, m) for (n, m) in members if n not in tried]
    if critics:
        adv.critiques = await council.fan_out(
            critics, _critic_messages_for(prompt, proposal.answer or "")
        )
        result.answers.extend(adv.critiques)

    # Step 4: the judge weighs proposal vs critiques and issues a verdict.
    await _adversarial_judge(council, prompt, adv)
    result.adversarial = adv
    result.synthesis = adv.verdict
    result.synthesis_error = adv.verdict_error
    result.synthesizer = adv.judge
    result.synthesizer_model_id = adv.judge_model_id
    return result


async def _degrade_to_synthesize(
    council: Council,
    result: CouncilResult,
    failed_proposers: list[ModelAnswer],
) -> None:
    """Fall back to plain synthesize when no member can produce a proposal.

    Every requested member was tried as proposer and none returned a usable
    answer, so there is nothing to refute. Rather than emit "no verdict produced",
    we run the standard synthesizer over whatever members *did* answer (here:
    none succeeded, by definition of reaching this path) and always surface an
    actionable warning on ``result.synthesis_error`` explaining the degrade. The
    warning is mirrored into an :class:`AdversarialResult` so consumers reading
    ``result.adversarial.verdict_error`` (e.g. the CLI) see it too.
    """
    names = ", ".join(a.name for a in failed_proposers) or "(none)"
    warning = (
        "adversarial degraded to synthesize: no council member produced a usable "
        f"proposal (tried: {names}). Showing a consolidated answer over the "
        "surviving members instead of a propose/refute/verdict result."
    )
    logger.warning(warning)

    # Reuse the single synthesizer path. successful_answers is empty here (all
    # proposer attempts failed), so _synthesize records its own no-answers error;
    # we override synthesis_error afterwards so the degrade reason is what surfaces.
    await council._synthesize(result)
    result.synthesis_error = warning
    result.synthesizer = council.synthesizer
    result.synthesizer_model_id = council.config.resolve_model_id(council.synthesizer)

    # Mirror the degrade into the adversarial structure so the field-specific CLI
    # renderer and library consumers reading result.adversarial both see it. The
    # first failed attempt stands in as the (failed) proposal for shape parity.
    if failed_proposers:
        adv = AdversarialResult(proposer=failed_proposers[0].name, proposal=failed_proposers[0])
        adv.verdict_error = warning
        adv.judge = council.synthesizer
        adv.judge_model_id = council.config.resolve_model_id(council.synthesizer)
        result.adversarial = adv


def _critic_messages_for(prompt: str, proposal_text: str):
    """Build the per-critic message factory for the refutation step."""

    def critic_messages(_name: str, _model_id: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": prompts.CRITIC_SYSTEM},
            {"role": "user", "content": prompts.critic_user(prompt, proposal_text)},
        ]

    return critic_messages


def _proposer_order(members: list[tuple[str, str]], requested: str) -> list[tuple[str, str]]:
    """Return members ordered for proposer selection: requested first, then rest.

    The requested proposer is tried first when it is available; every other
    available member follows in council order so a failed proposer can fall back
    to the next candidate. A requested name with no key is simply absent from
    ``members`` (filtered upstream), so the council order leads.
    """
    requested_member = next((m for m in members if m[0] == requested), None)
    if requested_member is None:
        return list(members)
    rest = [m for m in members if m[0] != requested]
    return [requested_member, *rest]


async def _adversarial_judge(council: Council, prompt: str, adv: AdversarialResult) -> None:
    """Run the judge over the proposal + critiques, mutating ``adv``."""
    judge_id = council.config.resolve_model_id(council.synthesizer)
    adv.judge = council.synthesizer
    adv.judge_model_id = judge_id

    if not adv.proposal.ok:
        adv.verdict_error = (
            f"proposal from '{adv.proposer}' failed ({adv.proposal.error}); no verdict produced"
        )
        logger.warning(adv.verdict_error)
        return
    if not key_present(judge_id):
        adv.verdict_error = (
            f"judge '{council.synthesizer}' ({judge_id}) has no API key; "
            "returning proposal and critiques only"
        )
        logger.warning(adv.verdict_error)
        return

    usable_critiques = adv.successful_critiques
    if usable_critiques:
        critique_blocks = "\n\n".join(
            f"### Critique from {c.name} ({c.model_id})\n{c.answer}" for c in usable_critiques
        )
    else:
        critique_blocks = "(no usable critiques were produced)"

    user_content = prompts.judge_user(
        prompt, adv.proposer, adv.proposal.answer or "", critique_blocks
    )
    answer = await council.synthesize_blocks(prompts.JUDGE_SYSTEM, user_content)
    if answer.ok:
        adv.verdict = answer.answer
    else:
        adv.verdict_error = answer.error
