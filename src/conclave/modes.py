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

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from . import prompts
from .logging import get_logger
from .models import AdversarialResult, CouncilResult, DebateRound, ModelAnswer, VoteResult
from .registry import key_present

if TYPE_CHECKING:  # avoid a circular import at runtime; only needed for typing
    from .council import Council

logger = get_logger("modes")


async def run_vote(
    council: Council,
    prompt: str,
    choices: list[str],
) -> CouncilResult:
    """Run a constrained-choice vote and return a :class:`CouncilResult`.

    Each council member is shown the prompt and a fixed option set labelled A,
    B, C, ... and asked to respond with a single letter. Responses are tallied;
    the plurality winner (if any) is stored in ``result.vote.winner``. A tie
    sets ``result.vote.split = True`` and ``result.vote.winner = None``.

    Args:
        council: The :class:`Council` providing fan-out and config.
        prompt: The user question to vote on.
        choices: Two or more option strings (e.g. ``["Option 1", "Option 2"]``).
            Each choice is assigned a consecutive uppercase letter starting at A.

    Returns:
        A :class:`CouncilResult` with ``mode="vote"`` and ``vote`` populated.
        ``synthesis`` carries a human-readable summary of the tally.
        Zero available members yields an empty result, not an error.
    """
    if len(choices) < 2:
        raise ValueError("vote mode requires at least 2 choices")

    members, skipped = council._available_members()
    result = CouncilResult(prompt=prompt, mode="vote", skipped=skipped)

    if not members:
        logger.warning("no council members have keys available; nothing to vote")
        result.vote = VoteResult(choices=choices)
        return result

    labels = [chr(65 + i) for i in range(len(choices))]
    label_to_choice = dict(zip(labels, choices, strict=False))

    messages_for = _vote_messages_for(prompt, choices)
    result.answers = await council.fan_out(members, messages_for)

    # Parse each member's response into a label.
    member_votes: dict[str, str | None] = {}
    for ans in result.answers:
        if not ans.ok or not ans.answer:
            member_votes[ans.name] = None
            continue
        raw = ans.answer.strip().upper()
        # Accept the first standalone label letter (word-boundary match).
        # A letter buried inside a word (e.g. "cANnot") is not accepted.
        chosen: str | None = None
        for m in re.finditer(r"\b([A-Z])\b", raw):
            if m.group(1) in label_to_choice:
                chosen = m.group(1)
                break
        member_votes[ans.name] = chosen

    # Tally votes.
    tally: dict[str, int] = {lbl: 0 for lbl in labels}
    for chosen in member_votes.values():
        if chosen is not None:
            tally[chosen] = tally.get(chosen, 0) + 1

    # Remove labels with zero votes for a cleaner tally.
    tally = {lbl: cnt for lbl, cnt in tally.items() if cnt > 0}

    # Determine winner (plurality = most votes; None on tie).
    winner: str | None = None
    split = False
    if tally:
        max_votes = max(tally.values())
        leaders = [lbl for lbl, cnt in tally.items() if cnt == max_votes]
        if len(leaders) == 1:
            winner = leaders[0]
        else:
            split = True

    vote_result = VoteResult(
        choices=choices,
        votes=member_votes,
        tally=tally,
        winner=winner,
        split=split,
    )
    result.vote = vote_result

    # Build a human-readable synthesis summarising the outcome.
    result.synthesis = _vote_summary(vote_result, label_to_choice, len(result.answers))
    return result


def _vote_messages_for(prompt: str, choices: list[str]):
    """Build the per-member message factory for a vote round."""
    messages = [
        {"role": "system", "content": prompts.VOTE_SYSTEM},
        {"role": "user", "content": prompts.vote_user(prompt, choices)},
    ]
    return lambda _name, _model_id: messages


def _vote_summary(vote_result: VoteResult, label_to_choice: dict[str, str], n_members: int) -> str:
    """Build a brief text summary of the vote outcome."""
    lines = [f"Vote result ({n_members} member(s) polled):"]
    for lbl, cnt in sorted(vote_result.tally.items(), key=lambda x: -x[1]):
        choice_text = label_to_choice.get(lbl, lbl)
        lines.append(f"  {lbl}. {choice_text}: {cnt} vote(s)")
    unparsed = sum(1 for v in vote_result.votes.values() if v is None)
    if unparsed:
        lines.append(f"  (unrecognised/failed responses: {unparsed})")
    if vote_result.winner is not None:
        winner_text = label_to_choice.get(vote_result.winner, vote_result.winner)
        lines.append(f"\nWinner: {vote_result.winner}. {winner_text}")
    elif vote_result.split:
        tied = [f"{lbl}. {label_to_choice.get(lbl, lbl)}" for lbl in sorted(vote_result.tally)]
        lines.append(f"\nTie: {' vs '.join(tied)}")
    else:
        lines.append("\nNo votes cast.")
    return "\n".join(lines)


async def run_debate(
    council: Council,
    prompt: str,
    rounds: int = 2,
    converge_threshold: float | None = None,
) -> CouncilResult:
    """Run a multi-round debate and return a structured :class:`CouncilResult`.

    Args:
        council: The :class:`Council` providing fan-out, config, and synthesizer.
        prompt: The user prompt.
        rounds: Maximum number of rounds (clamped to ``>= 1``). Round 1 is
            independent; each later round shows members their peers' anonymized
            prior answers.
        converge_threshold: Opt-in early-stop threshold in ``[0.0, 1.0]``. When
            ``None`` (default), the debate runs exactly ``rounds`` -- identical
            to the historic fixed-rounds behavior. When set, after each round
            ``>= 2`` the round-over-round answer stability is scored (see
            :func:`_round_convergence`); if it reaches the threshold the debate
            stops early. A degenerate score (e.g. no comparable answers) never
            triggers a stop and never crashes -- it falls back to running the
            remaining fixed rounds.

    Returns:
        A :class:`CouncilResult` with ``rounds`` (per-round answers), ``answers``
        mirroring the final round, and ``synthesis`` from the final consolidation.
        ``converged``/``convergence_score`` record whether (and at what score) an
        early stop fired; the actual rounds run is ``len(rounds)``. Survivors are
        tracked per round: a member that errors drops out of the next round. Zero
        available members yields an empty result, not an error.
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

        by_name = {a.name: a for a in answers}

        # Early-stop check: only from round 2 on (round 1 has no prior to compare
        # against), only when opted in, and only if more rounds would otherwise
        # run. A detector failure degrades to continuing the fixed rounds.
        if (
            converge_threshold is not None
            and round_no >= 2
            and round_no < rounds
            and _should_stop(prior, by_name, converge_threshold, result, round_no)
        ):
            break

        # Survivors for the next round = members that succeeded this round.
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


def _should_stop(
    prev: dict[str, ModelAnswer],
    curr: dict[str, ModelAnswer],
    threshold: float,
    result: CouncilResult,
    round_no: int,
) -> bool:
    """Decide whether the debate has converged enough to stop after this round.

    Scores the current round against the previous round via
    :func:`_round_convergence` and, when the score reaches ``threshold``, records
    the early stop on ``result`` (``converged`` + ``convergence_score``) and
    returns ``True``. Any unexpected failure in scoring is swallowed and treated
    as "not converged" so a detector bug can never crash the debate -- it simply
    degrades to running the remaining fixed rounds.
    """
    try:
        score = _round_convergence(prev, curr)
    except Exception as exc:  # noqa: BLE001 -- detector must never crash the run
        logger.warning("convergence scoring failed at round %d (%s); continuing", round_no, exc)
        return False

    if score is None:
        # No comparable answers (degenerate input): cannot conclude convergence.
        logger.info("round %d: no comparable answers for convergence; continuing", round_no)
        return False

    logger.info("round %d convergence score: %.3f (threshold %.3f)", round_no, score, threshold)
    if score >= threshold:
        result.converged = True
        result.convergence_score = score
        logger.info("debate converged at round %d (score %.3f); stopping early", round_no, score)
        return True
    return False


def _round_convergence(
    prev: dict[str, ModelAnswer],
    curr: dict[str, ModelAnswer],
) -> float | None:
    """Score round-over-round answer stability in ``[0.0, 1.0]``, or ``None``.

    The signal is **round-over-round stability**: for each member that produced a
    usable answer in *both* the previous and current round, compute the
    :class:`difflib.SequenceMatcher` ratio between the two answer texts (1.0 =
    identical, 0.0 = nothing in common), then average across those members. A high
    mean means members stopped revising -- the debate has stabilized.

    This signal is deliberately simple, deterministic, and dependency-free
    (stdlib ``difflib`` only), so it is fully offline-testable and adds no heavy
    dependency. It is preferred over cross-member agreement because a debate
    converging means members *stop changing their answers*, which is faithfully
    captured by self-stability and does not penalize a legitimate, stable
    disagreement between members.

    Returns:
        The mean stability ratio, or ``None`` when no member has a usable answer
        in both rounds (degenerate input -- the caller treats this as "not
        converged" rather than crashing or falsely stopping).
    """
    ratios: list[float] = []
    for name, curr_ans in curr.items():
        prev_ans = prev.get(name)
        if prev_ans is None or not prev_ans.ok or not curr_ans.ok:
            continue
        ratios.append(SequenceMatcher(None, prev_ans.answer or "", curr_ans.answer or "").ratio())
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


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
