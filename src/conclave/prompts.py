"""Prompt templates for conclave deliberation modes.

Separated from :mod:`conclave.modes` so the orchestration (when to call whom)
stays distinct from the wording (what each role is told). The synthesize-mode
system prompt lives in :mod:`conclave.council` as ``_SYNTH_SYSTEM``; the strings
here belong to the debate and adversarial modes.
"""

from __future__ import annotations

from .models import ModelAnswer

# Version identifier for the synthesis/judge prompt *set*. Bump this string
# whenever ANY synthesizer-facing prompt changes -- the synthesize-mode system
# prompt (``conclave.council._SYNTH_SYSTEM``), the debate consolidation prompt
# (:data:`DEBATE_FINAL_SYSTEM`), or the adversarial judge prompt
# (:data:`JUDGE_SYSTEM`). It is surfaced on :class:`conclave.models.CouncilResult`
# (the ``prompt_version`` field) so a downstream eval or regression suite can
# detect that the wording the synthesis was produced under has shifted, rather
# than silently absorbing a prompt change as a quality regression. The value is
# opaque (a date-stamped tag); only equality/inequality is meaningful.
SYNTHESIS_PROMPT_VERSION = "2026-06-14"

# Stable position-based labels used to anonymize peers in debate rounds 2..N.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEBATE_SYSTEM = (
    "You are one member of a council of AI models debating a prompt over several "
    "rounds. You are shown your own previous answer and your anonymized peers' "
    "previous answers (labeled 'Model A', 'Model B', ...). Critically weigh the "
    "peer answers: where they expose a flaw or a better argument, revise your "
    "answer; where you remain correct, defend your position and say why. Produce "
    "a complete standalone answer to the original prompt -- not just a diff."
)

DEBATE_FINAL_SYSTEM = (
    "You are the synthesizer concluding a multi-round council debate. You are "
    "given the original prompt and each surviving member's final-round answer. "
    "Produce one consolidated, accurate answer. Reconcile where the debate "
    "converged, surface and adjudicate any durable disagreement, and note any "
    "answer that is clearly wrong. Rely only on the answers provided."
)

CRITIC_SYSTEM = (
    "You are a critic on an adversarial review council. You are given a prompt "
    "and a PROPOSAL answer from another model. Your job is to refute it: find the "
    "strongest flaws, unsupported claims, missing cases, and errors in the "
    "proposal. Do not agree to be agreeable -- if the proposal is largely correct, "
    "still stress-test it and name its weakest points and what would break it. Be "
    "specific and technical. State your critique, not a rewritten answer."
)

JUDGE_SYSTEM = (
    "You are the judge of an adversarial review. You are given the original "
    "prompt, a PROPOSAL answer, and several CRITIQUES of that proposal. Weigh the "
    "proposal against the critiques: accept critiques that are correct, reject "
    "ones that are wrong or overstated, and issue a verdict. Then produce the "
    "single strengthened final answer that survives the critiques. Rely only on "
    "the material provided; do not invent positions."
)


def anonymized_peer_block(
    self_name: str,
    self_letter: str,
    prior: dict[str, ModelAnswer],
    letters: dict[str, str],
) -> str:
    """Build the peer-answers text for one member in debate rounds 2..N.

    Args:
        self_name: The member receiving this block (excluded from "peers").
        self_letter: The anonymized label assigned to this member.
        prior: ``name -> ModelAnswer`` from the previous round (survivors only).
        letters: ``name -> letter`` stable label assignment for all survivors.

    Returns:
        A markdown block: the member's own prior answer plus each peer's prior
        answer relabeled by letter (brand identity withheld to reduce bias).
    """
    parts: list[str] = []
    own = prior.get(self_name)
    if own is not None and own.ok:
        parts.append(f"### Your previous answer (you are Model {self_letter})\n{own.answer}")
    for name, ans in prior.items():
        if name == self_name or not ans.ok:
            continue
        parts.append(f"### Model {letters[name]} (peer) previous answer\n{ans.answer}")
    return "\n\n".join(parts)


def debate_round_user(prompt: str, round_no: int, rounds: int, peer_block: str) -> str:
    """User-role content for a debate round >= 2."""
    return (
        f"Original prompt:\n{prompt}\n\n"
        f"Round {round_no} of {rounds}. Here are the answers from the previous "
        f"round:\n\n{peer_block}\n\n"
        "Now give your revised or defended answer to the original prompt."
    )


def debate_final_user(prompt: str, n_rounds: int, blocks: str) -> str:
    """User-role content for the debate's final synthesis."""
    return (
        f"Original prompt:\n{prompt}\n\n"
        f"Final-round answers after {n_rounds} round(s):\n\n{blocks}\n\n"
        "Now produce the consolidated answer."
    )


def critic_user(prompt: str, proposal_text: str) -> str:
    """User-role content for an adversarial critic."""
    return (
        f"Original prompt:\n{prompt}\n\n"
        f"PROPOSAL (from another model):\n{proposal_text}\n\n"
        "Now refute this proposal: give your strongest critique."
    )


def judge_user(prompt: str, proposer: str, proposal_text: str, critique_blocks: str) -> str:
    """User-role content for the adversarial judge."""
    return (
        f"Original prompt:\n{prompt}\n\n"
        f"PROPOSAL (from {proposer}):\n{proposal_text}\n\n"
        f"CRITIQUES:\n\n{critique_blocks}\n\n"
        "Now issue your verdict and the strengthened final answer."
    )
