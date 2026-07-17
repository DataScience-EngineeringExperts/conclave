"""Prompt templates for conclave deliberation modes.

Separated from :mod:`conclave.modes` so the orchestration (when to call whom)
stays distinct from the wording (what each role is told). The synthesize-mode
system prompt lives in :mod:`conclave.council` as ``_SYNTH_SYSTEM``; the strings
here belong to the debate and adversarial modes.
"""

from __future__ import annotations

from collections.abc import Sequence

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
SYNTHESIS_PROMPT_VERSION = "2026-06-29"

# Elite's prompt version is independent from the orchestration version owned by
# ``models.ELITE_PROTOCOL_VERSION``. Bump this whenever the critic or revision
# wording below changes; both versions are part of result-cache identity.
ELITE_PROMPT_VERSION = "1"

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

ELITE_CRITIC_SYSTEM = (
    "You are a claim auditor in an elite decision council. Independently "
    "stress-test the anonymized answers against the original prompt. Organize "
    "your audit into three explicit categories: SUPPORTED claims, CONFLICTING "
    "claims, and EXTERNALLY UNVERIFIED claims. Distinguish evidence supplied in "
    "the answers from assumptions that would require outside verification. Do "
    "not invent citations, sources, quotations, measurements, or facts. The "
    "displayed answer IDs provide within-run answer provenance only; they do "
    "not prove any claim against external sources. Cite those IDs when identifying "
    "a claim. Preserve meaningful "
    "minority positions and be precise about uncertainty."
)

ELITE_REVISION_SYSTEM = (
    "You are revising your independent answer after a claim audit by an "
    "elite decision council. Use the anonymized initial panel and critiques to "
    "correct unsupported claims, resolve conflicts where the supplied material "
    "permits, and clearly mark externally unverified claims. Do not invent "
    "citations, sources, quotations, measurements, or facts. Produce a complete "
    "standalone answer to the original prompt, not a description of your edits."
)


VOTE_SYSTEM = (
    "You are a voting member of a multi-model council. You will be given a "
    "question and a fixed set of labelled choices. You MUST respond with ONLY "
    "the single uppercase letter label of your chosen option (e.g. 'A' or 'B'). "
    "Do not write anything else — no explanation, no punctuation, no extra text. "
    "Just the single letter."
)


def vote_user(prompt: str, choices: list[str]) -> str:
    """User-role content for a constrained vote."""
    choice_block = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"Question:\n{prompt}\n\n"
        f"Choices:\n{choice_block}\n\n"
        "Respond with only the single letter of your chosen option."
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


def _elite_artifact_block(answers: Sequence[ModelAnswer], phase: str) -> str:
    """Render successful artifacts with stable position aliases and safe IDs."""
    parts: list[str] = []
    for index, answer in enumerate(answer for answer in answers if answer.ok):
        alias = LETTERS[index]
        answer_id = answer.answer_id or f"{phase}-{index + 1:03d}"
        parts.append(f"### Model {alias} {phase} (Answer ID: {answer_id})\n{answer.answer}")
    return "\n\n".join(parts)


def elite_critic_user(prompt: str, answers: Sequence[ModelAnswer]) -> str:
    """Build an anonymized claim-audit prompt with within-run answer IDs."""
    panel = _elite_artifact_block(answers, "initial answer")
    return (
        f"Original prompt:\n{prompt}\n\n"
        f"Anonymized initial panel:\n\n{panel}\n\n"
        "Audit the panel using the required claim categories."
    )


def elite_revision_user(
    prompt: str,
    original_answer: ModelAnswer,
    answers: Sequence[ModelAnswer],
    critiques: Sequence[ModelAnswer],
) -> str:
    """Build an anonymized elite revision prompt with all prior artifacts."""
    original_id = original_answer.answer_id or "initial-original"
    initial_panel = _elite_artifact_block(answers, "initial answer")
    critique_panel = _elite_artifact_block(critiques, "critique")
    return (
        f"Original prompt:\n{prompt}\n\n"
        f"Your original answer (Answer ID: {original_id}):\n"
        f"{original_answer.answer or ''}\n\n"
        f"Anonymized initial panel:\n\n{initial_panel}\n\n"
        f"Anonymized claim audits:\n\n{critique_panel}\n\n"
        "Now provide your claim-audit-aware revised answer."
    )
