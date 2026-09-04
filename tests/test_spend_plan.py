"""The worst-case call plan per mode, derived from modes.py arithmetic (DSE-1514)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from conclave.council import Council
from conclave.manifest import ModelHarnessManifest, ProviderExecutionReceipt
from conclave.models import CouncilResult, DebateRound
from conclave.pricing import reserve_cost
from tests.test_pricing_receipts import _install_snapshot, _snapshot

MEMBERS = ["grok", "gemini", "openai"]  # N = 3


def _council(**kwargs) -> Council:
    kwargs.setdefault("models", MEMBERS)
    kwargs.setdefault("synthesizer", "claude")
    kwargs.setdefault("max_output_tokens", 1_000)
    return Council(**kwargs)


@pytest.mark.parametrize(
    ("mode", "kwargs", "expected"),
    [
        ("raw", {}, 3),  # N
        ("synthesize", {}, 3 + 1 + 2),  # N + C + 2C
        ("vote", {"choices": ["a", "b"]}, 3),  # N
        ("debate", {"rounds": 3}, 3 * 3 + 1),  # N*R + C
        ("adversarial", {}, 3 + 1),  # N + C
        ("elite", {}, 3 * 3 + 1 + 2),  # 3N + C + 2C
    ],
)
def test_worst_case_call_counts_per_mode(keys, mode, kwargs, expected):
    plan = _council().plan_calls(mode, "q", **kwargs)
    assert len(plan.calls) == expected
    assert plan.mode == mode
    assert plan.member_count == 3
    assert plan.chain_count == 1


def test_a_longer_chain_multiplies_every_adjudication_role(keys):
    council = _council(synthesizer="claude>grok>gemini")  # C = 3
    assert len(council.plan_calls("synthesize", "q").calls) == 3 + 3 + 6
    assert len(council.plan_calls("elite", "q").calls) == 9 + 3 + 6
    assert len(council.plan_calls("adversarial", "q").calls) == 3 + 3
    assert council.plan_calls("synthesize", "q").chain_count == 3


def test_an_unkeyed_chain_candidate_is_not_planned(monkeypatch, keys):
    # mistral is unkeyed here -> it can never be called, so it can never cost
    # anything, so it is not in the plan.
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    council = _council(synthesizer="claude>mistral")
    plan = council.plan_calls("synthesize", "q")
    assert plan.chain_count == 1
    assert len(plan.calls) == 3 + 1 + 2


def test_verdict_extraction_off_removes_exactly_two_calls_per_candidate(keys):
    on = _council().plan_calls("synthesize", "q")
    off = _council(extract_verdict=False).plan_calls("synthesize", "q")
    assert len(on.calls) - len(off.calls) == 2


def test_every_planned_call_is_bounded_and_names_its_model(keys):
    for call in _council().plan_calls("elite", "q").calls:
        assert call.max_output_tokens == 1_000
        assert "/" in call.model_id
        assert call.prompt_token_upper_bound >= len(b"q")
        assert call.prompt_template_token_allowance >= 0
        assert call.provider_framing_token_allowance >= 64
        assert call.upstream_output_call_count >= 0


def test_downstream_phases_declare_their_upstream_dependencies(keys):
    by_phase: dict[str, list] = {}
    for call in _council().plan_calls("elite", "q").calls:
        by_phase.setdefault(call.phase, []).append(call)

    assert all(c.upstream_output_call_count == 0 for c in by_phase["initial"])
    assert all(c.upstream_output_call_count == 3 for c in by_phase["critique"])  # N initials
    # DSE-1514 review, Fix A: N (initial panel) + N (critique panel) + 1 -- every
    # reviser's OWN initial answer is embedded a second time as its standalone
    # "original answer" (modes._elite_revision_messages_for), not just inside
    # the anonymized panel. Was asserted as 6 (2N); corrected to 7 (2N+1) after
    # the byte-lower-bound regression test proved 2N alone undercounts the real
    # message.
    assert all(c.upstream_output_call_count == 7 for c in by_phase["revision"])
    assert by_phase["synthesis"][0].upstream_output_call_count == 3  # N revisions
    assert by_phase["verdict_extraction"][0].upstream_output_call_count == 3
    assert by_phase["verdict_repair"][0].upstream_output_call_count == 4  # + its own attempt


def test_an_unknown_mode_is_a_value_error(keys):
    with pytest.raises(ValueError, match="unknown mode"):
        _council().plan_calls("telepathy", "q")


def test_planning_without_an_output_cap_is_refused(keys):
    with pytest.raises(ValueError, match="cannot bound spend: no output cap"):
        Council(models=MEMBERS, synthesizer="claude").plan_calls("synthesize", "q")


# --- DSE-1514 Round 4 review, Fix A: the adversarial byte-worst-case shape ---
# and a general byte-lower-bound regression covering every phase whose input
# embeds a prior call's output. The ticket's guarantee is "never below the
# real request bytes"; these tests hold every phase to it using the REAL
# message-building functions (never a hand re-derivation of their output).

MAX_OUTPUT_TOKENS = 1_000
MAX_OUTPUT_BYTES_PER_TOKEN = 8
MAX_LEN_TEXT = "x" * (MAX_OUTPUT_TOKENS * MAX_OUTPUT_BYTES_PER_TOKEN)
PROMPT = "q"


def _plan_members() -> list[tuple[str, str]]:
    """The exact (name, model_id) pairs _council() resolves, N=3."""
    return _council()._available_members()[0]


def _max_len_answers():
    from conclave.council import _ANSWER_ID_PROBE
    from conclave.models import ModelAnswer

    return [
        ModelAnswer(name=name, model_id=model_id, answer=MAX_LEN_TEXT, answer_id=_ANSWER_ID_PROBE)
        for name, model_id in _plan_members()
    ]


def _phase_call(plan, phase: str):
    return next(c for c in plan.calls if c.phase == phase)


def test_adversarial_byte_worst_case_is_one_proposer_and_n_minus_one_critics(keys):
    """1 proposer (upstream=0) + N-1 critics (upstream=1 each); judge upstream=N."""
    plan = _council().plan_calls("adversarial", PROMPT)
    proposals = [c for c in plan.calls if c.phase == "proposal"]
    critiques = [c for c in plan.calls if c.phase == "critique"]
    judges = [c for c in plan.calls if c.phase == "judge"]

    assert len(proposals) == 1
    assert proposals[0].upstream_output_call_count == 0
    assert len(critiques) == 2
    assert all(c.upstream_output_call_count == 1 for c in critiques)
    assert len(judges) == 1
    assert judges[0].upstream_output_call_count == 3


def test_a_critic_call_has_a_strictly_larger_input_bound_than_the_proposal(keys):
    plan = _council().plan_calls("adversarial", PROMPT)
    proposal = _phase_call(plan, "proposal")
    critique = _phase_call(plan, "critique")
    bound_kwargs = {"upstream_output_bytes_per_token": MAX_OUTPUT_BYTES_PER_TOKEN}
    assert critique.input_bytes_bound(**bound_kwargs) > proposal.input_bytes_bound(**bound_kwargs)


def _critic_case():
    from conclave.modes import _critic_messages_for

    plan = _council().plan_calls("adversarial", PROMPT)
    call = _phase_call(plan, "critique")
    messages = _critic_messages_for(PROMPT, MAX_LEN_TEXT)("critic", "x/x")
    return call, messages


def _synthesis_case():
    import conclave.council as council_mod

    plan = _council().plan_calls("synthesize", PROMPT)
    call = _phase_call(plan, "synthesis")
    content = council_mod._synth_user_content(PROMPT, _max_len_answers())
    messages = [
        {"role": "system", "content": council_mod._SYNTH_SYSTEM},
        {"role": "user", "content": content},
    ]
    return call, messages


def _debate_round2_case():
    from conclave import prompts

    plan = _council().plan_calls("debate", PROMPT, rounds=2)
    call = _phase_call(plan, "round-2")
    members = _plan_members()
    answers = _max_len_answers()
    letters = {name: prompts.LETTERS[i] for i, (name, _mid) in enumerate(members)}
    prior = {name: answer for (name, _mid), answer in zip(members, answers, strict=True)}
    self_name = members[0][0]
    peer_block = prompts.anonymized_peer_block(self_name, letters[self_name], prior, letters)
    messages = [
        {"role": "system", "content": prompts.DEBATE_SYSTEM},
        {"role": "user", "content": prompts.debate_round_user(PROMPT, 2, 2, peer_block)},
    ]
    return call, messages


def _elite_critique_case():
    from conclave import prompts

    plan = _council().plan_calls("elite", PROMPT)
    call = _phase_call(plan, "critique")
    messages = [
        {"role": "system", "content": prompts.ELITE_CRITIC_SYSTEM},
        {"role": "user", "content": prompts.elite_critic_user(PROMPT, _max_len_answers())},
    ]
    return call, messages


def _elite_revision_case():
    from conclave import prompts

    plan = _council().plan_calls("elite", PROMPT)
    call = _phase_call(plan, "revision")
    answers = _max_len_answers()
    critiques = _max_len_answers()
    messages = [
        {"role": "system", "content": prompts.ELITE_REVISION_SYSTEM},
        {
            "role": "user",
            "content": prompts.elite_revision_user(PROMPT, answers[0], answers, critiques),
        },
    ]
    return call, messages


def _judge_case():
    """The adversarial judge: proposal + N-1 critiques, all at max length.

    DSE-1514 review, re-review important: the original byte-lower-bound suite
    omitted this phase (real 24,602 B vs planned 24,640 B -- 38 bytes of
    margin, unverified). The default proposer is the first requested member
    (``modes.run_adversarial``'s ``council.requested_models[0]``), so the
    remaining N-1 members are the critics whose real names/model ids appear
    inside the judge's ``critique_blocks``, exactly as ``_adversarial_judge``
    builds it.
    """
    from conclave import prompts

    plan = _council().plan_calls("adversarial", PROMPT)
    call = _phase_call(plan, "judge")
    members = _plan_members()
    proposer_name = members[0][0]
    answers = _max_len_answers()
    proposal_text = answers[0].answer
    critiques = answers[1:]
    critique_blocks = "\n\n".join(
        f"### Critique from {c.name} ({c.model_id})\n{c.answer}" for c in critiques
    )
    messages = [
        {"role": "system", "content": prompts.JUDGE_SYSTEM},
        {
            "role": "user",
            "content": prompts.judge_user(PROMPT, proposer_name, proposal_text, critique_blocks),
        },
    ]
    return call, messages


def _verdict_extraction_case():
    from conclave.verdict_synthesis import _build_messages

    plan = _council().plan_calls("synthesize", PROMPT)
    call = _phase_call(plan, "verdict_extraction")
    messages = _build_messages(PROMPT, _max_len_answers())
    return call, messages


def _verdict_repair_case():
    from conclave.verdict_synthesis import (
        VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES,
        _build_messages,
        _repair_instruction,
    )

    plan = _council().plan_calls("synthesize", PROMPT)
    call = _phase_call(plan, "verdict_repair")
    messages = _build_messages(PROMPT, _max_len_answers()) + [
        {
            "role": "user",
            "content": _repair_instruction("e" * VERDICT_REPAIR_ERROR_DETAIL_MAX_BYTES),
        }
    ]
    return call, messages


@pytest.mark.parametrize(
    "case_builder",
    [
        _critic_case,
        _judge_case,
        _synthesis_case,
        _debate_round2_case,
        _elite_critique_case,
        _elite_revision_case,
        _verdict_extraction_case,
        _verdict_repair_case,
    ],
    ids=[
        "adversarial-critique",
        "adversarial-judge",
        "synthesize-synthesis",
        "debate-round-2",
        "elite-critique",
        "elite-revision",
        "verdict-extraction",
        "verdict-repair",
    ],
)
def test_the_planned_byte_bound_never_falls_below_the_real_worst_case_message(keys, case_builder):
    """DSE-1514 review, Fix A: "never below the real request bytes", every phase.

    Builds the REAL message list for each phase (via the actual mode/prompt
    builders, never a hand-derived approximation) with worst-case-length
    (``max_output_tokens * max_output_bytes_per_token``) upstream text, and
    proves the planned call's byte bound covers it.
    """
    call, messages = case_builder()
    real_bytes = sum(len(m["content"].encode("utf-8")) for m in messages)
    assert real_bytes <= call.input_bytes_bound(
        upstream_output_bytes_per_token=MAX_OUTPUT_BYTES_PER_TOKEN
    )


# --------------------------------------------------------------------------- #
# DSE-1514 Round 4, QA C1: a usage-less RECEIPT's phase-aware reservation
# (Council._price_manifest, via Council._reservation_row_for_phase) must never
# price below the real worst-case message for its phase -- the regression that
# makes the flat-constant bug (a synthesis/judge/verdict/revision receipt,
# which embeds upstream output, priced 3-9x low) impossible to re-introduce.
# --------------------------------------------------------------------------- #


def _floor_reservation_usd(rates, real_bytes: int, cap: int) -> Decimal:
    """The cheapest a phase's real worst-case message could possibly cost.

    Treats ``real_bytes`` as pure prompt content with ZERO extra template/
    framing allowance, so this is a floor: the real reservation
    ``Council._price_manifest`` prices from a matched plan row can only be >=
    this (its own template/framing/upstream terms are non-negative additions
    on top of at least as many bytes), never below it.
    """
    return reserve_cost(
        rates,
        prompt_token_upper_bound=real_bytes,
        prompt_template_token_allowance=0,
        provider_framing_token_allowance=0,
        upstream_output_token_ceilings=(),
        upstream_output_bytes_per_token=MAX_OUTPUT_BYTES_PER_TOKEN,
        max_output_tokens=cap,
    ).reserved_cost_usd


def _priced_receipt(council, *, mode: str, phase: str | None, model_id: str, rounds_count: int):
    """Price ONE hand-built, usage-less receipt through the real pricing path.

    Bypasses a full simulated council run (which would need a distinct
    provider-mocking scenario per phase) by constructing the minimal
    :class:`CouncilResult` + manifest ``_price_manifest`` actually reads:
    ``mode`` (which plan table to rebuild), ``rounds`` (debate's real round
    count), and one receipt of the phase under test carrying no usage.
    """
    receipt = ProviderExecutionReceipt(
        name="target", provider=model_id.split("/", 1)[0], model_id=model_id, phase=phase
    )
    result = CouncilResult(
        prompt=PROMPT,
        mode=mode,
        rounds=[DebateRound(round_number=n, answers=[]) for n in range(1, rounds_count + 1)],
    )
    result.manifest = ModelHarnessManifest(
        request_id="r", conclave_version="0", mode=mode, model_ids=[model_id], receipts=[receipt]
    )
    council._price_manifest(result)
    return result.manifest.receipts[0]


# (label, receipt phase, mode, debate rounds run, model id, real-message builder)
_PHASE_RESERVATION_CASES = [
    ("member", None, "raw", 0, "xai/grok-4.3", lambda: [{"role": "user", "content": PROMPT}]),
    ("round-2", "round-2", "debate", 2, "xai/grok-4.3", lambda: _debate_round2_case()[1]),
    (
        "synthesis",
        "synthesis",
        "synthesize",
        0,
        "anthropic/claude-sonnet-4-6",
        lambda: _synthesis_case()[1],
    ),
    ("judge", "judge", "adversarial", 0, "anthropic/claude-sonnet-4-6", lambda: _judge_case()[1]),
    (
        "verdict_extraction",
        "verdict_extraction",
        "synthesize",
        0,
        "anthropic/claude-sonnet-4-6",
        lambda: _verdict_extraction_case()[1],
    ),
    (
        "verdict_repair",
        "verdict_repair",
        "synthesize",
        0,
        "anthropic/claude-sonnet-4-6",
        lambda: _verdict_repair_case()[1],
    ),
    ("critique", "critique", "elite", 0, "xai/grok-4.3", lambda: _elite_critique_case()[1]),
    ("revision", "revision", "elite", 0, "xai/grok-4.3", lambda: _elite_revision_case()[1]),
]


@pytest.mark.parametrize(
    ("phase", "receipt_phase", "mode", "rounds_count", "model_id", "messages_builder"),
    _PHASE_RESERVATION_CASES,
    ids=[case[0] for case in _PHASE_RESERVATION_CASES],
)
def test_phase_aware_reservation_never_prices_below_the_real_worst_case_message(
    monkeypatch, keys, phase, receipt_phase, mode, rounds_count, model_id, messages_builder
):
    """DSE-1514 Round 4, QA C1: the assertion that makes the regression impossible.

    For every phase whose receipt can be usage-less and reservation-priced,
    prices a manifest holding exactly one such receipt (``usage=None``,
    ``phase=receipt_phase`` -- ``None`` for an untagged raw-mode member call)
    and proves the reservation is >= the cost of the REAL worst-case message
    for that phase (built via the actual mode/prompt builders in
    ``tests.test_spend_plan``'s existing case functions, never a hand
    re-derivation), priced at the input rate alone with zero extra allowance
    -- a floor the real reservation's own non-negative template/framing/
    upstream terms can only sit at or above. A flat, phase-blind constant
    (the QA C1 bug) would fail this for every upstream-embedding phase
    (synthesis, judge, verdict_extraction, verdict_repair, revision).
    """
    _install_snapshot(
        monkeypatch,
        _snapshot(
            "xai/grok-4.3",
            "gemini/gemini-2.5-pro",
            "openai/gpt-4.1",
            "anthropic/claude-sonnet-4-6",
        ),
    )
    council = _council()
    messages = messages_builder()
    real_bytes = sum(len(m["content"].encode("utf-8")) for m in messages)

    priced = _priced_receipt(
        council,
        mode=mode,
        phase=receipt_phase,
        model_id=model_id,
        rounds_count=rounds_count,
    )

    assert priced.cost_basis == "reservation"
    assert priced.cost_ceiling_usd is not None

    import conclave.council as council_mod

    snapshot = council_mod.load_default_price_snapshot()
    rates = snapshot.rates_for(model_id)
    floor = _floor_reservation_usd(rates, real_bytes, MAX_OUTPUT_TOKENS)
    assert priced.cost_ceiling_usd >= floor
