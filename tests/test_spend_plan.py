"""The worst-case call plan per mode, derived from modes.py arithmetic (DSE-1514)."""

from __future__ import annotations

import pytest

from conclave.council import Council

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
    assert all(c.upstream_output_call_count == 6 for c in by_phase["revision"])  # N + N
    assert by_phase["synthesis"][0].upstream_output_call_count == 3  # N revisions
    assert by_phase["verdict_extraction"][0].upstream_output_call_count == 3
    assert by_phase["verdict_repair"][0].upstream_output_call_count == 4  # + its own attempt


def test_an_unknown_mode_is_a_value_error(keys):
    with pytest.raises(ValueError, match="unknown mode"):
        _council().plan_calls("telepathy", "q")


def test_planning_without_an_output_cap_is_refused(keys):
    with pytest.raises(ValueError, match="cannot bound spend: no output cap"):
        Council(models=MEMBERS, synthesizer="claude").plan_calls("synthesize", "q")
