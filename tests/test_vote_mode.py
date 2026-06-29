"""Tests for the vote deliberation mode (issue #3 / CAC-09).

All tests run offline via the ``patch_call_model`` fixture. The handler
returns a single letter ('A', 'B', etc.) for member calls so the tally
logic can be exercised without a live provider.
"""

from __future__ import annotations

import pytest

from conclave import Council
from conclave.config import ConclaveConfig
from conclave.models import VoteResult
from conclave.modes import _vote_summary, run_vote
from conclave.prompts import VOTE_SYSTEM, vote_user
from tests.conftest import make_response


def _two_member_config() -> ConclaveConfig:
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "claude": "anthropic/claude-sonnet-4-6",
        },
        councils={"default": ["grok", "claude"]},
        synthesizer="claude",
    )


def _three_member_config() -> ConclaveConfig:
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": "anthropic/claude-sonnet-4-6",
        },
        councils={"default": ["grok", "gemini", "claude"]},
        synthesizer="claude",
    )


def _set_keys(monkeypatch):
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")


# ---------------------------------------------------------------------------
# Prompt template tests (offline, no Council needed)
# ---------------------------------------------------------------------------


class TestVoteSystem:
    def test_vote_system_is_string(self):
        assert isinstance(VOTE_SYSTEM, str)

    def test_vote_system_mentions_single_letter(self):
        assert "single" in VOTE_SYSTEM.lower() or "letter" in VOTE_SYSTEM.lower()


class TestVoteUser:
    def test_contains_choices(self):
        content = vote_user("Who should win?", ["Alice", "Bob"])
        assert "A. Alice" in content
        assert "B. Bob" in content

    def test_contains_prompt(self):
        content = vote_user("My question", ["X", "Y"])
        assert "My question" in content

    def test_three_choices_labeled_abc(self):
        content = vote_user("Pick one", ["X", "Y", "Z"])
        assert "A. X" in content
        assert "B. Y" in content
        assert "C. Z" in content


# ---------------------------------------------------------------------------
# VoteResult model tests
# ---------------------------------------------------------------------------


class TestVoteResult:
    def test_defaults(self):
        vr = VoteResult(choices=["Yes", "No"])
        assert vr.winner is None
        assert vr.split is False
        assert vr.tally == {}
        assert vr.votes == {}

    def test_serialises_cleanly(self):
        vr = VoteResult(
            choices=["Yes", "No"],
            votes={"grok": "A", "claude": "B"},
            tally={"A": 1, "B": 1},
            winner=None,
            split=True,
        )
        d = vr.model_dump()
        assert d["split"] is True
        assert d["winner"] is None


# ---------------------------------------------------------------------------
# run_vote unit tests (transport mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vote_majority_winner(monkeypatch, patch_call_model):
    """Three members: grok→A, gemini→A, claude→B → A wins."""
    _set_keys(monkeypatch)

    responses = {"xai/grok-4.3": "A", "gemini/gemini-2.5-pro": "A", "anthropic/claude-sonnet-4-6": "B"}

    def handler(model_id, messages):
        return make_response(responses[model_id])

    patch_call_model(handler)
    cfg = _three_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    result = await run_vote(council, "Best option?", ["Alpha", "Beta"])

    assert result.mode == "vote"
    assert result.vote is not None
    assert result.vote.winner == "A"
    assert result.vote.split is False
    assert result.vote.tally.get("A", 0) == 2
    assert result.vote.tally.get("B", 0) == 1


@pytest.mark.asyncio
async def test_vote_split(monkeypatch, patch_call_model):
    """Two members: grok→A, claude→B → tie (split)."""
    _set_keys(monkeypatch)

    responses = {"xai/grok-4.3": "A", "anthropic/claude-sonnet-4-6": "B"}

    def handler(model_id, messages):
        return make_response(responses[model_id])

    patch_call_model(handler)
    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    result = await run_vote(council, "A or B?", ["Alpha", "Beta"])

    assert result.vote is not None
    assert result.vote.split is True
    assert result.vote.winner is None


@pytest.mark.asyncio
async def test_vote_synthesis_is_produced(monkeypatch, patch_call_model):
    """Synthesis (plain-text summary) is populated after a vote run."""
    _set_keys(monkeypatch)

    def handler(model_id, messages):
        return make_response("A")

    patch_call_model(handler)
    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    result = await run_vote(council, "Best option?", ["Yes", "No"])

    assert result.synthesis is not None
    assert "Vote result" in result.synthesis


@pytest.mark.asyncio
async def test_vote_unrecognised_response_excluded_from_tally(monkeypatch, patch_call_model):
    """A member that returns junk text does not contribute to the tally."""
    _set_keys(monkeypatch)

    responses = {"xai/grok-4.3": "I cannot decide", "anthropic/claude-sonnet-4-6": "A"}

    def handler(model_id, messages):
        return make_response(responses[model_id])

    patch_call_model(handler)
    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    result = await run_vote(council, "Pick one", ["Yes", "No"])

    assert result.vote is not None
    assert result.vote.winner == "A"
    assert result.vote.tally.get("A", 0) == 1
    assert result.vote.votes["grok"] is None


@pytest.mark.asyncio
async def test_vote_letter_embedded_in_prose_is_parsed(monkeypatch, patch_call_model):
    """A letter buried in extra text is still parsed out."""
    _set_keys(monkeypatch)

    responses = {"xai/grok-4.3": "  B  ", "anthropic/claude-sonnet-4-6": "My answer is B."}

    def handler(model_id, messages):
        return make_response(responses[model_id])

    patch_call_model(handler)
    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    result = await run_vote(council, "Pick", ["Alpha", "Beta"])

    assert result.vote is not None
    assert result.vote.winner == "B"


@pytest.mark.asyncio
async def test_vote_no_members_returns_empty(monkeypatch):
    """No available members → result with empty vote, no crash."""
    # Remove all keys so no member can run.
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    result = await run_vote(council, "Pick", ["Yes", "No"])

    assert result.vote is not None
    assert result.vote.winner is None
    assert result.vote.tally == {}


@pytest.mark.asyncio
async def test_vote_requires_at_least_two_choices(monkeypatch, patch_call_model):
    """Fewer than 2 choices raises ValueError."""
    _set_keys(monkeypatch)

    def handler(model_id, messages):
        return make_response("A")

    patch_call_model(handler)
    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    with pytest.raises(ValueError, match="at least 2 choices"):
        await run_vote(council, "Solo?", ["OnlyOne"])


@pytest.mark.asyncio
async def test_vote_choices_stored_on_result(monkeypatch, patch_call_model):
    """The choices list is preserved verbatim on VoteResult."""
    _set_keys(monkeypatch)

    def handler(model_id, messages):
        return make_response("A")

    patch_call_model(handler)
    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    choices = ["Option Alpha", "Option Beta", "Option Gamma"]
    result = await run_vote(council, "Which?", choices)

    assert result.vote is not None
    assert result.vote.choices == choices


@pytest.mark.asyncio
async def test_vote_failed_member_excluded_from_tally(monkeypatch, patch_call_model):
    """A member call that errors does not crash the vote; it is excluded."""
    _set_keys(monkeypatch)

    def handler(model_id, messages):
        if "grok" in model_id:
            raise RuntimeError("provider error")
        return make_response("B")

    patch_call_model(handler)
    cfg = _two_member_config()
    council = Council(models=cfg.resolve_council("default"), config=cfg)

    result = await run_vote(council, "Which?", ["Alpha", "Beta"])

    assert result.vote is not None
    assert result.vote.winner == "B"
    assert result.vote.votes.get("grok") is None


# ---------------------------------------------------------------------------
# _vote_summary helper tests
# ---------------------------------------------------------------------------


class TestVoteSummary:
    def test_winner_line_present(self):
        vr = VoteResult(
            choices=["Yes", "No"],
            votes={"m1": "A", "m2": "A"},
            tally={"A": 2},
            winner="A",
            split=False,
        )
        summary = _vote_summary(vr, {"A": "Yes", "B": "No"}, 2)
        assert "Winner" in summary
        assert "Yes" in summary

    def test_split_line_present(self):
        vr = VoteResult(
            choices=["Yes", "No"],
            votes={"m1": "A", "m2": "B"},
            tally={"A": 1, "B": 1},
            winner=None,
            split=True,
        )
        summary = _vote_summary(vr, {"A": "Yes", "B": "No"}, 2)
        assert "Tie" in summary or "split" in summary.lower()

    def test_no_votes_cast_line_present(self):
        vr = VoteResult(choices=["Yes", "No"], votes={}, tally={}, winner=None, split=False)
        summary = _vote_summary(vr, {"A": "Yes", "B": "No"}, 0)
        assert "No votes" in summary


# ---------------------------------------------------------------------------
# CLI integration tests (vote mode)
# ---------------------------------------------------------------------------


class TestCLIVoteMode:
    """CLI-level integration using CliRunner + the council mock."""

    def test_vote_mode_requires_choices(self, monkeypatch):
        from typer.testing import CliRunner

        from conclave.cli import app

        _set_keys(monkeypatch)
        runner = CliRunner()
        result = runner.invoke(app, ["ask", "Best?", "--mode", "vote"])
        assert result.exit_code == 2
        assert "--choices" in result.output or "--choices" in (result.stderr or "")

    def test_vote_mode_renders_winner(self, monkeypatch, patch_call_model):
        from typer.testing import CliRunner

        from conclave.cli import app

        _set_keys(monkeypatch)

        def handler(model_id, messages):
            return make_response("A")

        patch_call_model(handler)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["ask", "Best?", "--mode", "vote", "--choices", "Alpha,Beta", "--council", "grok,claude"],
        )
        assert result.exit_code == 0
        assert "Alpha" in result.output or "WINNER" in result.output
