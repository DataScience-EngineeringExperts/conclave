"""Tests for the Elite Decision Protocol."""

import pytest
from pydantic import ValidationError

from conclave import (
    ELITE_MIN_RESPONDERS,
    ELITE_PROTOCOL_VERSION,
    CouncilResult,
    EliteResult,
    ModelAnswer,
)


def test_elite_result_defaults_to_incomplete_v1_protocol() -> None:
    result = EliteResult()

    assert ELITE_PROTOCOL_VERSION == "elite_v1"
    assert ELITE_MIN_RESPONDERS == 3
    assert result.protocol_version == ELITE_PROTOCOL_VERSION
    assert result.required_responders == ELITE_MIN_RESPONDERS
    assert result.completed is False
    assert result.failure_reason is None


def test_elite_result_rejects_fewer_than_three_required_responders() -> None:
    with pytest.raises(ValidationError):
        EliteResult(required_responders=2)


def test_elite_result_serializes_phase_artifacts() -> None:
    initial = ModelAnswer(name="member-a", model_id="provider/model", answer="initial")
    critique = ModelAnswer(name="member-b", model_id="provider/model", answer="critique")
    revision = ModelAnswer(name="member-a", model_id="provider/model", answer="revision")

    serialized = EliteResult(
        initial_answers=[initial],
        critiques=[critique],
        revisions=[revision],
    ).model_dump()

    assert serialized["initial_answers"][0]["answer"] == "initial"
    assert serialized["critiques"][0]["answer"] == "critique"
    assert serialized["revisions"][0]["answer"] == "revision"


def test_existing_council_result_defaults_elite_to_none() -> None:
    result = CouncilResult(prompt="Should we proceed?")

    assert result.elite is None
