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
from conclave.prompts import (
    ELITE_CRITIC_SYSTEM,
    ELITE_REVISION_SYSTEM,
    elite_critic_user,
    elite_revision_user,
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


def _elite_prompt_answers() -> list[ModelAnswer]:
    return [
        ModelAnswer(
            name="alpha-provider",
            model_id="vendor-one/secret-model",
            answer="Adopt option one because the measured risk is lower.",
            answer_id="initial-001",
        ),
        ModelAnswer(
            name="beta-provider",
            model_id="vendor-two/hidden-model",
            answer="Adopt option two because it preserves flexibility.",
            answer_id="initial-002",
        ),
        ModelAnswer(
            name="gamma-provider",
            model_id="vendor-three/private-model",
            answer="Delay the choice until the key assumption is tested.",
            answer_id="initial-003",
        ),
    ]


def test_elite_critic_prompt_is_anonymized_evidence_audit() -> None:
    answers = _elite_prompt_answers()

    built = elite_critic_user("Choose the strongest option.", answers)

    assert "Model A" in built
    assert "Model B" in built
    assert "Model C" in built
    assert "initial-001" in built
    assert "initial-002" in built
    assert "initial-003" in built
    for answer in answers:
        assert answer.name not in built
        assert answer.model_id not in built
    assert "SUPPORTED" in ELITE_CRITIC_SYSTEM
    assert "CONFLICTING" in ELITE_CRITIC_SYSTEM
    assert "EXTERNALLY UNVERIFIED" in ELITE_CRITIC_SYSTEM
    assert "Do not invent citations" in ELITE_CRITIC_SYSTEM


def test_elite_revision_prompt_includes_original_panel_and_critiques() -> None:
    answers = _elite_prompt_answers()
    critiques = [
        ModelAnswer(
            name=f"critic-{index}",
            model_id=f"critic-vendor/model-{index}",
            answer=f"Evidence audit {index}",
            answer_id=f"critique-{index:03d}",
        )
        for index in range(1, 4)
    ]

    built = elite_revision_user("Choose the strongest option.", answers[1], answers, critiques)

    assert "Your original answer" in built
    assert answers[1].answer in built
    assert "initial-002" in built
    assert all(f"Model {letter}" in built for letter in "ABC")
    assert all(critique.answer_id in built for critique in critiques)
    for artifact in [*answers, *critiques]:
        assert artifact.name not in built
        assert artifact.model_id not in built
    assert "Do not invent citations" in ELITE_REVISION_SYSTEM


def test_elite_prompt_builders_are_deterministic() -> None:
    answers = _elite_prompt_answers()
    critiques = [
        ModelAnswer(
            name="critic-a",
            model_id="vendor/model",
            answer="Audit the evidence base.",
            answer_id="critique-001",
        )
    ]

    assert elite_critic_user("Decide.", answers) == elite_critic_user("Decide.", answers)
    assert elite_revision_user("Decide.", answers[0], answers, critiques) == elite_revision_user(
        "Decide.", answers[0], answers, critiques
    )
