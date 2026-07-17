"""Tests for the Elite Decision Protocol."""

import json

import pytest
from pydantic import ValidationError

from conclave import (
    ELITE_MIN_RESPONDERS,
    ELITE_PROTOCOL_VERSION,
    Council,
    CouncilResult,
    EliteResult,
    ModelAnswer,
)
from conclave.config import ConclaveConfig
from conclave.manifest import SECRET_SAFETY_VERIFIED
from conclave.modes import run_elite
from conclave.prompts import (
    ELITE_CRITIC_SYSTEM,
    ELITE_REVISION_SYSTEM,
    elite_critic_user,
    elite_revision_user,
)
from tests.conftest import make_response


def _all_keys(monkeypatch) -> None:
    for variable in (
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
    ):
        monkeypatch.setenv(variable, "dummy-key")


def _elite_config() -> ConclaveConfig:
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": "anthropic/claude-sonnet-4-6",
            "perplexity": "perplexity/sonar-pro",
        },
        councils={"default": ["grok", "gemini", "claude", "perplexity"]},
        synthesizer="claude",
    )


def _phase(messages) -> str:
    system = next((message["content"] for message in messages if message["role"] == "system"), "")
    if system == ELITE_CRITIC_SYSTEM:
        return "critique"
    if system == ELITE_REVISION_SYSTEM:
        return "revision"
    return "initial"


def _is_verdict_call(messages) -> bool:
    return bool(messages) and messages[0].get("content", "").startswith(
        "You are the verdict extractor"
    )


def _elite_verdict_json() -> str:
    members = ["grok", "gemini", "perplexity"]
    return json.dumps(
        {
            "verdict_applies": True,
            "verdict_type": "decision",
            "headline": "Proceed.",
            "recommendation": "Choose the strongest option.",
            "positions": [
                {
                    "label": "proceed",
                    "summary": "The revised council agrees.",
                    "providers": members,
                    "evidence_answer_ids": [],
                }
            ],
            "provider_votes": [
                {"provider": member, "position_label": "proceed"} for member in members
            ],
            "minority_reports": [],
            "conflicts": [],
            "caveats": [],
            "dissent_summary": None,
        }
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


def test_elite_critic_prompt_is_anonymized_claim_audit() -> None:
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
    assert "claim auditor" in ELITE_CRITIC_SYSTEM
    assert "within-run answer provenance" in ELITE_CRITIC_SYSTEM
    assert "evidence auditor" not in ELITE_CRITIC_SYSTEM


def test_elite_revision_prompt_includes_original_panel_and_critiques() -> None:
    answers = _elite_prompt_answers()
    critiques = [
        ModelAnswer(
            name=f"critic-{index}",
            model_id=f"critic-vendor/model-{index}",
            answer=f"Claim audit {index}",
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


async def test_run_elite_executes_three_stages_and_captures_artifacts(
    monkeypatch, patch_call_model
) -> None:
    _all_keys(monkeypatch)
    calls: list[tuple[str, str]] = []

    def handler(model_id, messages):
        phase = _phase(messages)
        calls.append((phase, model_id))
        return make_response(f"{phase} from {model_id}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_elite_config(),
        extract_verdict=False,
    )

    result = await run_elite(council, "Choose the strongest option.")

    assert result.mode == "elite"
    assert result.elite is not None
    assert result.elite.completed is True
    assert result.elite.failure_reason is None
    assert len(result.elite.initial_answers) == 3
    assert len(result.elite.critiques) == 3
    assert len(result.elite.revisions) == 3
    assert [phase for phase, _model in calls] == ["initial"] * 3 + ["critique"] * 3 + [
        "revision"
    ] * 3
    assert all(answer.ok for answer in result.answers)
    assert all(answer.answer.startswith("revision") for answer in result.answers)


async def test_run_elite_assigns_phase_specific_ids_and_preserves_them_round_trip(
    monkeypatch, patch_call_model
) -> None:
    _all_keys(monkeypatch)

    def handler(model_id, messages):
        return make_response(f"{_phase(messages)} from {model_id}")

    patch_call_model(handler)
    result = await run_elite(
        Council(
            models=["grok", "gemini", "perplexity"],
            config=_elite_config(),
            extract_verdict=False,
        ),
        "Decide.",
    )

    assert result.elite is not None
    phase_artifacts = {
        "initial": result.elite.initial_answers,
        "critique": result.elite.critiques,
        "revision": result.elite.revisions,
    }
    for phase, artifacts in phase_artifacts.items():
        assert all((answer.answer_id or "").startswith(f"ca_{phase}_") for answer in artifacts)
    assert [answer.answer_id for answer in result.answers] == [
        answer.answer_id for answer in result.elite.revisions if answer.ok
    ]

    reconstructed = CouncilResult.model_validate_json(result.model_dump_json())
    assert reconstructed.elite is not None
    assert [answer.answer_id for answer in reconstructed.elite.initial_answers] == [
        answer.answer_id for answer in result.elite.initial_answers
    ]
    assert [answer.answer_id for answer in reconstructed.elite.critiques] == [
        answer.answer_id for answer in result.elite.critiques
    ]
    assert [answer.answer_id for answer in reconstructed.elite.revisions] == [
        answer.answer_id for answer in result.elite.revisions
    ]


@pytest.mark.parametrize("failed_phase", ["initial", "critique", "revision"])
async def test_run_elite_four_members_survives_one_failure_in_any_phase(
    monkeypatch, patch_call_model, failed_phase
) -> None:
    _all_keys(monkeypatch)

    def handler(model_id, messages):
        phase = _phase(messages)
        if phase == failed_phase and model_id == "gemini/gemini-2.5-pro":
            raise RuntimeError(f"{phase} provider failure")
        return make_response(f"{phase} from {model_id}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "claude", "perplexity"],
        config=_elite_config(),
        extract_verdict=False,
    )

    result = await run_elite(council, "Decide.")

    assert result.elite is not None
    assert result.elite.completed is True
    assert len(result.successful_answers) == 3
    assert all(answer.answer.startswith("revision") for answer in result.answers)
    phase_artifacts = {
        "initial": result.elite.initial_answers,
        "critique": result.elite.critiques,
        "revision": result.elite.revisions,
    }
    assert sum(not answer.ok for answer in phase_artifacts[failed_phase]) == 1


async def test_run_elite_stops_after_failed_initial_gate(monkeypatch, patch_call_model) -> None:
    _all_keys(monkeypatch)
    phases: list[str] = []

    def handler(model_id, messages):
        phase = _phase(messages)
        phases.append(phase)
        if model_id == "gemini/gemini-2.5-pro":
            raise RuntimeError("initial unavailable")
        return make_response(f"{phase} from {model_id}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_elite_config(),
        extract_verdict=False,
    )

    result = await run_elite(council, "Decide.")

    assert result.elite is not None
    assert result.elite.completed is False
    assert result.elite.failure_reason == "initial phase required 3 successful responders; got 2"
    assert phases == ["initial"] * 3
    assert result.elite.critiques == []
    assert result.elite.revisions == []
    assert len(result.answers) == 2
    assert all(answer.answer.startswith("initial") for answer in result.answers)


async def test_run_elite_stops_after_failed_critique_gate(monkeypatch, patch_call_model) -> None:
    _all_keys(monkeypatch)
    phases: list[str] = []

    def handler(model_id, messages):
        phase = _phase(messages)
        phases.append(phase)
        if phase == "critique" and model_id == "gemini/gemini-2.5-pro":
            raise RuntimeError("critic unavailable")
        return make_response(f"{phase} from {model_id}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_elite_config(),
        extract_verdict=False,
    )

    result = await run_elite(council, "Decide.")

    assert result.elite is not None
    assert result.elite.completed is False
    assert result.elite.failure_reason == "critique phase required 3 successful responders; got 2"
    assert phases == ["initial"] * 3 + ["critique"] * 3
    assert len(result.elite.critiques) == 3
    assert result.elite.revisions == []
    assert len(result.answers) == 3
    assert all(answer.answer.startswith("initial") for answer in result.answers)


async def test_run_elite_stops_after_failed_revision_gate(monkeypatch, patch_call_model) -> None:
    _all_keys(monkeypatch)

    def handler(model_id, messages):
        phase = _phase(messages)
        if phase == "revision" and model_id == "gemini/gemini-2.5-pro":
            raise RuntimeError("revision unavailable")
        return make_response(f"{phase} from {model_id}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_elite_config(),
        extract_verdict=False,
    )

    result = await run_elite(council, "Decide.")

    assert result.elite is not None
    assert result.elite.completed is False
    assert result.elite.failure_reason == "revision phase required 3 successful responders; got 2"
    assert len(result.elite.revisions) == 3
    assert len(result.answers) == 3
    assert all(answer.answer.startswith("initial") for answer in result.answers)


async def test_council_elite_synthesizes_revisions_and_applies_canonical_verdict(
    monkeypatch, patch_call_model
) -> None:
    _all_keys(monkeypatch)
    synthesis_inputs: list[str] = []

    def handler(model_id, messages):
        if _is_verdict_call(messages):
            return make_response(_elite_verdict_json())
        system = next(
            (message["content"] for message in messages if message["role"] == "system"), ""
        )
        if system.startswith("You are the synthesizer of a council"):
            synthesis_inputs.append(messages[-1]["content"])
            return make_response("elite synthesis")
        phase = _phase(messages)
        return make_response(f"{phase} from {model_id}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_elite_config(),
    )

    result = await council.elite("Choose the strongest option.")

    assert result.elite is not None
    assert result.elite.completed is True
    assert result.synthesis == "elite synthesis"
    assert len(synthesis_inputs) == 1
    assert "revision from xai/grok-4.3" in synthesis_inputs[0]
    assert "initial from xai/grok-4.3" not in synthesis_inputs[0]
    assert all(f"Answer ID: {answer.answer_id}" in synthesis_inputs[0] for answer in result.answers)
    assert result.verdict is not None
    assert result.consensus_score == result.verdict.consensus_score == 1.0
    assert result.provider_votes == result.verdict.provider_votes
    assert result.manifest is not None
    assert result.manifest.verdict_extraction.model_id == "anthropic/claude-sonnet-4-6"
    assert result.manifest.verdict_extraction.prompt_version is not None
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED


async def test_council_elite_incomplete_skips_synthesis_and_verdict(
    monkeypatch, patch_call_model
) -> None:
    _all_keys(monkeypatch)
    calls = {"synthesis": 0, "verdict": 0}

    def handler(model_id, messages):
        if model_id == "gemini/gemini-2.5-pro":
            raise RuntimeError("initial unavailable")
        return make_response(f"initial from {model_id}")

    async def unexpected_synthesis(_result):
        calls["synthesis"] += 1

    async def unexpected_verdict(_result):
        calls["verdict"] += 1

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_elite_config(),
    )
    monkeypatch.setattr(council, "_synthesize", unexpected_synthesis)
    monkeypatch.setattr(council, "_apply_verdict", unexpected_verdict)

    result = await council.elite("Choose.")

    assert result.elite is not None
    assert result.elite.completed is False
    assert result.synthesis is None
    assert result.verdict is None
    assert calls == {"synthesis": 0, "verdict": 0}


def test_council_elite_sync_exposes_completed_protocol(monkeypatch, patch_call_model) -> None:
    _all_keys(monkeypatch)

    def handler(model_id, messages):
        return make_response(f"{_phase(messages)} from {model_id}")

    patch_call_model(handler)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        config=_elite_config(),
        extract_verdict=False,
    )

    result = council.elite_sync("Choose.")

    assert result.elite is not None
    assert result.elite.completed is True
    assert result.synthesis is not None
