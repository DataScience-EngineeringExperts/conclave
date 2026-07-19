from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import pytest

from conclave import prompts
from conclave.evals.live_protocols import (
    LIVE_PROMPT_VERSIONS,
    StageCall,
    allocate_stage_caps,
    execute_live_condition,
    stage_call_sequence,
)
from conclave.evals.models import EVAL_CONDITION_IDS, ProviderModelSpec, PublicTask
from conclave.evals.protocols import live_protocol_registry
from conclave.models import ELITE_MIN_RESPONDERS, ModelAnswer
from conclave.verdict import CouncilVerdict, verdict_extraction_json_schema


def _roster(size: int = 3) -> tuple[ProviderModelSpec, ...]:
    return tuple(
        ProviderModelSpec(
            provider_id=f"fictional-provider-{index}",
            model_id=f"fictional-model-{index}",
            model_revision=f"fixture-r{index}",
        )
        for index in range(1, size + 1)
    )


def _task() -> PublicTask:
    return PublicTask(
        task_id="public-decision",
        prompt="Choose the strongest public option.",
        reference_packets=("Public packet one.", "Public packet two."),
    )


class RecordingSequentialClient:
    def __init__(self, *, fail_stages: Sequence[tuple[str, str]] = ()) -> None:
        self.calls: list[StageCall] = []
        self.active = 0
        self.max_active = 0
        self._fail_stages = set(fail_stages)

    async def call(self, call: StageCall) -> ModelAnswer:
        self.calls.append(call)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if (call.stage, call.provider_id) in self._fail_stages:
            return ModelAnswer(
                name=call.provider_id,
                model_id=call.model_id,
                error="bounded fake failure",
            )
        if call.stage in {"verdict", "verdict_repair"}:
            return ModelAnswer(
                name=call.provider_id,
                model_id=call.model_id,
                answer=_valid_verdict_extraction(),
            )
        return ModelAnswer(
            name=call.provider_id,
            model_id=call.model_id,
            answer=f"decision-artifact:{call.stage}:{len(self.calls)}",
        )


def _valid_verdict_extraction() -> str:
    return json.dumps(
        {
            "verdict_applies": True,
            "verdict_type": "decision",
            "headline": "Choose option A.",
            "recommendation": "Adopt option A with the stated safeguards.",
            "positions": [
                {
                    "label": "option-a",
                    "summary": "Prefer option A.",
                    "providers": ["Model A", "Model B"],
                    "evidence_answer_ids": ["answer-a", "answer-b"],
                },
                {
                    "label": "option-b",
                    "summary": "Prefer option B.",
                    "providers": ["Model C"],
                    "evidence_answer_ids": ["answer-c"],
                },
            ],
            "provider_votes": [
                {"provider": "Model A", "position_label": "option-a"},
                {"provider": "Model B", "position_label": "option-a"},
                {"provider": "Model C", "position_label": "option-b"},
            ],
            "conflicts": [
                {
                    "topic": "preferred option",
                    "position_labels": ["option-a", "option-b"],
                    "summary": "The panel split between A and B.",
                }
            ],
            "minority_reports": [],
            "caveats": [],
            # The canonical engine must ignore this model-emitted number and
            # deterministically recompute 2 / 3 from the provider votes.
            "consensus_score": 0.01,
        }
    )


class ScriptedVerdictClient(RecordingSequentialClient):
    def __init__(self, *verdict_responses: str) -> None:
        super().__init__()
        self._verdict_responses = iter(verdict_responses)

    async def call(self, call: StageCall) -> ModelAnswer:
        if call.stage not in {"verdict", "verdict_repair"}:
            return await super().call(call)
        self.calls.append(call)
        return ModelAnswer(
            name=call.provider_id,
            model_id=call.model_id,
            answer=next(self._verdict_responses),
        )


def _message_text(call: StageCall) -> str:
    return "\n".join(message.content for message in call.messages)


def test_live_registry_covers_exactly_six_versioned_conditions() -> None:
    registry = live_protocol_registry()

    assert tuple(registry) == EVAL_CONDITION_IDS
    assert all(entry.protocol_version.endswith("_v2") for entry in registry.values())
    assert set(LIVE_PROMPT_VERSIONS) == set(EVAL_CONDITION_IDS)
    assert all(LIVE_PROMPT_VERSIONS[condition] for condition in EVAL_CONDITION_IDS)


def test_stage_caps_are_positive_deterministic_and_sum_to_cell_ceiling() -> None:
    import conclave.evals.live_protocols as live_protocols

    assert live_protocols.LIVE_STAGE_MINIMUM_CAPS_VERSION == "live_stage_minimum_caps_v1"
    assert all(cap > 1 for cap in live_protocols.LIVE_STAGE_MINIMUM_CAPS.values())
    for condition_id in EVAL_CONDITION_IDS:
        call_count = len(stage_call_sequence(condition_id, roster_size=3))
        minimum = live_protocols.minimum_cell_ceiling(condition_id, roster_size=3)
        ceiling = minimum + call_count * 17 + 5

        first = allocate_stage_caps(condition_id, roster_size=3, cell_ceiling=ceiling)
        second = allocate_stage_caps(condition_id, roster_size=3, cell_ceiling=ceiling)

        assert first == second
        assert len(first) == call_count
        assert all(cap > 0 for cap in first)
        assert sum(first) == ceiling
        assert all(
            cap >= live_protocols.LIVE_STAGE_MINIMUM_CAPS[stage]
            for cap, (stage, _member_index) in zip(
                first,
                stage_call_sequence(condition_id, roster_size=3),
                strict=True,
            )
        )


@pytest.mark.asyncio
async def test_elite_uses_guarded_canonical_verdict_and_skips_repair_when_valid() -> None:
    client = ScriptedVerdictClient(_valid_verdict_extraction())

    output = await execute_live_condition(
        "elite_full",
        task=_task(),
        roster=_roster(),
        cell_ceiling=6144,
        client=client,
    )

    verdict_calls = [call for call in client.calls if call.stage.startswith("verdict")]
    assert [call.stage for call in verdict_calls] == ["verdict"]
    assert verdict_calls[0].output_contract is not None
    assert verdict_calls[0].output_contract.strict is True
    assert verdict_calls[0].output_contract.schema == verdict_extraction_json_schema()
    verdict = CouncilVerdict.model_validate_json(output)
    assert verdict.recommendation == "Adopt option A with the stated safeguards."
    assert verdict.consensus_score == pytest.approx(2 / 3)
    assert verdict.consensus_label == "majority"


@pytest.mark.asyncio
async def test_elite_repairs_malformed_verdict_once_through_guarded_gateway() -> None:
    client = ScriptedVerdictClient("not-json", _valid_verdict_extraction())

    output = await execute_live_condition(
        "elite_full",
        task=_task(),
        roster=_roster(),
        cell_ceiling=6144,
        client=client,
    )

    verdict_calls = [call for call in client.calls if call.stage.startswith("verdict")]
    assert [call.stage for call in verdict_calls] == ["verdict", "verdict_repair"]
    assert all(call.output_contract is not None for call in verdict_calls)
    assert "could not be used" in _message_text(verdict_calls[-1])
    assert CouncilVerdict.model_validate_json(output).consensus_score == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_elite_failed_verdict_repair_is_non_success_and_never_returns_raw_text() -> None:
    client = ScriptedVerdictClient("first malformed", "repair also malformed")

    with pytest.raises(ValueError, match="canonical CouncilVerdict"):
        await execute_live_condition(
            "elite_full",
            task=_task(),
            roster=_roster(),
            cell_ceiling=6144,
            client=client,
        )

    verdict_calls = [call for call in client.calls if call.stage.startswith("verdict")]
    assert [call.stage for call in verdict_calls] == ["verdict", "verdict_repair"]


@pytest.mark.asyncio
async def test_single_and_self_refine_use_only_frozen_lead_member() -> None:
    roster = _roster()
    expected_stages = {
        "single_frontier": ["initial"],
        "self_refine": ["draft", "self_revision"],
    }

    for condition_id, stages in expected_stages.items():
        client = RecordingSequentialClient()
        output = await execute_live_condition(
            condition_id,
            task=_task(),
            roster=roster,
            cell_ceiling=1024,
            client=client,
        )

        assert [(call.stage, call.provider_id) for call in client.calls] == [
            (stage, roster[0].provider_id) for stage in stages
        ]
        assert output == f"decision-artifact:{stages[-1]}:{len(stages)}"
        assert client.max_active == 1


@pytest.mark.asyncio
async def test_multi_model_conditions_call_members_in_frozen_order() -> None:
    roster = _roster()
    providers = [member.provider_id for member in roster]
    expected = {
        "independent_synthesis": [
            *(("initial", provider) for provider in providers),
            ("synthesis", providers[0]),
        ],
        "critique_only": [
            *(("initial", provider) for provider in providers),
            *(("critique", provider) for provider in providers),
            ("synthesis", providers[0]),
        ],
        "revision_only": [
            *(("initial", provider) for provider in providers),
            *(("revision", provider) for provider in providers),
            ("synthesis", providers[0]),
        ],
        "elite_full": [
            *(("initial", provider) for provider in providers),
            *(("critique", provider) for provider in providers),
            *(("revision", provider) for provider in providers),
            ("synthesis", providers[0]),
            ("verdict", providers[0]),
        ],
    }

    for condition_id, expected_calls in expected.items():
        client = RecordingSequentialClient()
        output = await execute_live_condition(
            condition_id,
            task=_task(),
            roster=roster,
            cell_ceiling=6144,
            client=client,
        )

        assert [(call.stage, call.provider_id) for call in client.calls] == expected_calls
        if condition_id == "elite_full":
            assert CouncilVerdict.model_validate_json(output).consensus_score == pytest.approx(
                2 / 3
            )
        else:
            assert output == f"decision-artifact:{expected_calls[-1][0]}:{len(expected_calls)}"
        assert client.max_active == 1


@pytest.mark.asyncio
async def test_critique_revision_and_elite_prompts_are_anonymized() -> None:
    roster = _roster()

    for condition_id in ("critique_only", "revision_only", "elite_full"):
        client = RecordingSequentialClient()
        await execute_live_condition(
            condition_id,
            task=_task(),
            roster=roster,
            cell_ceiling=6144,
            client=client,
        )

        peer_calls = [call for call in client.calls if call.stage != "initial"]
        for call in peer_calls:
            text = _message_text(call)
            for member in roster:
                assert member.provider_id not in text
                assert member.model_id not in text
                assert member.model_revision not in text
        assert any("Model A" in _message_text(call) for call in peer_calls)


@pytest.mark.asyncio
async def test_elite_uses_current_versioned_prompt_builders_and_three_responder_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_critic = prompts.elite_critic_user
    original_revision = prompts.elite_revision_user
    builder_calls = {"critic": 0, "revision": 0}

    def tracked_critic(*args, **kwargs):
        builder_calls["critic"] += 1
        return original_critic(*args, **kwargs)

    def tracked_revision(*args, **kwargs):
        builder_calls["revision"] += 1
        return original_revision(*args, **kwargs)

    monkeypatch.setattr(prompts, "elite_critic_user", tracked_critic)
    monkeypatch.setattr(prompts, "elite_revision_user", tracked_revision)
    roster = _roster()
    client = RecordingSequentialClient()

    await execute_live_condition(
        "elite_full",
        task=_task(),
        roster=roster,
        cell_ceiling=6144,
        client=client,
    )

    assert ELITE_MIN_RESPONDERS == 3
    assert prompts.ELITE_PROMPT_VERSION in LIVE_PROMPT_VERSIONS["elite_full"]
    assert builder_calls == {"critic": 1, "revision": 3}

    failed = RecordingSequentialClient(fail_stages=(("initial", roster[-1].provider_id),))
    with pytest.raises(ValueError, match="initial phase required 3 successful responders; got 2"):
        await execute_live_condition(
            "elite_full",
            task=_task(),
            roster=roster,
            cell_ceiling=6144,
            client=failed,
        )
    assert [call.stage for call in failed.calls] == ["initial"] * 3

    too_small_roster_client = RecordingSequentialClient()
    with pytest.raises(ValueError, match="at least three"):
        await execute_live_condition(
            "elite_full",
            task=_task(),
            roster=_roster(2),
            cell_ceiling=6144,
            client=too_small_roster_client,
        )
    assert too_small_roster_client.calls == []


@pytest.mark.asyncio
async def test_too_small_cell_budget_fails_before_any_call() -> None:
    import conclave.evals.live_protocols as live_protocols

    roster = _roster()

    for condition_id in EVAL_CONDITION_IDS:
        client = RecordingSequentialClient()
        too_small = live_protocols.minimum_cell_ceiling(condition_id, roster_size=len(roster)) - 1

        with pytest.raises(ValueError, match="too small"):
            await execute_live_condition(
                condition_id,
                task=_task(),
                roster=roster,
                cell_ceiling=too_small,
                client=client,
            )

        assert client.calls == []
