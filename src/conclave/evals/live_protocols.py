"""Sequential, versioned call graphs for the six paid evaluation conditions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

from pydantic import Field

from conclave import prompts
from conclave.council import _SYNTH_SYSTEM
from conclave.models import ELITE_MIN_RESPONDERS, ModelAnswer, derive_phase_answer_id
from conclave.verdict import VERDICT_EXTRACTION_PROMPT_VERSION
from conclave.verdict_synthesis import _build_messages as _verdict_messages

from .models import (
    EVAL_CONDITION_IDS,
    ConditionId,
    EvalModel,
    ProviderModelSpec,
    PublicTask,
)

StageName = Literal[
    "initial",
    "draft",
    "self_revision",
    "critique",
    "revision",
    "synthesis",
    "verdict",
]


class ChatMessage(EvalModel):
    """One immutable OpenAI-style prompt message."""

    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class StageCall(EvalModel):
    """One fully resolved, sequential provider call in a live condition."""

    stage: StageName
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    upstream_output_token_ceilings: tuple[int, ...] = ()


class SequentialStageClient(Protocol):
    """Narrow client contract implemented by the guarded live gateway."""

    def call(self, call: StageCall) -> Awaitable[ModelAnswer]: ...


@dataclass(frozen=True)
class LiveProtocolSpec:
    """Version metadata for one frozen live condition graph."""

    condition_id: ConditionId
    protocol_version: str
    prompt_version: str
    executor: Callable[[_Execution, str], Awaitable[str]]


_STAGE_ALLOCATION_TABLE: Mapping[
    ConditionId, tuple[tuple[StageName, Literal["lead", "all"]], ...]
] = MappingProxyType(
    {
        "single_frontier": (("initial", "lead"),),
        "self_refine": (("draft", "lead"), ("self_revision", "lead")),
        "independent_synthesis": (("initial", "all"), ("synthesis", "lead")),
        "critique_only": (
            ("initial", "all"),
            ("critique", "all"),
            ("synthesis", "lead"),
        ),
        "revision_only": (
            ("initial", "all"),
            ("revision", "all"),
            ("synthesis", "lead"),
        ),
        "elite_full": (
            ("initial", "all"),
            ("critique", "all"),
            ("revision", "all"),
            ("synthesis", "lead"),
            ("verdict", "lead"),
        ),
    }
)

LIVE_PROMPT_VERSIONS: Mapping[ConditionId, str] = MappingProxyType(
    {
        "single_frontier": "live_single_frontier_prompt_v1",
        "self_refine": "live_self_refine_prompt_v1",
        "independent_synthesis": f"live_synthesis_{prompts.SYNTHESIS_PROMPT_VERSION}",
        "critique_only": (
            f"live_critique_{prompts.ELITE_PROMPT_VERSION}_synthesis_"
            f"{prompts.SYNTHESIS_PROMPT_VERSION}"
        ),
        "revision_only": (f"live_revision_v1_synthesis_{prompts.SYNTHESIS_PROMPT_VERSION}"),
        "elite_full": (
            f"elite_{prompts.ELITE_PROMPT_VERSION}_synthesis_"
            f"{prompts.SYNTHESIS_PROMPT_VERSION}_verdict_{VERDICT_EXTRACTION_PROMPT_VERSION}"
        ),
    }
)


def stage_call_sequence(
    condition_id: ConditionId, *, roster_size: int
) -> tuple[tuple[StageName, int], ...]:
    """Expand a condition's fixed stage layout into stage/member call positions."""

    if isinstance(roster_size, bool) or not isinstance(roster_size, int) or roster_size < 1:
        raise ValueError("roster_size must be a positive integer")
    return tuple(
        (stage, member_index)
        for stage, participants in _STAGE_ALLOCATION_TABLE[condition_id]
        for member_index in (range(roster_size) if participants == "all" else range(1))
    )


def allocate_stage_caps(
    condition_id: ConditionId, *, roster_size: int, cell_ceiling: int
) -> tuple[int, ...]:
    """Allocate positive equal caps, assigning integer remainder to graded output."""

    if isinstance(cell_ceiling, bool) or not isinstance(cell_ceiling, int):
        raise TypeError("cell_ceiling must be an integer")
    call_count = len(stage_call_sequence(condition_id, roster_size=roster_size))
    if cell_ceiling < call_count:
        raise ValueError(
            f"cell ceiling {cell_ceiling} is too small for {call_count} positive stage caps"
        )
    per_call, remainder = divmod(cell_ceiling, call_count)
    return (*([per_call] * (call_count - 1)), per_call + remainder)


def _public_prompt(task: PublicTask) -> str:
    packets = "\n\n".join(
        f"### Reference packet {index}\n{packet}"
        for index, packet in enumerate(task.reference_packets, start=1)
    )
    return (
        task.prompt if not packets else f"{task.prompt}\n\nPublic reference packets:\n\n{packets}"
    )


def _messages(user: str, *, system: str | None = None) -> tuple[ChatMessage, ...]:
    items = ([ChatMessage(role="system", content=system)] if system else []) + [
        ChatMessage(role="user", content=user)
    ]
    return tuple(items)


def _anonymized(
    answer: ModelAnswer, *, phase: str, index: int, parents: Sequence[str] = ()
) -> ModelAnswer:
    artifact = answer.model_copy(
        update={"name": f"Model {prompts.LETTERS[index]}", "model_id": "anonymized"}
    )
    return derive_phase_answer_id(artifact, phase, parent_answer_ids=parents)


def _successful(
    answers: Sequence[tuple[int, ModelAnswer]], *, phase: str
) -> list[tuple[int, ModelAnswer]]:
    return [
        (index, _anonymized(answer, phase=phase, index=index))
        for index, answer in answers
        if answer.ok and answer.answer and answer.answer.strip()
    ]


def _synthesis_user(
    public_prompt: str,
    artifacts: Sequence[ModelAnswer],
    *,
    audits: Sequence[ModelAnswer] = (),
) -> str:
    panel = "\n\n".join(
        f"### Model {prompts.LETTERS[index]} decision artifact\n{artifact.answer}"
        for index, artifact in enumerate(artifacts)
    )
    audit_panel = "\n\n".join(
        f"### Model {prompts.LETTERS[index]} anonymized audit\n{artifact.answer}"
        for index, artifact in enumerate(audits)
    )
    audit_section = f"\n\nAnonymized claim audits:\n\n{audit_panel}" if audit_panel else ""
    return (
        f"Original prompt and public material:\n{public_prompt}\n\n"
        f"Anonymized decision artifacts:\n\n{panel}{audit_section}\n\n"
        "Produce the consolidated decision artifact."
    )


def _revision_only_user(
    public_prompt: str,
    original: ModelAnswer,
    panel: Sequence[ModelAnswer],
) -> str:
    peer_panel = "\n\n".join(
        f"### Model {prompts.LETTERS[index]} peer answer\n{artifact.answer}"
        for index, artifact in enumerate(panel)
    )
    return (
        f"Original prompt and public material:\n{public_prompt}\n\n"
        f"Your original answer:\n{original.answer}\n\n"
        f"Anonymized peer panel:\n\n{peer_panel}\n\n"
        "Produce one complete revised answer to the original prompt."
    )


class _Execution:
    def __init__(
        self,
        condition_id: ConditionId,
        roster: Sequence[ProviderModelSpec],
        caps: Sequence[int],
        client: SequentialStageClient,
    ) -> None:
        sequence = stage_call_sequence(condition_id, roster_size=len(roster))
        self._caps = dict(zip(sequence, caps, strict=True))
        self.roster = tuple(roster)
        self.client = client

    async def call(
        self,
        stage: StageName,
        member_index: int,
        messages: tuple[ChatMessage, ...],
        *,
        upstream: Sequence[int] = (),
    ) -> ModelAnswer:
        member = self.roster[member_index]
        return await self.client.call(
            StageCall(
                stage=stage,
                provider_id=member.provider_id,
                model_id=member.model_id,
                model_revision=member.model_revision,
                messages=messages,
                max_output_tokens=self._caps[(stage, member_index)],
                upstream_output_token_ceilings=tuple(upstream),
            )
        )

    async def all(
        self,
        stage: StageName,
        messages_for: Callable[[int], tuple[ChatMessage, ...]],
        *,
        members: Sequence[int] | None = None,
        upstream: Sequence[int] = (),
    ) -> list[tuple[int, ModelAnswer]]:
        indexes = tuple(range(len(self.roster))) if members is None else tuple(members)
        return [
            (index, await self.call(stage, index, messages_for(index), upstream=upstream))
            for index in indexes
        ]

    def caps(self, stage: StageName, members: Sequence[int]) -> tuple[int, ...]:
        return tuple(self._caps[(stage, index)] for index in members)


def _required(answer: ModelAnswer, stage: str) -> str:
    if not answer.ok or not answer.answer or not answer.answer.strip():
        raise ValueError(f"{stage} stage did not produce a decision artifact")
    return answer.answer


def _elite_gate(
    phase: str, answers: Sequence[tuple[int, ModelAnswer]]
) -> list[tuple[int, ModelAnswer]]:
    successful = _successful(answers, phase=phase)
    if len(successful) < ELITE_MIN_RESPONDERS:
        raise ValueError(
            f"{phase} phase required {ELITE_MIN_RESPONDERS} successful responders; "
            f"got {len(successful)}"
        )
    return successful


async def _initial_panel(run: _Execution, public_prompt: str) -> list[tuple[int, ModelAnswer]]:
    answers = await run.all("initial", lambda _index: _messages(public_prompt))
    successful = _successful(answers, phase="initial")
    if not successful:
        raise ValueError("initial phase produced no successful responders")
    return successful


async def execute_single_frontier(run: _Execution, public_prompt: str) -> str:
    answer = await run.call("initial", 0, _messages(public_prompt))
    return _required(answer, "initial")


async def execute_self_refine(run: _Execution, public_prompt: str) -> str:
    draft = await run.call("draft", 0, _messages(public_prompt))
    draft_text = _required(draft, "draft")
    revision = await run.call(
        "self_revision",
        0,
        _messages(
            f"Original prompt and public material:\n{public_prompt}\n\n"
            f"Your draft:\n{draft_text}\n\nProduce one complete, improved final answer."
        ),
        upstream=run.caps("draft", (0,)),
    )
    return _required(revision, "self_revision")


async def execute_independent_synthesis(run: _Execution, public_prompt: str) -> str:
    initial = await _initial_panel(run, public_prompt)
    artifacts = [answer for _index, answer in initial]
    synthesis = await run.call(
        "synthesis",
        0,
        _messages(_synthesis_user(public_prompt, artifacts), system=_SYNTH_SYSTEM),
        upstream=run.caps("initial", [index for index, _answer in initial]),
    )
    return _required(synthesis, "synthesis")


async def execute_critique_only(run: _Execution, public_prompt: str) -> str:
    initial = await _initial_panel(run, public_prompt)
    members = [index for index, _answer in initial]
    artifacts = [answer for _index, answer in initial]
    critique_messages = _messages(
        prompts.elite_critic_user(public_prompt, artifacts),
        system=prompts.ELITE_CRITIC_SYSTEM,
    )
    critiques = _successful(
        await run.all(
            "critique",
            lambda _index: critique_messages,
            members=members,
            upstream=run.caps("initial", members),
        ),
        phase="critique",
    )
    synthesis = await run.call(
        "synthesis",
        0,
        _messages(
            _synthesis_user(
                public_prompt,
                artifacts,
                audits=[answer for _index, answer in critiques],
            ),
            system=_SYNTH_SYSTEM,
        ),
        upstream=(
            *run.caps("initial", members),
            *run.caps("critique", [index for index, _answer in critiques]),
        ),
    )
    return _required(synthesis, "synthesis")


async def execute_revision_only(run: _Execution, public_prompt: str) -> str:
    initial = await _initial_panel(run, public_prompt)
    members = [index for index, _answer in initial]
    by_member = dict(initial)
    artifacts = [answer for _index, answer in initial]
    revisions = _successful(
        await run.all(
            "revision",
            lambda index: _messages(
                _revision_only_user(public_prompt, by_member[index], artifacts)
            ),
            members=members,
            upstream=run.caps("initial", members),
        ),
        phase="revision",
    )
    revision_artifacts = [answer for _index, answer in revisions]
    synthesis = await run.call(
        "synthesis",
        0,
        _messages(_synthesis_user(public_prompt, revision_artifacts), system=_SYNTH_SYSTEM),
        upstream=run.caps("revision", [index for index, _answer in revisions]),
    )
    return _required(synthesis, "synthesis")


async def execute_elite_full(run: _Execution, public_prompt: str) -> str:
    initial_raw = await run.all("initial", lambda _index: _messages(public_prompt))
    initial = _elite_gate("initial", initial_raw)
    members = [index for index, _answer in initial]
    initial_artifacts = [answer for _index, answer in initial]
    critique_messages = _messages(
        prompts.elite_critic_user(public_prompt, initial_artifacts),
        system=prompts.ELITE_CRITIC_SYSTEM,
    )
    critique_raw = await run.all(
        "critique",
        lambda _index: critique_messages,
        members=members,
        upstream=run.caps("initial", members),
    )
    critiques = _elite_gate("critique", critique_raw)
    critique_members = [index for index, _answer in critiques]
    critique_artifacts = [answer for _index, answer in critiques]
    initial_by_member = dict(initial)
    parent_ids = tuple(
        answer.answer_id or "" for answer in [*initial_artifacts, *critique_artifacts]
    )
    revision_raw = await run.all(
        "revision",
        lambda index: _messages(
            prompts.elite_revision_user(
                public_prompt,
                initial_by_member[index],
                initial_artifacts,
                critique_artifacts,
            ),
            system=prompts.ELITE_REVISION_SYSTEM,
        ),
        members=critique_members,
        upstream=(
            *run.caps("initial", members),
            *run.caps("critique", critique_members),
        ),
    )
    revisions = _elite_gate("revision", revision_raw)
    revision_artifacts = [
        (
            index,
            _anonymized(answer, phase="revision", index=index, parents=parent_ids),
        )
        for index, answer in revision_raw
        if answer.ok and answer.answer and answer.answer.strip()
    ]
    revision_answers = [answer for _index, answer in revision_artifacts]
    synthesis = await run.call(
        "synthesis",
        0,
        _messages(_synthesis_user(public_prompt, revision_answers), system=_SYNTH_SYSTEM),
        upstream=run.caps("revision", [index for index, _answer in revisions]),
    )
    _required(synthesis, "synthesis")
    verdict = await run.call(
        "verdict",
        0,
        tuple(
            ChatMessage.model_validate(message)
            for message in _verdict_messages(public_prompt, revision_answers)
        ),
        upstream=run.caps("revision", [index for index, _answer in revisions]),
    )
    return _required(verdict, "verdict")


_EXECUTORS: Mapping[ConditionId, Callable[[_Execution, str], Awaitable[str]]] = MappingProxyType(
    {
        "single_frontier": execute_single_frontier,
        "self_refine": execute_self_refine,
        "independent_synthesis": execute_independent_synthesis,
        "critique_only": execute_critique_only,
        "revision_only": execute_revision_only,
        "elite_full": execute_elite_full,
    }
)

LIVE_PROTOCOL_REGISTRY: Mapping[ConditionId, LiveProtocolSpec] = MappingProxyType(
    {
        condition_id: LiveProtocolSpec(
            condition_id=condition_id,
            protocol_version=f"{condition_id}_live_v1",
            prompt_version=LIVE_PROMPT_VERSIONS[condition_id],
            executor=_EXECUTORS[condition_id],
        )
        for condition_id in EVAL_CONDITION_IDS
    }
)


async def execute_live_condition(
    condition_id: ConditionId,
    *,
    task: PublicTask,
    roster: Sequence[ProviderModelSpec],
    cell_ceiling: int,
    client: SequentialStageClient,
) -> str:
    """Execute one frozen condition with sequential awaits and no Council fan-out."""

    if len(roster) < ELITE_MIN_RESPONDERS:
        raise ValueError("paid live rosters require at least three members")
    caps = allocate_stage_caps(
        condition_id,
        roster_size=len(roster),
        cell_ceiling=cell_ceiling,
    )
    run = _Execution(condition_id, roster, caps, client)
    return await LIVE_PROTOCOL_REGISTRY[condition_id].executor(run, _public_prompt(task))
