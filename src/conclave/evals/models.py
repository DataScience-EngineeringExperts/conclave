"""Versioned, immutable data contracts for the experimental eval harness."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EVAL_SCHEMA_VERSION = "conclave_eval_v1"
SchemaVersion = Literal["conclave_eval_v1"]

ConditionId = Literal[
    "single_frontier",
    "self_refine",
    "independent_synthesis",
    "critique_only",
    "revision_only",
    "elite_full",
]

EVAL_CONDITION_IDS: tuple[ConditionId, ...] = (
    "single_frontier",
    "self_refine",
    "independent_synthesis",
    "critique_only",
    "revision_only",
    "elite_full",
)

Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
RunOutcome = Literal["success", "failed", "timed_out", "malformed", "abstained", "incomplete"]


class EvalModel(BaseModel):
    """Base contract that rejects drift and mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION


class PublicTask(EvalModel):
    """Material visible to every condition during study execution."""

    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference_packets: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)


class GraderKey(EvalModel):
    """Private grading material loaded only by the scoring process."""

    task_id: str = Field(min_length=1)
    required_facts: tuple[str, ...] = ()
    critical_errors: tuple[str, ...] = ()
    rubric: dict[str, str] = Field(default_factory=dict)


class ConditionSpec(EvalModel):
    """One frozen experimental comparison condition."""

    condition_id: ConditionId
    description: str = Field(min_length=1)


class PlannedRun(EvalModel):
    """An immutable cell declared before any model execution."""

    planned_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")
    study_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    condition_id: ConditionId
    replicate: int = Field(ge=1)
    max_output_tokens: int = Field(gt=0)


class ProtocolExecution(EvalModel):
    """Typed output returned by an injected offline protocol executor."""

    outcome: RunOutcome
    output: str | None = None
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    error_category: str | None = None


class RunRecord(EvalModel):
    """Failure-inclusive result for one predeclared cell."""

    planned_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")
    outcome: RunOutcome
    output: str | None = None
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    error_category: str | None = None


class StudyRun(EvalModel):
    """Complete failure-inclusive execution result for a frozen manifest."""

    study_id: str = Field(min_length=1)
    records: tuple[RunRecord, ...]
    total_planned_runs: int = Field(ge=1)
    outcome_counts: dict[RunOutcome, int]
    total_completion_tokens: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0)


class ScoreRecord(EvalModel):
    """One atomic grader judgment without execution-only task material."""

    planned_run_id: str = Field(pattern=r"^run_[A-Za-z0-9_-]+$")
    grader_id: str = Field(min_length=1)
    critical_error_free: bool
    dimensions: dict[str, int] = Field(default_factory=dict)
    notes: str | None = None


class StudyManifest(EvalModel):
    """Frozen preregistration of the complete execution matrix."""

    study_id: str = Field(min_length=1)
    seed: int
    replicates: int = Field(ge=1)
    task_ids: tuple[str, ...] = Field(min_length=1)
    public_tasks_hash: Sha256Digest
    planned_runs: tuple[PlannedRun, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> StudyManifest:
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")

        expected = {
            (task_id, condition_id, replicate)
            for task_id in self.task_ids
            for condition_id in EVAL_CONDITION_IDS
            for replicate in range(1, self.replicates + 1)
        }
        actual = {(run.task_id, run.condition_id, run.replicate) for run in self.planned_runs}
        if actual != expected or len(self.planned_runs) != len(expected):
            raise ValueError(
                "planned_runs must contain the complete task x condition x replicate matrix"
            )
        if any(run.study_id != self.study_id for run in self.planned_runs):
            raise ValueError("every planned run must belong to this study")
        if len({run.planned_run_id for run in self.planned_runs}) != len(self.planned_runs):
            raise ValueError("planned_run_id values must be unique")
        return self


class PublicTaskDataset(EvalModel):
    """On-disk public dataset envelope."""

    tasks: tuple[PublicTask, ...]


class GraderKeyDataset(EvalModel):
    """On-disk grader-only dataset envelope."""

    grader_keys: tuple[GraderKey, ...]
