"""Versioned, immutable data contracts for the experimental eval harness."""

from __future__ import annotations

import hashlib
import json
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
EvidenceClassification = Literal["synthetic_exploratory", "paid_exploratory_pilot", "confirmatory"]


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


class ProviderModelSpec(EvalModel):
    """One provider/model revision frozen into an experimental roster."""

    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)


class RosterSpec(EvalModel):
    """A named, immutable provider/model roster."""

    roster_id: str = Field(min_length=1)
    members: tuple[ProviderModelSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_members(self) -> RosterSpec:
        identities = [(item.provider_id, item.model_id) for item in self.members]
        if len(set(identities)) != len(identities):
            raise ValueError("roster provider/model identities must be unique")
        return self


class ExclusionDeviationPolicy(EvalModel):
    """Predeclared exclusions while keeping execution deviations in analysis."""

    predefined_task_exclusions: tuple[str, ...] = ()
    output_level_exclusions_allowed: Literal[False] = False
    deviations_remain_in_denominator: Literal[True] = True


class TimeoutRetryPolicy(EvalModel):
    """Frozen operational failure policy."""

    timeout_seconds: float = Field(gt=0)
    retry_attempts: int = Field(ge=0)
    exhausted_runs_remain_in_denominator: Literal[True] = True


class RandomizationConfig(EvalModel):
    """Frozen blocked-randomization algorithm and seed."""

    master_seed: int
    method: Literal["sha256_task_roster_block_v1"] = "sha256_task_roster_block_v1"


class BootstrapConfig(EvalModel):
    """Frozen uncertainty-estimation settings."""

    seed: int
    samples: int = Field(ge=1)
    confidence_level: Literal[0.95] = 0.95
    unit: Literal["task"] = "task"


class PriceSnapshot(EvalModel):
    """Immutable identifier and digest for the price table used by a study."""

    snapshot_id: str = Field(min_length=1)
    captured_at: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    prices_hash: Sha256Digest


class FrozenStudyDesign(EvalModel):
    """Complete experimental provenance frozen before any study execution."""

    evidence_classification: EvidenceClassification
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    task_family_map: dict[str, str] = Field(min_length=1)
    rosters: tuple[RosterSpec, ...] = Field(min_length=2)
    condition_prompt_versions: dict[ConditionId, str]
    condition_protocol_versions: dict[ConditionId, str]
    generation_settings_hash: Sha256Digest
    evaluator_version: str = Field(min_length=1)
    analysis_code_hash: Sha256Digest
    rubric_hash: Sha256Digest
    grader_instructions_hash: Sha256Digest
    grader_keys_hash: Sha256Digest
    exclusion_deviation_policy: ExclusionDeviationPolicy
    timeout_retry_policy: TimeoutRetryPolicy
    randomization: RandomizationConfig
    bootstrap: BootstrapConfig
    price_snapshot: PriceSnapshot
    approved_spend_ceiling_usd: float = Field(ge=0)
    preregistration_id: str | None = None
    preregistration_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_freeze(self) -> FrozenStudyDesign:
        roster_ids = [roster.roster_id for roster in self.rosters]
        if len(set(roster_ids)) != len(roster_ids):
            raise ValueError("frozen roster_id values must be unique")
        expected_conditions = set(EVAL_CONDITION_IDS)
        if set(self.condition_prompt_versions) != expected_conditions:
            raise ValueError("condition_prompt_versions must cover exactly all conditions")
        if set(self.condition_protocol_versions) != expected_conditions:
            raise ValueError("condition_protocol_versions must cover exactly all conditions")
        if any(not value for value in self.condition_prompt_versions.values()):
            raise ValueError("condition prompt versions must be nonempty")
        if any(not value for value in self.condition_protocol_versions.values()):
            raise ValueError("condition protocol versions must be nonempty")
        if any(not value for value in self.task_family_map.values()):
            raise ValueError("task family values must be nonempty")
        if self.evidence_classification in {"paid_exploratory_pilot", "confirmatory"}:
            if self.approved_spend_ceiling_usd <= 0:
                raise ValueError("paid studies require a positive approved spend ceiling")
        if self.evidence_classification == "confirmatory" and (
            not self.preregistration_id or self.preregistration_hash is None
        ):
            raise ValueError("confirmatory studies require preregistration ID and hash")
        return self


def hash_frozen_study_design(design: FrozenStudyDesign) -> str:
    """Return a canonical digest that detects any frozen-design drift."""

    canonical = json.dumps(
        design.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def derive_planned_run_id(
    *,
    study_id: str,
    task_id: str,
    condition_id: ConditionId,
    replicate: int,
    max_output_tokens: int,
    roster_id: str = "legacy_default",
    frozen_design_hash: str | None = None,
) -> str:
    """Derive the canonical cell identity, preserving the legacy payload exactly."""

    fields: dict[str, str | int] = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "study_id": study_id,
        "task_id": task_id,
        "condition_id": condition_id,
        "replicate": replicate,
        "max_output_tokens": max_output_tokens,
    }
    if roster_id != "legacy_default":
        fields["roster_id"] = roster_id
    if frozen_design_hash is not None:
        fields["frozen_design_hash"] = frozen_design_hash
    identity = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"run_{hashlib.sha256(identity).hexdigest()[:24]}"


class PlannedRun(EvalModel):
    """An immutable cell declared before any model execution."""

    planned_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")
    study_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    roster_id: str = Field(default="legacy_default", min_length=1)
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
    evidence_classification: EvidenceClassification = "synthetic_exploratory"
    promotable: Literal[False] = False
    seed: int
    replicates: int = Field(ge=1)
    task_ids: tuple[str, ...] = Field(min_length=1)
    public_tasks_hash: Sha256Digest
    frozen_design: FrozenStudyDesign | None = None
    frozen_design_hash: Sha256Digest | None = None
    planned_runs: tuple[PlannedRun, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> StudyManifest:
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")

        if self.frozen_design is None:
            if self.evidence_classification != "synthetic_exploratory":
                raise ValueError("a legacy manifest classification must be synthetic exploratory")
            if self.frozen_design_hash is not None:
                raise ValueError("a design hash requires a frozen design")
            roster_ids = ("legacy_default",)
        else:
            if self.evidence_classification != self.frozen_design.evidence_classification:
                raise ValueError("manifest and frozen design classification must match")
            if self.frozen_design_hash != hash_frozen_study_design(self.frozen_design):
                raise ValueError("frozen design hash does not match its contents")
            if set(self.frozen_design.task_family_map) != set(self.task_ids):
                raise ValueError("task_family_map must exactly cover manifest task_ids")
            if self.seed != self.frozen_design.randomization.master_seed:
                raise ValueError("manifest seed must match the frozen randomization master seed")
            roster_ids = tuple(roster.roster_id for roster in self.frozen_design.rosters)

        expected = {
            (task_id, roster_id, condition_id, replicate)
            for task_id in self.task_ids
            for roster_id in roster_ids
            for condition_id in EVAL_CONDITION_IDS
            for replicate in range(1, self.replicates + 1)
        }
        actual = {
            (run.task_id, run.roster_id, run.condition_id, run.replicate)
            for run in self.planned_runs
        }
        if actual != expected or len(self.planned_runs) != len(expected):
            raise ValueError(
                "planned_runs must contain the complete task x roster x condition x replicate matrix"
            )
        if any(run.study_id != self.study_id for run in self.planned_runs):
            raise ValueError("every planned run must belong to this study")
        if len({run.planned_run_id for run in self.planned_runs}) != len(self.planned_runs):
            raise ValueError("planned_run_id values must be unique")
        for run in self.planned_runs:
            expected_id = derive_planned_run_id(
                study_id=run.study_id,
                task_id=run.task_id,
                roster_id=run.roster_id,
                condition_id=run.condition_id,
                replicate=run.replicate,
                max_output_tokens=run.max_output_tokens,
                frozen_design_hash=self.frozen_design_hash,
            )
            if run.planned_run_id != expected_id:
                raise ValueError(f"planned_run_id does not match frozen cell: {run.planned_run_id}")
        return self


class PublicTaskDataset(EvalModel):
    """On-disk public dataset envelope."""

    tasks: tuple[PublicTask, ...]


class GraderKeyDataset(EvalModel):
    """On-disk grader-only dataset envelope."""

    grader_keys: tuple[GraderKey, ...]
