"""Failure-inclusive scoring and reliability statistics for frozen eval studies."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from .blinding import BlindMap
from .models import (
    AnalysisGateConfig,
    ConditionId,
    EvalModel,
    RunOutcome,
    Sha256Digest,
    StudyManifest,
    StudyRun,
)
from .runner import validate_run_records


class AtomicError(EvalModel):
    """One rubric-linked error identified in a grader-visible output."""

    rubric_item: str = Field(min_length=1)
    category: str = Field(min_length=1)
    severity: Literal["minor", "major", "severe"]
    detail: str = Field(min_length=1)
    evidence_packet_ids: tuple[str, ...] = ()


class RubricDimensions(EvalModel):
    """Fixed 0–2 holistic rubric dimensions."""

    constraint_recall: int = Field(ge=0, le=2)
    conflict_minority_recognition: int = Field(ge=0, le=2)
    unsupported_claim_avoidance: int = Field(ge=0, le=2)
    recommendation_correctness: int = Field(ge=0, le=2)
    completeness_actionability: int = Field(ge=0, le=2)
    readiness_calibration: int = Field(ge=0, le=2)


class GraderJudgment(EvalModel):
    """One immutable raw grader judgment tied to one blinded or planned output."""

    judgment_id: str = Field(min_length=1)
    planned_run_id: str | None = Field(default=None, pattern=r"^run_[0-9a-f]{24}$")
    opaque_output_id: str | None = Field(default=None, pattern=r"^output_[0-9a-f]{24}$")
    grader_id: str = Field(min_length=1)
    critical_error_free: bool
    dimensions: dict[str, int] = Field(default_factory=dict)
    notes: str | None = None
    atomic_errors: tuple[AtomicError, ...] = ()
    severe_error: bool | None = None
    rubric_dimensions: RubricDimensions | None = None
    reviewer_seconds: float | None = Field(default=None, gt=0)
    confidence: Literal["low", "medium", "high"] | None = None
    abstained: bool | None = None
    rubric_version: str | None = None
    rubric_hash: Sha256Digest | None = None
    grader_batch: str | None = None
    grader_order: int | None = Field(default=None, ge=1)
    condition_guess: ConditionId | Literal["unknown"] | None = None
    provider_guess: str | None = None
    readiness_correct: bool | None = None

    @model_validator(mode="after")
    def validate_one_target(self) -> GraderJudgment:
        if (self.planned_run_id is None) == (self.opaque_output_id is None):
            raise ValueError("exactly one of planned_run_id or opaque_output_id is required")
        has_severe = any(error.severity == "severe" for error in self.atomic_errors)
        if self.severe_error is not None and self.severe_error != has_severe:
            raise ValueError("severe_error must match the atomic error severities")
        has_disqualifying_error = any(
            error.severity in {"major", "severe"} for error in self.atomic_errors
        )
        if self.critical_error_free and has_disqualifying_error:
            raise ValueError("critical_error_free must be false for major or severe errors")
        if self.abstained is True and self.critical_error_free:
            raise ValueError("abstained judgments cannot be critical_error_free")
        return self


class AdjudicationRecord(EvalModel):
    """A separate resolution record; source judgments remain unchanged."""

    adjudication_id: str = Field(min_length=1)
    planned_run_id: str | None = Field(default=None, pattern=r"^run_[0-9a-f]{24}$")
    opaque_output_id: str | None = Field(default=None, pattern=r"^output_[0-9a-f]{24}$")
    critical_error_free: bool
    source_judgment_ids: tuple[str, ...] = Field(min_length=1)
    adjudicator_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_one_target(self) -> AdjudicationRecord:
        if (self.planned_run_id is None) == (self.opaque_output_id is None):
            raise ValueError("exactly one of planned_run_id or opaque_output_id is required")
        if len(set(self.source_judgment_ids)) != len(self.source_judgment_ids):
            raise ValueError("source_judgment_ids must be unique")
        return self


class RateInterval(EvalModel):
    """A bounded confidence interval for a proportion."""

    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)


class PairedDifference(EvalModel):
    """Task-paired Elite-minus-baseline bootstrap result."""

    baseline_condition_id: ConditionId
    elite_condition_id: ConditionId = "elite_full"
    estimate: float = Field(ge=-1.0, le=1.0)
    lower: float = Field(ge=-1.0, le=1.0)
    upper: float = Field(ge=-1.0, le=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    task_count: int = Field(ge=1)
    bootstrap_seed: int
    bootstrap_samples: int = Field(ge=1)


class ResolvedOutcome(EvalModel):
    """Failure-inclusive binary outcome for one predeclared cell."""

    planned_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")
    critical_error_free: bool
    resolution: str = Field(min_length=1)


class ConditionMetric(EvalModel):
    """Primary endpoint and execution distribution for one condition."""

    condition_id: ConditionId
    planned_runs: int = Field(ge=1)
    critical_error_free_count: int = Field(ge=0)
    critical_error_free_rate: float = Field(ge=0.0, le=1.0)
    wilson_95_interval: RateInterval
    outcome_distribution: dict[RunOutcome, int]


class ReliabilityMetrics(EvalModel):
    """Inter-grader agreement and separate adjudication workload."""

    grader_pair: tuple[str, str] | None = None
    cohen_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    paired_judgments: int = Field(ge=0)
    disagreements: int = Field(ge=0)
    adjudicated_disagreements: int = Field(ge=0)
    adjudication_rate: float = Field(ge=0.0, le=1.0)
    raw_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_prevalence: float | None = Field(default=None, ge=0.0, le=1.0)
    strata: tuple[ReliabilityStratum, ...] = ()


class ReliabilityStratum(EvalModel):
    """Agreement statistics for one frozen family or roster stratum."""

    stratum_type: Literal["family", "roster"]
    stratum_value: str = Field(min_length=1)
    paired_judgments: int = Field(ge=0)
    raw_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_prevalence: float | None = Field(default=None, ge=0.0, le=1.0)
    cohen_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)


class StudyScoreReport(EvalModel):
    """Machine-readable, explicitly non-confirmatory evaluation report."""

    study_id: str = Field(min_length=1)
    evidence_classification: str = "synthetic_exploratory"
    decision_eligibility: str = "not_yet_eligible"
    decision_gates: dict[str, str] = Field(
        default_factory=lambda: {
            "go": "not_yet_eligible",
            "redesign": "not_yet_eligible",
            "kill": "not_yet_eligible",
        }
    )
    total_planned_runs: int = Field(ge=1)
    run_outcome_distribution: dict[RunOutcome, int]
    condition_metrics: tuple[ConditionMetric, ...]
    paired_differences: tuple[PairedDifference, ...]
    reliability: ReliabilityMetrics
    raw_judgments: tuple[GraderJudgment, ...]
    adjudications: tuple[AdjudicationRecord, ...]
    resolved_outcomes: tuple[ResolvedOutcome, ...]
    paid_analysis: PaidAnalysisSummary | None = None


class PaidAnalysisSummary(EvalModel):
    """Task-clustered operational and quality summaries for paid studies."""

    primary_paired_difference: PairedDifference
    severe_error_difference: PairedDifference
    readiness_error_difference: PairedDifference
    reviewer_effort_ratio: float = Field(gt=0)
    p95_latency_ratio: float = Field(gt=0)
    elite_p95_latency_seconds: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    deviation_count: int = Field(ge=0)
    excluded_task_ids: tuple[str, ...]
    family_directions: dict[str, float]
    roster_directions: dict[str, float]


class ConfirmatoryGateMetrics(EvalModel):
    """Frozen inputs consumed by the deterministic confirmatory gate."""

    primary_effect: float
    primary_lower_95: float
    severe_error_upper_95: float
    readiness_error_upper_95: float
    reviewer_effort_upper_ratio_95: float = Field(gt=0)
    p95_latency_ratio: float = Field(gt=0)
    elite_p95_latency_seconds: float = Field(ge=0)
    family_directions: dict[str, float] = Field(min_length=1)
    roster_directions: dict[str, float] = Field(min_length=1)


class ConfirmatoryGateResult(EvalModel):
    """GO only when every frozen DSE-708 threshold passes."""

    status: Literal["GO", "REDESIGN"]
    failed_gates: tuple[str, ...]


def evaluate_confirmatory_gates(
    *,
    evidence_classification: str,
    config: AnalysisGateConfig,
    metrics: ConfirmatoryGateMetrics,
    expected_frozen_design_hash: str,
    observed_frozen_design_hash: str,
    expected_preregistration_hash: str,
    observed_preregistration_hash: str,
) -> ConfirmatoryGateResult:
    """Refuse non-confirmatory/drifted artifacts, then evaluate every canonical gate."""

    if evidence_classification != "confirmatory":
        raise ValueError("confirmatory gate evaluation requires a confirmatory manifest")
    if expected_frozen_design_hash != observed_frozen_design_hash:
        raise ValueError("observed frozen design hash does not match the preregistered artifact")
    if expected_preregistration_hash != observed_preregistration_hash:
        raise ValueError("observed preregistration hash does not match the frozen artifact")
    failed = []
    checks = (
        ("minimum_effect", metrics.primary_effect >= config.minimum_effect),
        ("superiority_interval", metrics.primary_lower_95 > 0),
        (
            "severe_error_noninferiority",
            metrics.severe_error_upper_95 <= config.severe_error_noninferiority_margin,
        ),
        (
            "readiness_noninferiority",
            metrics.readiness_error_upper_95 <= config.readiness_noninferiority_margin,
        ),
        (
            "reviewer_effort",
            metrics.reviewer_effort_upper_ratio_95 <= config.reviewer_effort_max_ratio,
        ),
        ("latency_ratio", metrics.p95_latency_ratio <= config.p95_latency_max_ratio),
        (
            "latency_ceiling",
            metrics.elite_p95_latency_seconds <= config.absolute_p95_latency_seconds,
        ),
        ("family_direction", all(value > 0 for value in metrics.family_directions.values())),
        ("roster_direction", all(value > 0 for value in metrics.roster_directions.values())),
    )
    failed.extend(name for name, passed in checks if not passed)
    return ConfirmatoryGateResult(
        status="GO" if not failed else "REDESIGN", failed_gates=tuple(failed)
    )


def wilson_interval(*, successes: int, trials: int, confidence_level: float = 0.95) -> RateInterval:
    """Return a Wilson score interval using the fixed 95% normal quantile."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if confidence_level != 0.95:
        raise ValueError("only the predeclared 95% interval is supported")
    if trials == 0:
        return RateInterval(lower=0.0, upper=1.0, confidence_level=confidence_level)
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return RateInterval(
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence_level=confidence_level,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def paired_bootstrap_difference(
    elite_by_task: Mapping[str, float],
    baseline_by_task: Mapping[str, float],
    *,
    seed: int,
    samples: int,
    confidence_level: float = 0.95,
    baseline_condition_id: ConditionId = "single_frontier",
) -> PairedDifference:
    """Bootstrap paired task differences with a deterministic local RNG."""

    if set(elite_by_task) != set(baseline_by_task) or not elite_by_task:
        raise ValueError("elite and baseline must contain the same task IDs")
    if samples < 1:
        raise ValueError("samples must be positive")
    if confidence_level != 0.95:
        raise ValueError("only the predeclared 95% interval is supported")
    differences = [
        elite_by_task[task_id] - baseline_by_task[task_id] for task_id in sorted(elite_by_task)
    ]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrapped = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(samples)
    )
    return PairedDifference(
        baseline_condition_id=baseline_condition_id,
        estimate=estimate,
        lower=_quantile(bootstrapped, 0.025),
        upper=_quantile(bootstrapped, 0.975),
        confidence_level=confidence_level,
        task_count=len(differences),
        bootstrap_seed=seed,
        bootstrap_samples=samples,
    )


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> float | None:
    """Return Cohen's kappa for aligned binary ratings, or ``None`` if undefined."""

    if not first or len(first) != len(second):
        return None
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
    first_true = sum(first) / len(first)
    second_true = sum(second) / len(second)
    expected = first_true * second_true + (1 - first_true) * (1 - second_true)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


def _target_map(blind_map: BlindMap | None) -> dict[str, str]:
    if blind_map is None:
        return {}
    mapping = {entry.opaque_output_id: entry.planned_run_id for entry in blind_map.entries}
    if len(mapping) != len(blind_map.entries):
        raise ValueError("blind map contains duplicate opaque_output_id values")
    return mapping


def _planned_target(record: GraderJudgment | AdjudicationRecord, mapping: Mapping[str, str]) -> str:
    if record.planned_run_id is not None:
        return record.planned_run_id
    assert record.opaque_output_id is not None
    try:
        return mapping[record.opaque_output_id]
    except KeyError as exc:
        raise ValueError("opaque target is missing from the blind map") from exc


def _paired_values(
    judgments_by_run: Mapping[str, Sequence[GraderJudgment]], run_ids: Sequence[str]
) -> tuple[tuple[str, str] | None, list[tuple[bool, bool]]]:
    graders = sorted({item.grader_id for values in judgments_by_run.values() for item in values})
    if len(graders) != 2:
        return None, []
    first, second = graders
    pairs = []
    for run_id in sorted(run_ids):
        by_grader = {item.grader_id: item.critical_error_free for item in judgments_by_run[run_id]}
        if first in by_grader and second in by_grader:
            pairs.append((by_grader[first], by_grader[second]))
    return (first, second), pairs


def _agreement_statistics(
    pairs: Sequence[tuple[bool, bool]],
) -> tuple[float | None, float | None, float | None]:
    if not pairs:
        return None, None, None
    first = [left for left, _ in pairs]
    second = [right for _, right in pairs]
    raw_agreement = sum(left == right for left, right in pairs) / len(pairs)
    prevalence = (sum(first) + sum(second)) / (2 * len(pairs))
    return raw_agreement, prevalence, cohen_kappa(first, second)


def _reliability(
    judgments_by_run: Mapping[str, Sequence[GraderJudgment]],
    adjudicated_runs: set[str],
    manifest: StudyManifest,
) -> ReliabilityMetrics:
    disagreements = sum(
        len({item.critical_error_free for item in values}) > 1
        for values in judgments_by_run.values()
    )
    adjudicated_disagreements = sum(
        run_id in adjudicated_runs
        for run_id, values in judgments_by_run.items()
        if len({item.critical_error_free for item in values}) > 1
    )
    rate = adjudicated_disagreements / disagreements if disagreements else 0.0
    grader_pair, pairs = _paired_values(judgments_by_run, tuple(judgments_by_run))
    raw_agreement, prevalence, kappa = _agreement_statistics(pairs)
    strata: list[ReliabilityStratum] = []
    if manifest.frozen_design is not None:
        for family in sorted(set(manifest.frozen_design.task_family_map.values())):
            run_ids = [
                run.planned_run_id
                for run in manifest.planned_runs
                if manifest.frozen_design.task_family_map[run.task_id] == family
            ]
            _, stratum_pairs = _paired_values(judgments_by_run, run_ids)
            agreement, stratum_prevalence, stratum_kappa = _agreement_statistics(stratum_pairs)
            strata.append(
                ReliabilityStratum(
                    stratum_type="family",
                    stratum_value=family,
                    paired_judgments=len(stratum_pairs),
                    raw_agreement=agreement,
                    positive_prevalence=stratum_prevalence,
                    cohen_kappa=stratum_kappa,
                )
            )
        for roster_id in sorted(roster.roster_id for roster in manifest.frozen_design.rosters):
            run_ids = [
                run.planned_run_id for run in manifest.planned_runs if run.roster_id == roster_id
            ]
            _, stratum_pairs = _paired_values(judgments_by_run, run_ids)
            agreement, stratum_prevalence, stratum_kappa = _agreement_statistics(stratum_pairs)
            strata.append(
                ReliabilityStratum(
                    stratum_type="roster",
                    stratum_value=roster_id,
                    paired_judgments=len(stratum_pairs),
                    raw_agreement=agreement,
                    positive_prevalence=stratum_prevalence,
                    cohen_kappa=stratum_kappa,
                )
            )
    return ReliabilityMetrics(
        grader_pair=grader_pair,
        cohen_kappa=kappa,
        paired_judgments=len(pairs),
        disagreements=disagreements,
        adjudicated_disagreements=adjudicated_disagreements,
        adjudication_rate=rate,
        raw_agreement=raw_agreement,
        positive_prevalence=prevalence,
        strata=tuple(strata),
    )


def _validate_paid_grading(
    *,
    manifest: StudyManifest,
    study_run: StudyRun,
    judgments_by_run: Mapping[str, Sequence[GraderJudgment]],
) -> None:
    """Fail closed on incomplete or non-independent paid-study grading."""

    if manifest.evidence_classification == "synthetic_exploratory":
        return
    run_by_id = {record.planned_run_id: record for record in study_run.records}
    required_fields = (
        "severe_error",
        "rubric_dimensions",
        "reviewer_seconds",
        "confidence",
        "abstained",
        "rubric_version",
        "rubric_hash",
        "grader_batch",
        "grader_order",
        "condition_guess",
        "provider_guess",
        "readiness_correct",
    )
    for run_id, judgments in judgments_by_run.items():
        if run_by_id[run_id].outcome != "success":
            raise ValueError("non-success cells must remain outside paid human grading")
        if any(
            getattr(judgment, field) is None for judgment in judgments for field in required_fields
        ):
            raise ValueError("paid judgments require complete paid-study fields")
        if manifest.frozen_design is not None and any(
            judgment.rubric_hash != manifest.frozen_design.rubric_hash for judgment in judgments
        ):
            raise ValueError("paid judgment rubric hash must match the frozen design")
    for record in study_run.records:
        if record.outcome != "success":
            continue
        judgments = judgments_by_run[record.planned_run_id]
        if len(judgments) != 2 or len({item.grader_id for item in judgments}) != 2:
            raise ValueError(
                "every successful paid cell requires exactly two independent judgments"
            )


def analysis_planned_runs(manifest: StudyManifest):
    """Apply only symmetric, predeclared task exclusions to the analysis matrix."""

    excluded = (
        set(manifest.frozen_design.exclusion_deviation_policy.excluded_task_ids)
        if manifest.frozen_design is not None
        else set()
    )
    return tuple(run for run in manifest.planned_runs if run.task_id not in excluded)


def score_study(
    *,
    manifest: StudyManifest,
    study_run: StudyRun,
    raw_judgments: Sequence[GraderJudgment],
    adjudications: Sequence[AdjudicationRecord] = (),
    blind_map: BlindMap | None = None,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> StudyScoreReport:
    """Score one frozen study without dropping failures or modifying raw judgments."""

    if study_run.study_id != manifest.study_id:
        raise ValueError("study run and manifest study_id values must match")
    validate_run_records(manifest, study_run.records)
    planned_by_id = {run.planned_run_id: run for run in manifest.planned_runs}
    run_by_id = {record.planned_run_id: record for record in study_run.records}
    if set(run_by_id) != set(planned_by_id) or len(run_by_id) != len(study_run.records):
        raise ValueError("study run must cover every planned_run_id exactly once")
    if len({item.judgment_id for item in raw_judgments}) != len(raw_judgments):
        raise ValueError("judgment_id values must be unique")
    if len({item.adjudication_id for item in adjudications}) != len(adjudications):
        raise ValueError("adjudication_id values must be unique")

    mapping = _target_map(blind_map)
    judgments_by_run: dict[str, list[GraderJudgment]] = defaultdict(list)
    judgment_ids = {item.judgment_id for item in raw_judgments}
    seen_grader_targets: set[tuple[str, str]] = set()
    for judgment in raw_judgments:
        run_id = _planned_target(judgment, mapping)
        if run_id not in planned_by_id:
            raise ValueError("grader judgment references an unplanned run")
        grader_target = (run_id, judgment.grader_id)
        if grader_target in seen_grader_targets:
            raise ValueError("a grader may provide only one judgment per planned run")
        seen_grader_targets.add(grader_target)
        judgments_by_run[run_id].append(judgment)

    _validate_paid_grading(
        manifest=manifest, study_run=study_run, judgments_by_run=judgments_by_run
    )

    adjudication_by_run: dict[str, AdjudicationRecord] = {}
    for adjudication in adjudications:
        run_id = _planned_target(adjudication, mapping)
        if run_id not in planned_by_id:
            raise ValueError("adjudication references an unplanned run")
        if run_id in adjudication_by_run:
            raise ValueError("only one adjudication is allowed per planned run")
        if not set(adjudication.source_judgment_ids).issubset(judgment_ids):
            raise ValueError("adjudication references an unknown source judgment")
        expected_source_ids = {item.judgment_id for item in judgments_by_run[run_id]}
        source_ids = set(adjudication.source_judgment_ids)
        if source_ids != expected_source_ids:
            raise ValueError("adjudication must cite exactly all same-run source judgments")
        values = {item.critical_error_free for item in judgments_by_run[run_id]}
        if len(expected_source_ids) < 2 or len(values) < 2:
            raise ValueError("adjudication requires a genuine multi-grader disagreement")
        adjudication_by_run[run_id] = adjudication

    resolved: list[ResolvedOutcome] = []
    for planned in manifest.planned_runs:
        run_record = run_by_id[planned.planned_run_id]
        values = [item.critical_error_free for item in judgments_by_run[planned.planned_run_id]]
        if run_record.outcome != "success":
            value, resolution = False, "automatic_non_success"
        elif planned.planned_run_id in adjudication_by_run:
            value = adjudication_by_run[planned.planned_run_id].critical_error_free
            resolution = "adjudicated"
        elif values and len(set(values)) == 1:
            value, resolution = values[0], "unanimous_raw"
        elif values:
            value, resolution = False, "unresolved_disagreement"
        else:
            value, resolution = False, "ungraded_success"
        resolved.append(
            ResolvedOutcome(
                planned_run_id=planned.planned_run_id,
                critical_error_free=value,
                resolution=resolution,
            )
        )

    resolved_by_id = {item.planned_run_id: item for item in resolved}
    excluded_task_ids = (
        manifest.frozen_design.exclusion_deviation_policy.excluded_task_ids
        if manifest.frozen_design is not None
        else ()
    )
    analysis_runs = analysis_planned_runs(manifest)
    if not analysis_runs:
        raise ValueError("task exclusions cannot remove every task from analysis")
    condition_metrics = []
    task_condition_values: dict[ConditionId, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for condition_id in dict.fromkeys(run.condition_id for run in analysis_runs):
        planned = [run for run in analysis_runs if run.condition_id == condition_id]
        successes = sum(resolved_by_id[run.planned_run_id].critical_error_free for run in planned)
        distribution = dict(
            sorted(Counter(run_by_id[run.planned_run_id].outcome for run in planned).items())
        )
        condition_metrics.append(
            ConditionMetric(
                condition_id=condition_id,
                planned_runs=len(planned),
                critical_error_free_count=successes,
                critical_error_free_rate=successes / len(planned),
                wilson_95_interval=wilson_interval(successes=successes, trials=len(planned)),
                outcome_distribution=distribution,
            )
        )
        for run in planned:
            task_condition_values[condition_id][run.task_id].append(
                resolved_by_id[run.planned_run_id].critical_error_free
            )

    means = {
        condition_id: {
            task_id: sum(values) / len(values) for task_id, values in task_values.items()
        }
        for condition_id, task_values in task_condition_values.items()
    }
    paired = tuple(
        paired_bootstrap_difference(
            means["elite_full"],
            means[condition_id],
            seed=bootstrap_seed,
            samples=bootstrap_samples,
            baseline_condition_id=condition_id,
        )
        for condition_id in means
        if condition_id != "elite_full"
    )
    paid_analysis = None
    if manifest.frozen_design is not None:
        design = manifest.frozen_design
        baseline = design.analysis_gates.primary_baseline
        primary = next(item for item in paired if item.baseline_condition_id == baseline)

        def task_means(cell_values: Mapping[str, float]) -> dict[ConditionId, dict[str, float]]:
            grouped: dict[ConditionId, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for run in analysis_runs:
                grouped[run.condition_id][run.task_id].append(cell_values[run.planned_run_id])
            return {
                condition: {task_id: sum(values) / len(values) for task_id, values in tasks.items()}
                for condition, tasks in grouped.items()
            }

        severe_cells = {
            run.planned_run_id: float(
                any(item.severe_error for item in judgments_by_run[run.planned_run_id])
            )
            for run in analysis_runs
        }
        readiness_error_cells = {
            run.planned_run_id: float(
                run_by_id[run.planned_run_id].outcome != "success"
                or not all(
                    item.readiness_correct is True for item in judgments_by_run[run.planned_run_id]
                )
            )
            for run in analysis_runs
        }
        effort_cells = {
            run.planned_run_id: (
                sum(item.reviewer_seconds or 0 for item in judgments_by_run[run.planned_run_id])
                / len(judgments_by_run[run.planned_run_id])
                if judgments_by_run[run.planned_run_id]
                else 0.0
            )
            for run in analysis_runs
        }
        severe_means = task_means(severe_cells)
        readiness_means = task_means(readiness_error_cells)
        effort_means = task_means(effort_cells)
        frozen_seed = design.bootstrap.seed
        frozen_samples = design.bootstrap.samples
        severe_difference = paired_bootstrap_difference(
            severe_means["elite_full"],
            severe_means[baseline],
            seed=frozen_seed,
            samples=frozen_samples,
            baseline_condition_id=baseline,
        )
        readiness_difference = paired_bootstrap_difference(
            readiness_means["elite_full"],
            readiness_means[baseline],
            seed=frozen_seed,
            samples=frozen_samples,
            baseline_condition_id=baseline,
        )
        baseline_effort = sum(effort_means[baseline].values()) / len(effort_means[baseline])
        elite_effort = sum(effort_means["elite_full"].values()) / len(effort_means["elite_full"])
        if baseline_effort <= 0:
            raise ValueError("paid reviewer-effort baseline must be positive")
        latencies = {
            condition: sorted(
                (run_by_id[run.planned_run_id].latency_ms or 0.0) / 1000
                for run in analysis_runs
                if run.condition_id == condition
            )
            for condition in (baseline, "elite_full")
        }
        baseline_p95 = _quantile(latencies[baseline], 0.95)
        elite_p95 = _quantile(latencies["elite_full"], 0.95)
        if baseline_p95 <= 0:
            raise ValueError("paid latency baseline must be positive")

        def directional_effects(attribute: str) -> dict[str, float]:
            values = {}
            if attribute == "family":
                groups = design.task_family_map
                names = sorted(set(groups.values()))

                def selected(run, name):
                    return groups[run.task_id] == name

            else:
                names = sorted(roster.roster_id for roster in design.rosters)

                def selected(run, name):
                    return run.roster_id == name

            for name in names:
                elite_values = [
                    resolved_by_id[run.planned_run_id].critical_error_free
                    for run in analysis_runs
                    if run.condition_id == "elite_full" and selected(run, name)
                ]
                baseline_values = [
                    resolved_by_id[run.planned_run_id].critical_error_free
                    for run in analysis_runs
                    if run.condition_id == baseline and selected(run, name)
                ]
                values[name] = sum(elite_values) / len(elite_values) - sum(baseline_values) / len(
                    baseline_values
                )
            return values

        paid_analysis = PaidAnalysisSummary(
            primary_paired_difference=primary,
            severe_error_difference=severe_difference,
            readiness_error_difference=readiness_difference,
            reviewer_effort_ratio=elite_effort / baseline_effort,
            p95_latency_ratio=elite_p95 / baseline_p95,
            elite_p95_latency_seconds=elite_p95,
            total_cost_usd=sum(run_by_id[run.planned_run_id].cost_usd for run in analysis_runs),
            deviation_count=sum(
                len(run_by_id[run.planned_run_id].deviation_codes) for run in analysis_runs
            ),
            excluded_task_ids=tuple(sorted(excluded_task_ids)),
            family_directions=directional_effects("family"),
            roster_directions=directional_effects("roster"),
        )
    return StudyScoreReport(
        study_id=manifest.study_id,
        evidence_classification=manifest.evidence_classification,
        total_planned_runs=len(manifest.planned_runs),
        run_outcome_distribution=dict(
            sorted(Counter(item.outcome for item in study_run.records).items())
        ),
        condition_metrics=tuple(condition_metrics),
        paired_differences=paired,
        reliability=_reliability(judgments_by_run, set(adjudication_by_run), manifest),
        raw_judgments=tuple(raw_judgments),
        adjudications=tuple(adjudications),
        resolved_outcomes=tuple(resolved),
        paid_analysis=paid_analysis,
    )
