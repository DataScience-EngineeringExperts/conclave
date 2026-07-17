"""Failure-inclusive scoring and reliability statistics for frozen eval studies."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from .blinding import BlindMap
from .models import (
    ConditionId,
    EvalModel,
    RunOutcome,
    StudyManifest,
    StudyRun,
)


class GraderJudgment(EvalModel):
    """One immutable raw grader judgment tied to one blinded or planned output."""

    judgment_id: str = Field(min_length=1)
    planned_run_id: str | None = Field(default=None, pattern=r"^run_[0-9a-f]{24}$")
    opaque_output_id: str | None = Field(default=None, pattern=r"^output_[0-9a-f]{24}$")
    grader_id: str = Field(min_length=1)
    critical_error_free: bool
    dimensions: dict[str, int] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_one_target(self) -> GraderJudgment:
        if (self.planned_run_id is None) == (self.opaque_output_id is None):
            raise ValueError("exactly one of planned_run_id or opaque_output_id is required")
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
        return 1.0 if observed == 1.0 else None
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


def _reliability(
    judgments_by_run: Mapping[str, Sequence[GraderJudgment]],
    adjudicated_runs: set[str],
) -> ReliabilityMetrics:
    graders = sorted({item.grader_id for values in judgments_by_run.values() for item in values})
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
    if len(graders) != 2:
        return ReliabilityMetrics(
            paired_judgments=0,
            disagreements=disagreements,
            adjudicated_disagreements=adjudicated_disagreements,
            adjudication_rate=rate,
        )
    first, second = graders
    pairs = []
    for run_id in sorted(judgments_by_run):
        by_grader = {item.grader_id: item.critical_error_free for item in judgments_by_run[run_id]}
        if first in by_grader and second in by_grader:
            pairs.append((by_grader[first], by_grader[second]))
    return ReliabilityMetrics(
        grader_pair=(first, second),
        cohen_kappa=cohen_kappa([left for left, _ in pairs], [right for _, right in pairs]),
        paired_judgments=len(pairs),
        disagreements=disagreements,
        adjudicated_disagreements=adjudicated_disagreements,
        adjudication_rate=rate,
    )


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
        if not set(adjudication.source_judgment_ids).issubset(expected_source_ids):
            raise ValueError("adjudication source judgments must target the same run")
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
    condition_metrics = []
    task_condition_values: dict[ConditionId, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for condition_id in dict.fromkeys(run.condition_id for run in manifest.planned_runs):
        planned = [run for run in manifest.planned_runs if run.condition_id == condition_id]
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
    return StudyScoreReport(
        study_id=manifest.study_id,
        total_planned_runs=len(manifest.planned_runs),
        run_outcome_distribution=dict(
            sorted(Counter(item.outcome for item in study_run.records).items())
        ),
        condition_metrics=tuple(condition_metrics),
        paired_differences=paired,
        reliability=_reliability(judgments_by_run, set(adjudication_by_run)),
        raw_judgments=tuple(raw_judgments),
        adjudications=tuple(adjudications),
        resolved_outcomes=tuple(resolved),
    )
