from __future__ import annotations

import pytest
from pydantic import ValidationError

from conclave.evals.models import (
    BootstrapConfig,
    ExclusionDeviationPolicy,
    FrozenStudyDesign,
    PriceSnapshot,
    ProviderModelSpec,
    PublicTask,
    RandomizationConfig,
    RosterSpec,
    RunRecord,
    StudyRun,
    TimeoutRetryPolicy,
)
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.scoring import (
    AtomicError,
    GraderJudgment,
    RubricDimensions,
    cohen_kappa,
    score_study,
)

DIGEST = "sha256:" + "a" * 64


def _paid_study():
    task = PublicTask(task_id="task-a", prompt="Choose.")
    design = FrozenStudyDesign(
        evidence_classification="paid_exploratory_pilot",
        base_commit="1" * 40,
        task_family_map={"task-a": "operational"},
        rosters=tuple(
            RosterSpec(
                roster_id=f"roster-{suffix}",
                members=(
                    ProviderModelSpec(
                        provider_id=f"provider-{suffix}",
                        model_id=f"model-{suffix}",
                        model_revision="v1",
                    ),
                ),
            )
            for suffix in ("a", "b")
        ),
        condition_prompt_versions=dict.fromkeys(CONDITION_IDS, "prompt-v1"),
        condition_protocol_versions=dict.fromkeys(CONDITION_IDS, "protocol-v1"),
        generation_settings_hash=DIGEST,
        evaluator_version="eval-v1",
        analysis_code_hash=DIGEST,
        rubric_hash=DIGEST,
        grader_instructions_hash=DIGEST,
        grader_keys_hash=DIGEST,
        exclusion_deviation_policy=ExclusionDeviationPolicy(),
        timeout_retry_policy=TimeoutRetryPolicy(timeout_seconds=60, retry_attempts=0),
        randomization=RandomizationConfig(master_seed=17),
        bootstrap=BootstrapConfig(seed=19, samples=100),
        price_snapshot=PriceSnapshot(
            snapshot_id="prices-v1",
            captured_at="2026-07-17T00:00:00Z",
            currency="USD",
            prices_hash=DIGEST,
        ),
        approved_spend_ceiling_usd=10,
    )
    manifest = build_study_manifest(
        study_id="paid",
        tasks=[task],
        replicates=1,
        seed=17,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 100),
        frozen_design=design,
    )
    records = tuple(
        RunRecord(
            planned_run_id=planned.planned_run_id,
            outcome="failed" if index == 0 else "success",
            output=None if index == 0 else "answer",
            error_category="executor_error" if index == 0 else None,
        )
        for index, planned in enumerate(manifest.planned_runs)
    )
    study_run = StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(records),
        outcome_counts={"failed": 1, "success": len(records) - 1},
        total_completion_tokens=0,
        total_latency_ms=0,
    )
    return manifest, study_run


def _judgment(run_id: str, grader: str, *, value: bool = True) -> GraderJudgment:
    return GraderJudgment(
        judgment_id=f"{run_id}-{grader}",
        planned_run_id=run_id,
        grader_id=grader,
        critical_error_free=value,
        atomic_errors=(),
        severe_error=False,
        rubric_dimensions=RubricDimensions(
            constraint_recall=2,
            conflict_minority_recognition=2,
            unsupported_claim_avoidance=2,
            recommendation_correctness=2,
            completeness_actionability=2,
            readiness_calibration=2,
        ),
        reviewer_seconds=12.5,
        confidence="high",
        abstained=False,
        rubric_version="rubric-v1",
        rubric_hash=DIGEST,
        grader_batch="batch-a",
        grader_order=1 if grader == "grader-a" else 2,
        condition_guess="unknown",
        provider_guess="unknown",
    )


@pytest.mark.parametrize(
    "dimension",
    (
        "constraint_recall",
        "conflict_minority_recognition",
        "unsupported_claim_avoidance",
        "recommendation_correctness",
        "completeness_actionability",
        "readiness_calibration",
    ),
)
def test_each_frozen_rubric_dimension_is_bounded_zero_to_two(dimension) -> None:
    names = (
        "constraint_recall",
        "conflict_minority_recognition",
        "unsupported_claim_avoidance",
        "recommendation_correctness",
        "completeness_actionability",
        "readiness_calibration",
    )
    values = dict.fromkeys(names, 1)

    assert getattr(RubricDimensions(**{**values, dimension: 0}), dimension) == 0
    assert getattr(RubricDimensions(**{**values, dimension: 2}), dimension) == 2
    with pytest.raises(ValidationError):
        RubricDimensions(**{**values, dimension: -1})
    with pytest.raises(ValidationError):
        RubricDimensions(**{**values, dimension: 3})


def test_atomic_errors_and_judgment_status_are_typed_and_consistent() -> None:
    error = AtomicError(
        rubric_item="factuality",
        category="unsupported_claim",
        severity="severe",
        detail="The recommendation contradicts packet p1.",
        evidence_packet_ids=("p1",),
    )
    judgment = _judgment("run_" + "a" * 24, "grader-a").model_copy(
        update={"atomic_errors": (error,), "severe_error": True, "critical_error_free": False}
    )

    assert judgment.atomic_errors[0].severity == "severe"
    with pytest.raises(ValidationError, match="severe_error"):
        GraderJudgment.model_validate({**judgment.model_dump(mode="json"), "severe_error": False})
    with pytest.raises(ValidationError, match="critical_error_free"):
        GraderJudgment.model_validate(
            {**judgment.model_dump(mode="json"), "critical_error_free": True}
        )
    with pytest.raises(ValidationError, match="abstained"):
        GraderJudgment.model_validate(
            {
                **_judgment("run_" + "b" * 24, "grader-a").model_dump(mode="json"),
                "abstained": True,
                "critical_error_free": True,
            }
        )


def test_paid_scoring_requires_two_complete_independent_judgments_per_success() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    complete = tuple(
        _judgment(record.planned_run_id, grader)
        for record in successful
        for grader in ("grader-a", "grader-b")
    )

    with pytest.raises(ValueError, match="exactly two"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=complete[:-1],
            bootstrap_seed=7,
            bootstrap_samples=20,
        )
    incomplete = complete[0].model_copy(update={"reviewer_seconds": None})
    with pytest.raises(ValueError, match="complete paid-study fields"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=(incomplete, *complete[1:]),
            bootstrap_seed=7,
            bootstrap_samples=20,
        )
    with pytest.raises(ValueError, match="non-success"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=(*complete, _judgment(study_run.records[0].planned_run_id, "grader-a")),
            bootstrap_seed=7,
            bootstrap_samples=20,
        )


def test_paid_reliability_reports_raw_agreement_prevalence_and_strata() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    judgments = []
    for index, record in enumerate(successful):
        judgments.append(_judgment(record.planned_run_id, "grader-a"))
        judgments.append(_judgment(record.planned_run_id, "grader-b", value=index != 0))

    report = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=tuple(judgments),
        bootstrap_seed=7,
        bootstrap_samples=20,
    )

    assert report.evidence_classification == "paid_exploratory_pilot"
    assert report.reliability.raw_agreement == pytest.approx(
        (len(successful) - 1) / len(successful)
    )
    assert 0 <= report.reliability.positive_prevalence <= 1
    assert {(item.stratum_type, item.stratum_value) for item in report.reliability.strata} == {
        ("family", "operational"),
        ("roster", "roster-a"),
        ("roster", "roster-b"),
    }
    assert report.resolved_outcomes[0].resolution == "automatic_non_success"


def test_constant_prevalence_kappa_is_undefined_not_perfect() -> None:
    assert cohen_kappa([True, True], [True, True]) is None
