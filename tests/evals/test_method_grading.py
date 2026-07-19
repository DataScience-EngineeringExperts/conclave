from __future__ import annotations

import pytest
from pydantic import ValidationError

from conclave.evals.blinding import BlindMap, BlindMapEntry, build_grader_queue, hash_blind_map
from conclave.evals.models import (
    AnalysisGateConfig,
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
        analysis_gates=AnalysisGateConfig(
            primary_baseline="self_refine",
            absolute_p95_latency_seconds=180,
            minimum_confirmatory_tasks=2,
        ),
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
            latency_ms=100.0 + index,
            cost_usd=0.01,
            cost_receipt_complete=True,
            deviation_codes=("retry_used",) if index == 1 else (),
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
        total_cost_usd=0.12,
        total_deviation_count=1,
    )
    return manifest, study_run


def _judgment(target_id: str, grader: str, *, value: bool = True) -> GraderJudgment:
    target = (
        {"planned_run_id": target_id}
        if target_id.startswith("run_")
        else {"opaque_output_id": target_id}
    )
    return GraderJudgment(
        judgment_id=f"{target_id}-{grader}",
        **target,
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
        readiness_correct=True,
    )


def _blind_targets(study_run: StudyRun) -> tuple[BlindMap, dict[str, str]]:
    _, blind_map = build_grader_queue(study_run.records, seed=31, forbidden_labels=())
    return blind_map, {entry.planned_run_id: entry.opaque_output_id for entry in blind_map.entries}


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
    blind_map, opaque_by_run = _blind_targets(study_run)
    complete = tuple(
        _judgment(opaque_by_run[record.planned_run_id], grader)
        for record in successful
        for grader in ("grader-a", "grader-b")
    )

    with pytest.raises(ValueError, match="exactly two"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=complete[:-1],
            blind_map=blind_map,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )
    incomplete = complete[0].model_copy(update={"reviewer_seconds": None})
    with pytest.raises(ValueError, match="complete paid-study fields"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=(incomplete, *complete[1:]),
            blind_map=blind_map,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )
    with pytest.raises(ValueError, match="missing from the blind map"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=(*complete, _judgment("output_" + "f" * 24, "grader-a")),
            blind_map=blind_map,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )


def test_paid_scoring_requires_a_hashed_blind_map() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    judgments = tuple(
        _judgment(record.planned_run_id, grader)
        for record in successful
        for grader in ("grader-a", "grader-b")
    )

    with pytest.raises(ValueError, match="hashed blind map"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=judgments,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )


def test_paid_scoring_requires_opaque_judgment_targets() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    judgments = tuple(
        _judgment(record.planned_run_id, grader)
        for record in successful
        for grader in ("grader-a", "grader-b")
    )
    _, blind_map = build_grader_queue(study_run.records, seed=31, forbidden_labels=())

    with pytest.raises(ValueError, match="opaque output IDs"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=judgments,
            blind_map=blind_map,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )


def test_paid_scoring_rejects_blind_maps_outside_the_successful_output_set() -> None:
    manifest, study_run = _paid_study()
    _, blind_map = build_grader_queue(study_run.records, seed=31, forbidden_labels=())
    opaque_by_run = {entry.planned_run_id: entry.opaque_output_id for entry in blind_map.entries}
    judgments = tuple(
        _judgment(opaque_by_run[record.planned_run_id], grader)
        for record in study_run.records
        if record.outcome == "success"
        for grader in ("grader-a", "grader-b")
    )
    entries = (
        *blind_map.entries,
        BlindMapEntry(
            opaque_output_id="output_" + "f" * 24,
            planned_run_id=study_run.records[0].planned_run_id,
        ),
    )
    expanded_map = BlindMap(entries=entries, blind_map_hash=hash_blind_map(entries))

    with pytest.raises(ValueError, match="successful output set"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=judgments,
            blind_map=expanded_map,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )


def test_paid_scoring_requires_complete_cost_and_latency_receipts() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    blind_map, opaque_by_run = _blind_targets(study_run)
    judgments = tuple(
        _judgment(opaque_by_run[record.planned_run_id], grader)
        for record in successful
        for grader in ("grader-a", "grader-b")
    )
    missing_latency = study_run.model_copy(
        update={
            "records": (
                study_run.records[0].model_copy(update={"latency_ms": None}),
                *study_run.records[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="latency receipt"):
        score_study(
            manifest=manifest,
            study_run=missing_latency,
            raw_judgments=judgments,
            blind_map=blind_map,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )
    missing_cost = study_run.model_copy(
        update={
            "records": (
                study_run.records[0].model_copy(update={"cost_receipt_complete": False}),
                *study_run.records[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="cost receipt"):
        score_study(
            manifest=manifest,
            study_run=missing_cost,
            raw_judgments=judgments,
            blind_map=blind_map,
            bootstrap_seed=19,
            bootstrap_samples=100,
        )


def test_paid_scoring_uses_frozen_bootstrap_and_reports_effort_interval_and_cost_sets() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    blind_map, opaque_by_run = _blind_targets(study_run)
    judgments = tuple(
        _judgment(opaque_by_run[record.planned_run_id], grader)
        for record in successful
        for grader in ("grader-a", "grader-b")
    )
    with pytest.raises(ValueError, match="frozen bootstrap"):
        score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=judgments,
            blind_map=blind_map,
            bootstrap_seed=7,
            bootstrap_samples=20,
        )
    report = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=judgments,
        blind_map=blind_map,
        bootstrap_seed=19,
        bootstrap_samples=100,
    )
    assert report.paid_analysis.primary_paired_difference.bootstrap_seed == 19
    assert report.paid_analysis.reviewer_effort_ratio_interval.upper >= 1
    assert report.paid_analysis.total_cost_usd == pytest.approx(study_run.total_cost_usd)
    assert report.paid_analysis.analysis_cost_usd <= report.paid_analysis.total_cost_usd


def test_paid_reliability_reports_raw_agreement_prevalence_and_strata() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    blind_map, opaque_by_run = _blind_targets(study_run)
    judgments = []
    for index, record in enumerate(successful):
        target_id = opaque_by_run[record.planned_run_id]
        judgments.append(_judgment(target_id, "grader-a"))
        judgments.append(_judgment(target_id, "grader-b", value=index != 0))

    report = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=tuple(judgments),
        blind_map=blind_map,
        bootstrap_seed=19,
        bootstrap_samples=100,
    )

    assert report.evidence_classification == "paid_exploratory_pilot"
    assert report.reliability.kappa_method == "cohen_fixed_pair"
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
    assert report.paid_analysis.primary_paired_difference.baseline_condition_id == "self_refine"
    assert report.paid_analysis.primary_paired_difference.task_count == 1
    assert report.paid_analysis.excluded_task_ids == ()


def test_constant_prevalence_kappa_is_undefined_not_perfect() -> None:
    assert cohen_kappa([True, True], [True, True]) is None


def test_rotating_grader_kappa_is_label_invariant_and_stratified() -> None:
    manifest, study_run = _paid_study()
    successful = [record for record in study_run.records if record.outcome == "success"]
    blind_map, opaque_by_run = _blind_targets(study_run)
    pairs = (("A", "B"), ("B", "C"), ("A", "C"), ("A", "B"))
    ratings = ((False, False), (False, True), (False, False), (False, True))

    def judgments(rename_b: bool):
        values = []
        for index, record in enumerate(successful):
            pair = pairs[index % len(pairs)]
            outcomes = ratings[index % len(ratings)]
            for grader, outcome in zip(pair, outcomes, strict=True):
                grader_id = "Z" if rename_b and grader == "B" else grader
                values.append(
                    _judgment(opaque_by_run[record.planned_run_id], grader_id, value=outcome)
                )
        return tuple(values)

    original = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=judgments(False),
        blind_map=blind_map,
        bootstrap_seed=19,
        bootstrap_samples=100,
    )
    renamed = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=judgments(True),
        blind_map=blind_map,
        bootstrap_seed=19,
        bootstrap_samples=100,
    )

    assert original.reliability.grader_pair is None
    assert original.reliability.kappa_method == "fleiss_rotating_pairs"
    assert original.reliability.cohen_kappa == pytest.approx(renamed.reliability.cohen_kappa)
    assert original.reliability.raw_agreement == renamed.reliability.raw_agreement
    assert all(item.kappa_method is not None for item in original.reliability.strata)
