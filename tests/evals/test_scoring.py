from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conclave.evals.blinding import BlindMap, BlindMapEntry
from conclave.evals.models import PublicTask, RunRecord, StudyRun
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.scoring import (
    AdjudicationRecord,
    GraderJudgment,
    cohen_kappa,
    paired_bootstrap_difference,
    score_study,
    wilson_interval,
)


def _study(*, task_count: int = 2):
    tasks = [PublicTask(task_id=f"task-{index}", prompt="Choose.") for index in range(task_count)]
    manifest = build_study_manifest(
        study_id="synthetic-study",
        tasks=tasks,
        replicates=1,
        seed=17,
        output_token_budgets={condition_id: 100 for condition_id in CONDITION_IDS},
    )
    records = tuple(
        RunRecord(planned_run_id=planned.planned_run_id, outcome="success", output="answer")
        for planned in manifest.planned_runs
    )
    run = StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(records),
        outcome_counts={"success": len(records)},
        total_completion_tokens=0,
        total_latency_ms=0,
    )
    return manifest, run


def test_atomic_judgments_accept_exactly_one_opaque_or_planned_target() -> None:
    planned = GraderJudgment(
        judgment_id="judgment-planned",
        planned_run_id="run_" + "a" * 24,
        grader_id="grader-a",
        critical_error_free=True,
    )
    opaque = GraderJudgment(
        judgment_id="judgment-opaque",
        opaque_output_id="output_" + "b" * 24,
        grader_id="grader-b",
        critical_error_free=False,
    )

    assert planned.planned_run_id and planned.opaque_output_id is None
    assert opaque.opaque_output_id and opaque.planned_run_id is None
    with pytest.raises(ValidationError, match="exactly one"):
        GraderJudgment(
            judgment_id="bad",
            planned_run_id="run_" + "a" * 24,
            opaque_output_id="output_" + "b" * 24,
            grader_id="grader-a",
            critical_error_free=True,
        )
    with pytest.raises(ValidationError, match="exactly one"):
        GraderJudgment(
            judgment_id="bad",
            grader_id="grader-a",
            critical_error_free=True,
        )


def test_scoring_preserves_raw_records_and_keeps_adjudication_separate() -> None:
    manifest, run = _study(task_count=1)
    planned_id = manifest.planned_runs[0].planned_run_id
    opaque_id = "output_" + "c" * 24
    raw = (
        GraderJudgment(
            judgment_id="judgment-a",
            opaque_output_id=opaque_id,
            grader_id="grader-a",
            critical_error_free=True,
            notes="raw-a",
        ),
        GraderJudgment(
            judgment_id="judgment-b",
            opaque_output_id=opaque_id,
            grader_id="grader-b",
            critical_error_free=False,
            notes="raw-b",
        ),
    )
    adjudication = AdjudicationRecord(
        adjudication_id="adjudication-1",
        planned_run_id=planned_id,
        critical_error_free=True,
        source_judgment_ids=("judgment-a", "judgment-b"),
        adjudicator_id="adjudicator",
        rationale="Resolved from rubric evidence.",
    )
    blind_map = BlindMap(
        entries=(BlindMapEntry(opaque_output_id=opaque_id, planned_run_id=planned_id),)
    )

    report = score_study(
        manifest=manifest,
        study_run=run,
        raw_judgments=raw,
        adjudications=(adjudication,),
        blind_map=blind_map,
        bootstrap_seed=9,
        bootstrap_samples=100,
    )

    assert report.raw_judgments == raw
    assert report.adjudications == (adjudication,)
    assert report.raw_judgments[0].critical_error_free is True
    assert report.raw_judgments[1].critical_error_free is False
    assert report.resolved_outcomes[0].critical_error_free is True
    assert report.reliability.disagreements == 1
    assert report.reliability.adjudicated_disagreements == 1
    assert report.reliability.adjudication_rate == 1.0


def test_critical_error_free_rate_keeps_all_planned_runs_in_denominator() -> None:
    manifest, successful = _study(task_count=1)
    first_condition = manifest.planned_runs[0].condition_id
    failed_id = manifest.planned_runs[0].planned_run_id
    records = tuple(
        RunRecord(
            planned_run_id=record.planned_run_id,
            outcome="failed" if record.planned_run_id == failed_id else record.outcome,
            output=None if record.planned_run_id == failed_id else record.output,
        )
        for record in successful.records
    )
    run = StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(records),
        outcome_counts={"failed": 1, "success": len(records) - 1},
        total_completion_tokens=0,
        total_latency_ms=0,
    )
    judgments = tuple(
        GraderJudgment(
            judgment_id=f"judgment-{index}",
            planned_run_id=record.planned_run_id,
            grader_id="grader-a",
            critical_error_free=True,
        )
        for index, record in enumerate(records)
        if record.outcome == "success"
    )

    report = score_study(
        manifest=manifest,
        study_run=run,
        raw_judgments=judgments,
        bootstrap_seed=3,
        bootstrap_samples=100,
    )
    metric = next(item for item in report.condition_metrics if item.condition_id == first_condition)

    assert metric.planned_runs == 1
    assert metric.critical_error_free_count == 0
    assert metric.critical_error_free_rate == 0.0
    assert report.resolved_outcomes[0].critical_error_free is False
    assert report.resolved_outcomes[0].resolution == "automatic_non_success"


def test_wilson_interval_has_expected_95_percent_bounds() -> None:
    interval = wilson_interval(successes=5, trials=10)

    assert interval.confidence_level == 0.95
    assert interval.lower == pytest.approx(0.2366, abs=0.0001)
    assert interval.upper == pytest.approx(0.7634, abs=0.0001)
    assert wilson_interval(successes=0, trials=0).lower == 0.0
    assert wilson_interval(successes=0, trials=0).upper == 1.0


def test_seeded_paired_bootstrap_is_deterministic_and_task_paired() -> None:
    elite = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 1.0}
    baseline = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0}

    first = paired_bootstrap_difference(
        elite, baseline, seed=42, samples=1000, confidence_level=0.95
    )
    second = paired_bootstrap_difference(
        elite, baseline, seed=42, samples=1000, confidence_level=0.95
    )

    assert first == second
    assert first.task_count == 4
    assert first.estimate == 0.5
    assert first.lower <= first.estimate <= first.upper
    with pytest.raises(ValueError, match="same task IDs"):
        paired_bootstrap_difference({"a": 1.0}, {"b": 0.0}, seed=1, samples=10)


def test_cohen_kappa_and_adjudication_rate_are_reported() -> None:
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(
        0.5
    )
    assert cohen_kappa([True], [True, False]) is None
    assert cohen_kappa([], []) is None


def test_report_contract_serializes_raw_provenance_without_mutation() -> None:
    manifest, run = _study(task_count=1)
    judgments = tuple(
        GraderJudgment(
            judgment_id=f"judgment-{index}",
            planned_run_id=record.planned_run_id,
            grader_id="grader-a",
            critical_error_free=True,
            notes="synthetic fixture judgment",
        )
        for index, record in enumerate(run.records)
    )

    report = score_study(
        manifest=manifest,
        study_run=run,
        raw_judgments=judgments,
        bootstrap_seed=5,
        bootstrap_samples=100,
    )
    payload = json.loads(report.model_dump_json())

    assert payload["evidence_classification"] == "synthetic_exploratory"
    assert payload["decision_eligibility"] == "not_yet_eligible"
    assert payload["raw_judgments"][0]["notes"] == "synthetic fixture judgment"
    assert payload["decision_gates"] == {
        "go": "not_yet_eligible",
        "redesign": "not_yet_eligible",
        "kill": "not_yet_eligible",
    }
