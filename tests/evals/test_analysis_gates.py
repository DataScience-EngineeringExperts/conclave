from __future__ import annotations

import pytest
from pydantic import ValidationError

from conclave.evals.models import AnalysisGateConfig, RunRecord, StudyRun
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.scoring import (
    AtomicError,
    analysis_planned_runs,
    evaluate_confirmatory_gates,
    hash_confirmatory_evidence,
    paired_bootstrap_ratio,
    recompute_simultaneous_upper_bounds,
    score_study,
)
from tests.evals.test_method_grading import _blind_targets, _judgment
from tests.evals.test_study_design import _design, _tasks


def _config(**updates):
    values = {
        "primary_baseline": "self_refine",
        "absolute_p95_latency_seconds": 180,
        "minimum_confirmatory_tasks": 2,
    }
    values.update(updates)
    return AnalysisGateConfig(**values)


def test_analysis_gate_config_freezes_canonical_dse_708_thresholds() -> None:
    config = _config()

    assert config.alpha == 0.05
    assert config.power == 0.80
    assert config.minimum_effect == 0.10
    assert config.severe_error_noninferiority_margin == 0.02
    assert config.readiness_noninferiority_margin == 0.05
    assert config.reviewer_effort_max_ratio == 1.20
    assert config.reviewer_effort_statistic == "ratio_of_task_medians"
    assert config.latency_baseline == "single_frontier"
    assert config.p95_latency_max_ratio == 3.0
    assert config.minimum_double_grading_rate == 0.95
    assert config.minimum_raw_agreement == 0.80
    assert config.minimum_overall_kappa == 0.60
    assert config.minimum_family_kappa == 0.50
    assert config.maximum_adjudication_rate == 0.20
    assert config.minimum_confirmatory_tasks == 2
    assert config.multiplicity_rule == "bonferroni_simultaneous_upper_bounds"
    assert config.unit == "task"
    with pytest.raises(ValidationError, match="primary_baseline"):
        _config(primary_baseline="elite_full")


def _confirmatory_report(*, minimum_tasks: int = 2):
    design = _design(
        evidence_classification="confirmatory",
        preregistration_id="osf:confirmatory-v1",
        preregistration_hash="sha256:" + "a" * 64,
    )
    design = design.model_copy(
        update={
            "analysis_gates": design.analysis_gates.model_copy(
                update={"minimum_confirmatory_tasks": minimum_tasks}
            )
        }
    )
    manifest = build_study_manifest(
        study_id="confirmatory",
        tasks=_tasks(),
        replicates=1,
        seed=20260717,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 100),
        frozen_design=design,
    )
    study_run = _confirmatory_report_run(manifest)
    blind_map, opaque_by_run = _blind_targets(study_run)
    records = study_run.records
    judgments = tuple(
        _judgment(
            opaque_by_run[record.planned_run_id],
            grader,
            value=planned.condition_id != "self_refine",
        )
        for planned, record in zip(manifest.planned_runs, records, strict=True)
        for grader in ("grader-a", "grader-b")
    )
    report = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=judgments,
        blind_map=blind_map,
        bootstrap_seed=design.bootstrap.seed,
        bootstrap_samples=design.bootstrap.samples,
    )
    return manifest, report


def _confirmatory_report_run(manifest):
    records = tuple(
        RunRecord(
            planned_run_id=planned.planned_run_id,
            outcome="success",
            output="answer",
            latency_ms=300 if planned.condition_id == "elite_full" else 100,
            cost_usd=0.01,
            cost_receipt_complete=True,
        )
        for planned in manifest.planned_runs
    )
    study_run = StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(records),
        outcome_counts={"success": len(records)},
        total_completion_tokens=0,
        total_latency_ms=sum(record.latency_ms for record in records),
        total_cost_usd=sum(record.cost_usd for record in records),
    )
    return study_run


def _gate(manifest, report):
    study_run = _confirmatory_report_run(manifest)
    blind_map, _ = _blind_targets(study_run)
    return evaluate_confirmatory_gates(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=report.raw_judgments,
        report=report,
        blind_map=blind_map,
    )


def _rebind_evidence(report):
    bounds = report.simultaneous_upper_bounds
    assert bounds is not None
    return report.model_copy(
        update={
            "simultaneous_upper_bounds": bounds.model_copy(
                update={"evidence_hash": hash_confirmatory_evidence(report)}
            )
        }
    )


def test_confirmatory_gate_is_bound_to_manifest_report_and_simultaneous_bounds() -> None:
    manifest, report = _confirmatory_report()
    result = _gate(manifest, report)
    assert result.status == "GO"

    undersized_manifest, undersized_report = _confirmatory_report(minimum_tasks=3)
    assert "sample_size" in _gate(undersized_manifest, undersized_report).failed_gates

    assert report.simultaneous_upper_bounds is not None
    assert report.simultaneous_upper_bounds.method == "task_clustered_percentile_bonferroni_v1"
    assert report.simultaneous_upper_bounds.endpoint_count == 3
    assert report.simultaneous_upper_bounds.per_endpoint_alpha == pytest.approx(0.05 / 3)
    assert report.simultaneous_upper_bounds.quantile_probability == pytest.approx(1 - 0.05 / 3)
    tampered = report.model_copy(
        update={
            "simultaneous_upper_bounds": report.simultaneous_upper_bounds.model_copy(
                update={"evidence_hash": "sha256:" + "f" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="re-scored raw artifacts"):
        _gate(manifest, tampered)


def test_confirmatory_gate_uses_every_derived_report_boundary() -> None:
    manifest, report = _confirmatory_report()
    paid = report.paid_analysis
    assert paid is not None
    cases = {
        "minimum_effect": paid.model_copy(
            update={
                "primary_paired_difference": paid.primary_paired_difference.model_copy(
                    update={"estimate": 0.09}
                )
            }
        ),
        "superiority_interval": paid.model_copy(
            update={
                "primary_paired_difference": paid.primary_paired_difference.model_copy(
                    update={"lower": 0.0}
                )
            }
        ),
        "latency_ratio": paid.model_copy(update={"p95_latency_ratio": 3.001}),
        "latency_ceiling": paid.model_copy(update={"elite_p95_latency_seconds": 180.1}),
        "spend_ceiling": paid.model_copy(update={"total_cost_usd": 251.0}),
        "family_direction": paid.model_copy(
            update={"family_directions": {"operational": 0.0, "stewardship": 1.0}}
        ),
        "roster_direction": paid.model_copy(
            update={"roster_directions": {"roster-a": 0.0, "roster-b": 1.0}}
        ),
    }
    for changed in cases.values():
        changed_report = report.model_copy(update={"paid_analysis": changed})
        changed_report = _rebind_evidence(changed_report)
        with pytest.raises(ValueError, match="re-scored raw artifacts"):
            _gate(manifest, changed_report)


@pytest.mark.parametrize(
    ("expected_gate", "updates"),
    (
        (
            "severe_error_noninferiority",
            {"elite_severe_error_rate": 1.0, "baseline_severe_error_rate": 0.0},
        ),
        (
            "readiness_noninferiority",
            {"elite_readiness_error_rate": 1.0, "baseline_readiness_error_rate": 0.0},
        ),
        (
            "reviewer_effort",
            {"elite_reviewer_effort": 25.0, "baseline_reviewer_effort": 10.0},
        ),
    ),
)
def test_confirmatory_gate_recomputes_secondary_bounds_from_task_inputs(
    expected_gate, updates
) -> None:
    manifest, report = _confirmatory_report()
    paid = report.paid_analysis
    assert paid is not None
    changed_paid = paid.model_copy(
        update={
            "secondary_task_statistics": tuple(
                item.model_copy(update=updates) for item in paid.secondary_task_statistics
            )
        }
    )
    changed_report = report.model_copy(update={"paid_analysis": changed_paid})
    changed_report = changed_report.model_copy(
        update={
            "simultaneous_upper_bounds": recompute_simultaneous_upper_bounds(
                manifest=manifest, report=changed_report
            )
        }
    )

    bounds = changed_report.simultaneous_upper_bounds
    assert bounds is not None
    values = {
        "severe_error_noninferiority": bounds.severe_error_upper_bound,
        "readiness_noninferiority": bounds.readiness_error_upper_bound,
        "reviewer_effort": bounds.reviewer_effort_ratio_upper_bound,
    }
    thresholds = {
        "severe_error_noninferiority": manifest.frozen_design.analysis_gates.severe_error_noninferiority_margin,
        "readiness_noninferiority": manifest.frozen_design.analysis_gates.readiness_noninferiority_margin,
        "reviewer_effort": manifest.frozen_design.analysis_gates.reviewer_effort_max_ratio,
    }
    assert values[expected_gate] > thresholds[expected_gate]
    with pytest.raises(ValueError, match="re-scored raw artifacts"):
        _gate(manifest, changed_report)


def test_caller_cannot_substitute_optimistic_bounds_to_turn_redesign_into_go() -> None:
    manifest, report = _confirmatory_report()
    study_run = _confirmatory_report_run(manifest)
    blind_map, opaque_by_run = _blind_targets(study_run)
    condition_by_output = {
        opaque_by_run[planned.planned_run_id]: planned.condition_id
        for planned in manifest.planned_runs
    }
    severe = AtomicError(
        rubric_item="safety",
        category="severe_harm",
        severity="severe",
        detail="Confirmatory severe error.",
    )
    bad_judgments = tuple(
        judgment.model_copy(
            update={
                "critical_error_free": False,
                "atomic_errors": (severe,),
                "severe_error": True,
            }
        )
        if condition_by_output[judgment.opaque_output_id] == "elite_full"
        else judgment
        for judgment in report.raw_judgments
    )
    redesign_report = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=bad_judgments,
        blind_map=blind_map,
        bootstrap_seed=manifest.frozen_design.bootstrap.seed,
        bootstrap_samples=manifest.frozen_design.bootstrap.samples,
    )
    assert (
        evaluate_confirmatory_gates(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=bad_judgments,
            report=redesign_report,
            blind_map=blind_map,
        ).status
        == "REDESIGN"
    )

    bounds = redesign_report.simultaneous_upper_bounds
    paid = redesign_report.paid_analysis
    assert bounds is not None and paid is not None
    substituted = redesign_report.model_copy(
        update={
            "paid_analysis": paid.model_copy(
                update={
                    "secondary_task_statistics": tuple(
                        item.model_copy(
                            update={
                                "elite_severe_error_rate": 0.0,
                                "baseline_severe_error_rate": 0.0,
                            }
                        )
                        for item in paid.secondary_task_statistics
                    )
                }
            ),
            "simultaneous_upper_bounds": bounds.model_copy(
                update={"severe_error_upper_bound": 0.0}
            ),
        }
    )
    substituted = _rebind_evidence(substituted)
    with pytest.raises(ValueError, match="re-scored raw artifacts"):
        evaluate_confirmatory_gates(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=bad_judgments,
            report=substituted,
            blind_map=blind_map,
        )


def test_confirmatory_gate_refuses_exploratory_missing_bounds_or_provenance_drift() -> None:
    manifest, report = _confirmatory_report()
    with pytest.raises(ValueError, match="confirmatory"):
        _gate(
            manifest.model_copy(update={"evidence_classification": "paid_exploratory_pilot"}),
            report,
        )
    with pytest.raises(ValueError, match="simultaneous"):
        _gate(manifest, report.model_copy(update={"simultaneous_upper_bounds": None}))
    with pytest.raises(ValueError, match="frozen design hash"):
        _gate(
            manifest,
            report.model_copy(update={"frozen_design_hash": "sha256:" + "c" * 64}),
        )
    with pytest.raises(ValueError, match="preregistration hash"):
        _gate(
            manifest,
            report.model_copy(update={"preregistration_hash": "sha256:" + "c" * 64}),
        )
    with pytest.raises(ValueError, match="analysis code hash"):
        _gate(
            manifest,
            report.model_copy(update={"analysis_code_hash": "sha256:" + "c" * 64}),
        )


def test_task_exclusion_refuses_to_empty_a_frozen_family() -> None:
    design = _design().model_copy(
        update={
            "exclusion_deviation_policy": _design().exclusion_deviation_policy.model_copy(
                update={"excluded_task_ids": ("task-a",)}
            )
        }
    )
    manifest = build_study_manifest(
        study_id="excluded-task",
        tasks=_tasks(),
        replicates=1,
        seed=20260717,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 100),
        frozen_design=design,
    )

    with pytest.raises(ValueError, match="family operational"):
        analysis_planned_runs(manifest)


def test_reviewer_effort_ratio_uses_task_clustered_median_ratio_with_95_percent_interval() -> None:
    result = paired_bootstrap_ratio(
        {"a": 1.0, "b": 1.0, "c": 100.0},
        {"a": 1.0, "b": 1.0, "c": 1.0},
        seed=7,
        samples=1000,
    )

    assert result.estimate == pytest.approx(1.0)
    assert result.confidence_level == 0.95
    with pytest.raises(ValueError, match="strictly positive"):
        paired_bootstrap_ratio({"a": 1.0}, {"a": 0.0}, seed=7, samples=100)


def test_confirmatory_gate_uses_single_model_latency_baseline_and_reliability() -> None:
    manifest, report = _confirmatory_report()
    paid = report.paid_analysis
    assert paid is not None
    assert paid.latency_baseline_condition_id == "single_frontier"

    weak = report.model_copy(
        update={
            "reliability": report.reliability.model_copy(
                update={"raw_agreement": 0.79, "cohen_kappa": 0.59}
            )
        }
    )
    with pytest.raises(ValueError, match="re-scored raw artifacts"):
        _gate(manifest, _rebind_evidence(weak))


def test_rotating_grader_pairs_still_produce_complete_reliability() -> None:
    manifest, report = _confirmatory_report()
    study_run = _confirmatory_report_run(manifest)
    blind_map, _ = _blind_targets(study_run)
    raw = []
    for index, judgment in enumerate(report.raw_judgments):
        pair = index // 2
        raw.append(
            judgment.model_copy(
                update={
                    "grader_id": f"grader-{pair % 3}-{index % 2}",
                    "judgment_id": f"rotated-{index}",
                }
            )
        )
    rescored = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=tuple(raw),
        blind_map=blind_map,
        bootstrap_seed=manifest.frozen_design.bootstrap.seed,
        bootstrap_samples=manifest.frozen_design.bootstrap.samples,
    )
    assert rescored.reliability.double_grading_rate == 1.0
    assert rescored.reliability.paired_judgments == len(manifest.planned_runs)
