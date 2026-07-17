from __future__ import annotations

import pytest
from pydantic import ValidationError

from conclave.evals.models import AnalysisGateConfig, RunRecord, StudyRun
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.scoring import (
    ConfirmatoryInference,
    analysis_planned_runs,
    evaluate_confirmatory_gates,
    score_study,
)
from tests.evals.test_method_grading import _judgment
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
    assert config.p95_latency_max_ratio == 3.0
    assert config.minimum_confirmatory_tasks == 2
    assert config.multiplicity_rule == "holm_secondary"
    assert config.unit == "task"
    with pytest.raises(ValidationError, match="primary_baseline"):
        _config(primary_baseline="elite_full")


def _passing_inference(**updates):
    values = {
        "achieved_power": 0.80,
        "secondary_p_values": {
            "severe_error_noninferiority": 0.01,
            "readiness_noninferiority": 0.01,
            "reviewer_effort": 0.01,
        },
    }
    values.update(updates)
    return ConfirmatoryInference(**values)


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
    judgments = tuple(
        _judgment(record.planned_run_id, grader, value=planned.condition_id != "self_refine")
        for planned, record in zip(manifest.planned_runs, records, strict=True)
        for grader in ("grader-a", "grader-b")
    )
    report = score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=judgments,
        bootstrap_seed=design.bootstrap.seed,
        bootstrap_samples=design.bootstrap.samples,
        confirmatory_inference=_passing_inference(),
    )
    return manifest, report


def test_confirmatory_gate_is_bound_to_manifest_report_power_and_holm() -> None:
    manifest, report = _confirmatory_report()
    result = evaluate_confirmatory_gates(manifest=manifest, report=report)
    assert result.status == "GO"

    undersized_manifest, undersized_report = _confirmatory_report(minimum_tasks=3)
    assert (
        "sample_size"
        in evaluate_confirmatory_gates(
            manifest=undersized_manifest, report=undersized_report
        ).failed_gates
    )

    weak_power = report.model_copy(
        update={"confirmatory_inference": _passing_inference(achieved_power=0.79)}
    )
    assert "power" in evaluate_confirmatory_gates(manifest=manifest, report=weak_power).failed_gates

    holm_failure = report.model_copy(
        update={
            "confirmatory_inference": _passing_inference(
                secondary_p_values={
                    "severe_error_noninferiority": 0.01,
                    "readiness_noninferiority": 0.03,
                    "reviewer_effort": 0.04,
                }
            )
        }
    )
    assert (
        "holm_secondary"
        in evaluate_confirmatory_gates(manifest=manifest, report=holm_failure).failed_gates
    )


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
        "severe_error_noninferiority": paid.model_copy(
            update={
                "severe_error_difference": paid.severe_error_difference.model_copy(
                    update={"upper": 0.021}
                )
            }
        ),
        "readiness_noninferiority": paid.model_copy(
            update={
                "readiness_error_difference": paid.readiness_error_difference.model_copy(
                    update={"upper": 0.051}
                )
            }
        ),
        "reviewer_effort": paid.model_copy(
            update={
                "reviewer_effort_ratio_interval": paid.reviewer_effort_ratio_interval.model_copy(
                    update={"upper": 1.201}
                )
            }
        ),
        "latency_ratio": paid.model_copy(update={"p95_latency_ratio": 3.001}),
        "latency_ceiling": paid.model_copy(update={"elite_p95_latency_seconds": 180.1}),
        "family_direction": paid.model_copy(
            update={"family_directions": {"operational": 0.0, "stewardship": 1.0}}
        ),
        "roster_direction": paid.model_copy(
            update={"roster_directions": {"roster-a": 0.0, "roster-b": 1.0}}
        ),
    }
    for expected_gate, changed in cases.items():
        result = evaluate_confirmatory_gates(
            manifest=manifest,
            report=report.model_copy(update={"paid_analysis": changed}),
        )
        assert expected_gate in result.failed_gates


def test_confirmatory_gate_refuses_exploratory_missing_inference_or_provenance_drift() -> None:
    manifest, report = _confirmatory_report()
    with pytest.raises(ValueError, match="confirmatory"):
        evaluate_confirmatory_gates(
            manifest=manifest.model_copy(
                update={"evidence_classification": "paid_exploratory_pilot"}
            ),
            report=report,
        )
    with pytest.raises(ValueError, match="inference"):
        evaluate_confirmatory_gates(
            manifest=manifest, report=report.model_copy(update={"confirmatory_inference": None})
        )
    with pytest.raises(ValueError, match="frozen design hash"):
        evaluate_confirmatory_gates(
            manifest=manifest,
            report=report.model_copy(update={"frozen_design_hash": "sha256:" + "c" * 64}),
        )
    with pytest.raises(ValueError, match="preregistration hash"):
        evaluate_confirmatory_gates(
            manifest=manifest,
            report=report.model_copy(update={"preregistration_hash": "sha256:" + "c" * 64}),
        )
    with pytest.raises(ValueError, match="analysis code hash"):
        evaluate_confirmatory_gates(
            manifest=manifest,
            report=report.model_copy(update={"analysis_code_hash": "sha256:" + "c" * 64}),
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
