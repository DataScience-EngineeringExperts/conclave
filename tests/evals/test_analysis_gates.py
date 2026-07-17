from __future__ import annotations

import pytest
from pydantic import ValidationError

from conclave.evals.models import AnalysisGateConfig
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.scoring import (
    ConfirmatoryGateMetrics,
    analysis_planned_runs,
    evaluate_confirmatory_gates,
)
from tests.evals.test_study_design import _design, _tasks


def _config(**updates):
    values = {
        "primary_baseline": "self_refine",
        "absolute_p95_latency_seconds": 180,
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
    assert config.multiplicity_rule == "holm_secondary"
    assert config.unit == "task"
    with pytest.raises(ValidationError, match="primary_baseline"):
        _config(primary_baseline="elite_full")


def _passing_metrics(**updates):
    values = {
        "primary_effect": 0.12,
        "primary_lower_95": 0.01,
        "severe_error_upper_95": 0.02,
        "readiness_error_upper_95": 0.05,
        "reviewer_effort_upper_ratio_95": 1.20,
        "p95_latency_ratio": 3.0,
        "elite_p95_latency_seconds": 180,
        "family_directions": {"operational": 0.01, "stewardship": 0.02},
        "roster_directions": {"roster-a": 0.01, "roster-b": 0.01},
    }
    values.update(updates)
    return ConfirmatoryGateMetrics(**values)


def test_confirmatory_gate_boundaries_are_inclusive_and_every_gate_is_required() -> None:
    config = _config()
    result = evaluate_confirmatory_gates(
        evidence_classification="confirmatory",
        config=config,
        metrics=_passing_metrics(),
        expected_frozen_design_hash="sha256:" + "a" * 64,
        observed_frozen_design_hash="sha256:" + "a" * 64,
        expected_preregistration_hash="sha256:" + "b" * 64,
        observed_preregistration_hash="sha256:" + "b" * 64,
    )

    assert result.status == "GO"
    for field, value in (
        ("primary_effect", 0.099),
        ("primary_lower_95", 0.0),
        ("severe_error_upper_95", 0.021),
        ("readiness_error_upper_95", 0.051),
        ("reviewer_effort_upper_ratio_95", 1.201),
        ("p95_latency_ratio", 3.001),
        ("elite_p95_latency_seconds", 180.1),
    ):
        failed = evaluate_confirmatory_gates(
            evidence_classification="confirmatory",
            config=config,
            metrics=_passing_metrics(**{field: value}),
            expected_frozen_design_hash="sha256:" + "a" * 64,
            observed_frozen_design_hash="sha256:" + "a" * 64,
            expected_preregistration_hash="sha256:" + "b" * 64,
            observed_preregistration_hash="sha256:" + "b" * 64,
        )
        assert failed.status == "REDESIGN"

    failed_direction = evaluate_confirmatory_gates(
        evidence_classification="confirmatory",
        config=config,
        metrics=_passing_metrics(family_directions={"operational": 0.0}),
        expected_frozen_design_hash="sha256:" + "a" * 64,
        observed_frozen_design_hash="sha256:" + "a" * 64,
        expected_preregistration_hash="sha256:" + "b" * 64,
        observed_preregistration_hash="sha256:" + "b" * 64,
    )
    assert failed_direction.status == "REDESIGN"


def test_confirmatory_gate_refuses_exploratory_or_hash_drift() -> None:
    kwargs = dict(
        config=_config(),
        metrics=_passing_metrics(),
        expected_frozen_design_hash="sha256:" + "a" * 64,
        observed_frozen_design_hash="sha256:" + "a" * 64,
        expected_preregistration_hash="sha256:" + "b" * 64,
        observed_preregistration_hash="sha256:" + "b" * 64,
    )
    with pytest.raises(ValueError, match="confirmatory"):
        evaluate_confirmatory_gates(evidence_classification="paid_exploratory_pilot", **kwargs)
    with pytest.raises(ValueError, match="frozen design hash"):
        evaluate_confirmatory_gates(
            evidence_classification="confirmatory",
            **{**kwargs, "observed_frozen_design_hash": "sha256:" + "c" * 64},
        )
    with pytest.raises(ValueError, match="preregistration hash"):
        evaluate_confirmatory_gates(
            evidence_classification="confirmatory",
            **{**kwargs, "observed_preregistration_hash": "sha256:" + "c" * 64},
        )


def test_task_exclusion_is_symmetric_across_every_roster_condition_cell() -> None:
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

    included = analysis_planned_runs(manifest)
    assert {run.task_id for run in included} == {"task-b"}
    assert len(included) == 2 * 6
    assert {run.roster_id for run in included} == {"roster-a", "roster-b"}
    assert {run.condition_id for run in included} == set(CONDITION_IDS)
