from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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
    StudyManifest,
    TimeoutRetryPolicy,
)
from conclave.evals.protocols import (
    CONDITION_IDS,
    blocked_condition_order,
    build_study_manifest,
)

DIGEST = "sha256:" + "a" * 64


def _design(
    *,
    evidence_classification="paid_exploratory_pilot",
    preregistration_id=None,
    preregistration_hash=None,
    rosters=None,
):
    rosters = rosters or (
        RosterSpec(
            roster_id="roster-a",
            members=(
                ProviderModelSpec(
                    provider_id="provider-a",
                    model_id="model-a",
                    model_revision="2026-07-01",
                ),
            ),
        ),
        RosterSpec(
            roster_id="roster-b",
            members=(
                ProviderModelSpec(
                    provider_id="provider-b",
                    model_id="model-b",
                    model_revision="2026-07-02",
                ),
            ),
        ),
    )
    return FrozenStudyDesign(
        evidence_classification=evidence_classification,
        base_commit="1" * 40,
        task_family_map={"task-a": "operational", "task-b": "stewardship"},
        rosters=rosters,
        condition_prompt_versions={condition: "prompt-v1" for condition in CONDITION_IDS},
        condition_protocol_versions={condition: "protocol-v1" for condition in CONDITION_IDS},
        generation_settings_hash=DIGEST,
        evaluator_version="evaluator-v1",
        analysis_code_hash=DIGEST,
        rubric_hash=DIGEST,
        grader_instructions_hash=DIGEST,
        grader_keys_hash=DIGEST,
        exclusion_deviation_policy=ExclusionDeviationPolicy(
            predefined_task_exclusions=("duplicate", "reference-invalid"),
        ),
        timeout_retry_policy=TimeoutRetryPolicy(timeout_seconds=120, retry_attempts=1),
        randomization=RandomizationConfig(master_seed=20260717),
        bootstrap=BootstrapConfig(seed=991, samples=10000),
        analysis_gates=AnalysisGateConfig(
            primary_baseline="self_refine", absolute_p95_latency_seconds=180
        ),
        price_snapshot=PriceSnapshot(
            snapshot_id="prices-2026-07-17",
            captured_at="2026-07-17T12:00:00Z",
            currency="USD",
            prices_hash=DIGEST,
        ),
        approved_spend_ceiling_usd=250.0,
        preregistration_id=preregistration_id,
        preregistration_hash=preregistration_hash,
    )


def _tasks():
    return [
        PublicTask(task_id="task-b", prompt="B"),
        PublicTask(task_id="task-a", prompt="A"),
    ]


def test_frozen_design_captures_complete_provenance_and_is_tamper_evident() -> None:
    manifest = build_study_manifest(
        study_id="paid-pilot",
        tasks=_tasks(),
        replicates=1,
        seed=20260717,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 1000),
        frozen_design=_design(),
    )
    payload = manifest.model_dump(mode="json")

    assert manifest.evidence_classification == "paid_exploratory_pilot"
    assert manifest.promotable is False
    assert manifest.frozen_design_hash.startswith("sha256:")
    assert payload["frozen_design"]["base_commit"] == "1" * 40
    assert payload["frozen_design"]["grader_keys_hash"] == DIGEST
    assert payload["frozen_design"]["approved_spend_ceiling_usd"] == 250.0
    assert payload["frozen_design"]["exclusion_deviation_policy"]
    assert payload["frozen_design"]["timeout_retry_policy"]
    assert payload["frozen_design"]["price_snapshot"]

    tampered = json.loads(manifest.model_dump_json())
    tampered["frozen_design"]["evaluator_version"] = "changed-after-freeze"
    with pytest.raises(ValidationError, match="frozen design hash"):
        StudyManifest.model_validate(tampered)

    changed_rosters = list(_design().rosters)
    changed_rosters[0] = changed_rosters[0].model_copy(
        update={
            "members": (
                changed_rosters[0].members[0].model_copy(update={"model_revision": "2026-07-18"}),
            )
        }
    )
    changed = build_study_manifest(
        study_id="paid-pilot",
        tasks=_tasks(),
        replicates=1,
        seed=20260717,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 1000),
        frozen_design=_design(rosters=tuple(changed_rosters)),
    )
    assert {run.planned_run_id for run in changed.planned_runs}.isdisjoint(
        run.planned_run_id for run in manifest.planned_runs
    )

    tampered_run_id = json.loads(manifest.model_dump_json())
    tampered_run_id["planned_runs"][0]["planned_run_id"] = "run_" + "f" * 24
    with pytest.raises(ValidationError, match="planned_run_id"):
        StudyManifest.model_validate(tampered_run_id)


def test_confirmatory_requires_preregistration_and_every_freeze_field() -> None:
    with pytest.raises(ValidationError, match="preregistration"):
        _design(evidence_classification="confirmatory")

    design = _design(
        evidence_classification="confirmatory",
        preregistration_id="osf:conclave-confirmatory-v1",
        preregistration_hash=DIGEST,
    )
    manifest = build_study_manifest(
        study_id="confirmatory-v1",
        tasks=_tasks(),
        replicates=1,
        seed=20260717,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 1000),
        frozen_design=design,
    )

    assert manifest.evidence_classification == "confirmatory"
    assert manifest.frozen_design.preregistration_hash == DIGEST


def test_exploratory_manifest_cannot_be_promoted_after_creation() -> None:
    manifest = build_study_manifest(
        study_id="exploratory",
        tasks=_tasks(),
        replicates=1,
        seed=20260717,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 1000),
        frozen_design=_design(evidence_classification="synthetic_exploratory"),
    )
    payload = json.loads(manifest.model_dump_json())
    payload["evidence_classification"] = "confirmatory"

    with pytest.raises(ValidationError, match="classification"):
        StudyManifest.model_validate(payload)
    with pytest.raises(ValidationError):
        StudyManifest.model_validate({**json.loads(manifest.model_dump_json()), "promotable": True})


def test_design_requires_two_unique_rosters_and_exact_task_family_coverage() -> None:
    one_roster = (
        RosterSpec(
            roster_id="only",
            members=(
                ProviderModelSpec(
                    provider_id="provider", model_id="model", model_revision="revision"
                ),
            ),
        ),
    )
    with pytest.raises(ValidationError, match="at least 2"):
        _design(rosters=one_roster)

    design = _design().model_copy(
        update={"task_family_map": {"task-a": "operational", "extra": "wrong"}}
    )
    with pytest.raises(ValueError, match="task_family_map"):
        build_study_manifest(
            study_id="bad-families",
            tasks=_tasks(),
            replicates=1,
            seed=20260717,
            output_token_budgets=dict.fromkeys(CONDITION_IDS, 1000),
            frozen_design=design,
        )


def test_blocked_plan_is_stable_complete_and_independent_per_task_roster() -> None:
    design = _design()
    kwargs = dict(
        study_id="blocked-pilot",
        replicates=2,
        seed=20260717,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 1000),
        frozen_design=design,
    )
    first = build_study_manifest(tasks=_tasks(), **kwargs)
    second = build_study_manifest(tasks=list(reversed(_tasks())), **kwargs)

    assert first == second
    assert len(first.planned_runs) == 2 * 2 * 6 * 2
    assert {run.roster_id for run in first.planned_runs} == {"roster-a", "roster-b"}
    assert len({run.planned_run_id for run in first.planned_runs}) == 48
    for task_id in ("task-a", "task-b"):
        for roster_id in ("roster-a", "roster-b"):
            block = tuple(
                run.condition_id
                for run in first.planned_runs
                if run.task_id == task_id and run.roster_id == roster_id and run.replicate == 1
            )
            assert block == blocked_condition_order(
                master_seed=20260717, task_id=task_id, roster_id=roster_id
            )
    assert blocked_condition_order(
        master_seed=20260717, task_id="task-a", roster_id="roster-a"
    ) != blocked_condition_order(master_seed=20260717, task_id="task-b", roster_id="roster-b")


def test_legacy_builder_remains_synthetic_exploratory_and_readable() -> None:
    manifest = build_study_manifest(
        study_id="legacy",
        tasks=[PublicTask(task_id="task-a", prompt="A")],
        replicates=1,
        seed=7,
        output_token_budgets=dict.fromkeys(CONDITION_IDS, 1000),
    )
    round_trip = StudyManifest.model_validate_json(manifest.model_dump_json())

    assert round_trip == manifest
    assert manifest.evidence_classification == "synthetic_exploratory"
    assert manifest.frozen_design is None
    assert {run.roster_id for run in manifest.planned_runs} == {"legacy_default"}
