from __future__ import annotations

import json

from conclave.evals.models import PublicTask, RunRecord, StudyRun
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.reporting import render_markdown_report, write_report_bundle
from conclave.evals.scoring import GraderJudgment, score_study


def _synthetic_report():
    tasks = [PublicTask(task_id="task-a", prompt="Choose.")]
    manifest = build_study_manifest(
        study_id="synthetic-report",
        tasks=tasks,
        replicates=1,
        seed=7,
        output_token_budgets={condition_id: 100 for condition_id in CONDITION_IDS},
    )
    records = tuple(
        RunRecord(
            planned_run_id=planned.planned_run_id,
            outcome="success",
            output="synthetic answer",
        )
        for planned in manifest.planned_runs
    )
    study_run = StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(records),
        outcome_counts={"success": len(records)},
        total_completion_tokens=0,
        total_latency_ms=0,
    )
    judgments = tuple(
        GraderJudgment(
            judgment_id=f"judgment-{index}",
            planned_run_id=record.planned_run_id,
            grader_id="synthetic-grader",
            critical_error_free=index % 2 == 0,
        )
        for index, record in enumerate(records)
    )
    return score_study(
        manifest=manifest,
        study_run=study_run,
        raw_judgments=judgments,
        bootstrap_seed=11,
        bootstrap_samples=100,
    )


def test_markdown_report_is_explicitly_exploratory_and_complete() -> None:
    markdown = render_markdown_report(_synthetic_report())

    assert "SYNTHETIC / EXPLORATORY" in markdown
    assert "not yet eligible" in markdown.lower()
    assert "Outcome distribution" in markdown
    assert "95% Wilson CI" in markdown
    assert "Paired bootstrap" in markdown
    assert "Raw grader provenance" in markdown
    assert "Go / redesign / kill" in markdown
    assert "confirmatory" in markdown.lower()


def test_markdown_report_uses_the_recorded_evidence_classification() -> None:
    report = _synthetic_report().model_copy(
        update={"evidence_classification": "paid_exploratory_pilot"}
    )

    markdown = render_markdown_report(report)

    assert "PAID / EXPLORATORY PILOT" in markdown
    assert "SYNTHETIC / EXPLORATORY" not in markdown


def test_report_bundle_writes_machine_json_and_human_markdown(tmp_path) -> None:
    report = _synthetic_report()
    json_path = tmp_path / "nested" / "report.json"
    markdown_path = tmp_path / "nested" / "report.md"

    write_report_bundle(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text())
    assert payload["study_id"] == "synthetic-report"
    assert payload["raw_judgments"]
    assert markdown_path.read_text().startswith("# Conclave Elite Evaluation")
