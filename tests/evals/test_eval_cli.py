from __future__ import annotations

import json

from typer.testing import CliRunner

from conclave.cli import app
from conclave.evals.models import PublicTask, RunRecord, StudyRun
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest

runner = CliRunner()


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _study_artifacts(tmp_path):
    tasks = [PublicTask(task_id="task-a", prompt="Choose one.")]
    tasks_path = tmp_path / "tasks.json"
    _write_json(
        tasks_path,
        {
            "schema_version": "conclave_eval_v1",
            "tasks": [task.model_dump(mode="json") for task in tasks],
        },
    )
    manifest = build_study_manifest(
        study_id="offline-cli",
        tasks=tasks,
        replicates=1,
        seed=7,
        output_token_budgets={condition_id: 100 for condition_id in CONDITION_IDS},
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    records = tuple(
        RunRecord(planned_run_id=item.planned_run_id, outcome="success", output="fixture")
        for item in manifest.planned_runs
    )
    study_run = StudyRun(
        study_id=manifest.study_id,
        records=records,
        total_planned_runs=len(records),
        outcome_counts={"success": len(records)},
        total_completion_tokens=0,
        total_latency_ms=0,
    )
    run_path = tmp_path / "run.json"
    _write_json(run_path, study_run.model_dump(mode="json"))
    return tasks_path, manifest_path, run_path, manifest


def test_eval_plan_writes_complete_manifest_atomically(tmp_path) -> None:
    tasks_path, _, _, _ = _study_artifacts(tmp_path)
    output = tmp_path / "nested" / "manifest.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "plan",
            str(tasks_path),
            str(output),
            "--study-id",
            "cli-plan",
            "--replicates",
            "2",
            "--seed",
            "19",
            "--max-output-tokens",
            "120",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["study_id"] == "cli-plan"
    assert payload["replicates"] == 2
    assert len(payload["planned_runs"]) == 12
    assert {item["max_output_tokens"] for item in payload["planned_runs"]} == {120}


def test_eval_run_rejects_live_and_validates_offline_replay_artifact(tmp_path) -> None:
    _, manifest_path, run_path, _ = _study_artifacts(tmp_path)
    output = tmp_path / "validated-run.json"

    rejected = runner.invoke(app, ["eval", "run", str(manifest_path), str(output)])
    assert rejected.exit_code == 2
    assert "offline replay artifact" in rejected.output.lower()

    accepted = runner.invoke(
        app,
        [
            "eval",
            "run",
            str(manifest_path),
            str(output),
            "--replay-artifact",
            str(run_path),
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(output.read_text())["total_planned_runs"] == 6

    tampered = json.loads(run_path.read_text())
    tampered["outcome_counts"] = {"success": 5}
    tampered_path = tmp_path / "tampered-run.json"
    _write_json(tampered_path, tampered)
    rejected_tamper = runner.invoke(
        app,
        [
            "eval",
            "run",
            str(manifest_path),
            str(output),
            "--replay-artifact",
            str(tampered_path),
        ],
    )
    assert rejected_tamper.exit_code == 2
    assert "summary" in rejected_tamper.output.lower()


def test_eval_run_rejects_records_that_bypass_manifest_execution_invariants(tmp_path) -> None:
    _, manifest_path, run_path, manifest = _study_artifacts(tmp_path)
    output = tmp_path / "validated-run.json"

    for name, replacement, expected in (
        (
            "over-budget",
            {"completion_tokens": manifest.planned_runs[0].max_output_tokens + 1},
            "output budget",
        ),
        ("missing-output", {"output": None}, "success output"),
    ):
        payload = json.loads(run_path.read_text())
        payload["records"][0].update(replacement)
        payload["total_completion_tokens"] = sum(
            item.get("completion_tokens") or 0 for item in payload["records"]
        )
        tampered_path = tmp_path / f"{name}.json"
        _write_json(tampered_path, payload)

        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                str(manifest_path),
                str(output),
                "--replay-artifact",
                str(tampered_path),
            ],
        )

        assert result.exit_code == 2
        assert expected in result.output.lower()


def test_eval_blind_writes_separate_grader_and_identity_artifacts(tmp_path) -> None:
    _, _, run_path, _ = _study_artifacts(tmp_path)
    grader_path = tmp_path / "grader.json"
    map_path = tmp_path / "restricted" / "blind-map.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "blind",
            str(run_path),
            str(grader_path),
            str(map_path),
            "--seed",
            "23",
        ],
    )

    assert result.exit_code == 0, result.output
    grader_payload = json.loads(grader_path.read_text())
    map_payload = json.loads(map_path.read_text())
    assert "planned_run_id" not in grader_path.read_text()
    assert len(grader_payload["outputs"]) == len(map_payload["entries"]) == 6


def test_eval_report_writes_json_and_markdown_from_atomic_inputs(tmp_path) -> None:
    _, manifest_path, run_path, manifest = _study_artifacts(tmp_path)
    judgments_path = tmp_path / "judgments.json"
    _write_json(
        judgments_path,
        {
            "judgments": [
                {
                    "schema_version": "conclave_eval_v1",
                    "judgment_id": f"j-{index}",
                    "planned_run_id": item.planned_run_id,
                    "grader_id": "human-a",
                    "critical_error_free": True,
                }
                for index, item in enumerate(manifest.planned_runs)
            ]
        },
    )
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    result = runner.invoke(
        app,
        [
            "eval",
            "report",
            str(manifest_path),
            str(run_path),
            str(judgments_path),
            str(json_output),
            str(markdown_output),
            "--bootstrap-seed",
            "29",
            "--bootstrap-samples",
            "50",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(json_output.read_text())["decision_eligibility"] == "not_yet_eligible"
    assert "SYNTHETIC / EXPLORATORY" in markdown_output.read_text()
