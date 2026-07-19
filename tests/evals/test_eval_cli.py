from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

import conclave.eval_cli as eval_cli_module
from conclave.cli import app
from conclave.evals.live import (
    build_checkpoint_bindings,
    create_live_checkpoint,
    load_live_checkpoint,
    write_live_checkpoint,
)
from conclave.evals.live_protocols import stage_call_sequence
from conclave.evals.models import PublicTask, RunRecord, StudyRun
from conclave.evals.protocols import CONDITION_IDS, build_study_manifest
from conclave.evals.runner import run_live_study as execute_live_study
from conclave.models import ModelAnswer, TokenUsage
from tests.evals.test_live_runner import _live_inputs

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]


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


def _live_artifacts(tmp_path, **input_overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tasks, manifest, price_book = _live_inputs(**input_overrides)
    tasks_path = tmp_path / "live-tasks.json"
    manifest_path = tmp_path / "live-manifest.json"
    price_book_path = tmp_path / "live-price-book.json"
    _write_json(
        tasks_path,
        {
            "schema_version": "conclave_eval_v1",
            "tasks": [task.model_dump(mode="json") for task in tasks],
        },
    )
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    _write_json(price_book_path, price_book.model_dump(mode="json"))
    return tasks, manifest, price_book, tasks_path, manifest_path, price_book_path


def _live_command(
    manifest_path,
    tasks_path,
    price_book_path,
    output_path,
    checkpoint_path,
    receipts_path,
    *options,
) -> list[str]:
    return [
        "eval",
        "live",
        str(manifest_path),
        str(tasks_path),
        str(price_book_path),
        str(output_path),
        str(checkpoint_path),
        str(receipts_path),
        *options,
    ]


async def _successful_live_provider(name, model_id, messages, **kwargs):
    del messages, kwargs
    return ModelAnswer(
        name=name,
        model_id=model_id,
        answer=f"fictional decision from {name}",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_eval_live_defaults_to_dry_run_and_never_calls_provider(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, _, tasks_path, manifest_path, price_book_path = _live_artifacts(tmp_path)
    output_path = tmp_path / "run.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    receipts_path = tmp_path / "receipts.json"
    provider_calls = 0

    async def forbidden_live_runner(**kwargs):
        nonlocal provider_calls
        del kwargs
        provider_calls += 1
        raise AssertionError("dry-run must never enter the provider-backed runner")

    monkeypatch.setattr(eval_cli_module, "run_live_study", forbidden_live_runner, raising=False)

    result = runner.invoke(
        app,
        _live_command(
            manifest_path,
            tasks_path,
            price_book_path,
            output_path,
            checkpoint_path,
            receipts_path,
            "--approve-spend-usd",
            "10.00",
        ),
    )

    assert result.exit_code == 0, result.output
    estimate = json.loads(result.stdout)
    assert estimate["planned_cells"] == len(manifest.planned_runs)
    assert Decimal(estimate["ceiling_usd"]) == Decimal("10.00")
    assert estimate["fits_ceiling"] is True
    assert provider_calls == 0
    assert not output_path.exists()
    assert not checkpoint_path.exists()
    assert not receipts_path.exists()


def test_eval_live_requires_execute_and_exact_frozen_spend_approval(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, tasks_path, manifest_path, price_book_path = _live_artifacts(tmp_path)
    provider_calls = 0

    async def forbidden_live_runner(**kwargs):
        nonlocal provider_calls
        del kwargs
        provider_calls += 1
        raise AssertionError("invalid approval must fail before provider execution")

    monkeypatch.setattr(eval_cli_module, "run_live_study", forbidden_live_runner, raising=False)

    for index, options in enumerate(
        (
            ("--execute",),
            ("--execute", "--approve-spend-usd", "9.99"),
            ("--execute", "--approve-spend-usd", "10.01"),
        )
    ):
        result = runner.invoke(
            app,
            _live_command(
                manifest_path,
                tasks_path,
                price_book_path,
                tmp_path / f"run-{index}.json",
                tmp_path / f"checkpoint-{index}.json",
                tmp_path / f"receipts-{index}.json",
                *options,
            ),
        )

        assert result.exit_code == 2
        assert "approval" in result.output.lower()

    dry_run = runner.invoke(
        app,
        _live_command(
            manifest_path,
            tasks_path,
            price_book_path,
            tmp_path / "dry-run.json",
            tmp_path / "dry-checkpoint.json",
            tmp_path / "dry-receipts.json",
            "--approve-spend-usd",
            "10.00",
        ),
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert provider_calls == 0


def test_live_evaluation_docs_preserve_operator_and_claim_boundaries() -> None:
    docs = {
        path: " ".join((ROOT / path).read_text(encoding="utf-8").lower().split())
        for path in (
            "README.md",
            "SYSTEM_CONTEXT_DIAGRAM.md",
            "DOCUMENTATION_INDEX.md",
            "docs/PRODUCT_DESIGN_DOCUMENT.md",
            "CHANGELOG.md",
        )
    }
    corpus = " ".join(docs.values())

    assert all("paid exploratory" in text for text in docs.values())
    assert "paid exploratory only" in corpus
    assert "dry-run is the default" in corpus
    assert "--execute" in corpus
    assert "--approve-spend-usd 10.00" in corpus
    assert "one provider call is in flight" in corpus
    assert "reservation is persisted before each call" in corpus
    assert "resume never repeats an interrupted cell" in corpus
    assert "correctness only" in corpus
    assert "not efficiency or decision quality" in corpus
    assert "not decision eligible" in corpus


def test_eval_live_rejects_confirmatory_legacy_or_snapshot_drift(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, live_tasks_path, live_manifest_path, live_price_book_path = _live_artifacts(
        tmp_path / "valid"
    )
    _, _, _, confirmatory_tasks, confirmatory_manifest, confirmatory_price_book = _live_artifacts(
        tmp_path / "confirmatory", evidence_classification="confirmatory"
    )
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_tasks, legacy_manifest, _, _ = _study_artifacts(legacy_dir)
    drifted_payload = json.loads(live_price_book_path.read_text(encoding="utf-8"))
    drifted_payload["snapshot_id"] = "drifted-snapshot"
    drifted_price_book = tmp_path / "drifted-price-book.json"
    _write_json(drifted_price_book, drifted_payload)
    boundary_calls = 0

    async def forbidden_boundary(**kwargs):
        nonlocal boundary_calls
        del kwargs
        boundary_calls += 1
        raise AssertionError("invalid live artifacts must fail before keys or network")

    monkeypatch.setattr(eval_cli_module, "estimate_live_study", forbidden_boundary, raising=False)
    monkeypatch.setattr(eval_cli_module, "run_live_study", forbidden_boundary, raising=False)

    cases = (
        (
            confirmatory_manifest,
            confirmatory_tasks,
            confirmatory_price_book,
            "paid exploratory",
        ),
        (legacy_manifest, legacy_tasks, live_price_book_path, "paid exploratory"),
        (live_manifest_path, live_tasks_path, drifted_price_book, "snapshot"),
    )
    for index, (manifest_path, tasks_path, price_book_path, expected) in enumerate(cases):
        result = runner.invoke(
            app,
            _live_command(
                manifest_path,
                tasks_path,
                price_book_path,
                tmp_path / f"invalid-run-{index}.json",
                tmp_path / f"invalid-checkpoint-{index}.json",
                tmp_path / f"invalid-receipts-{index}.json",
                "--execute",
                "--approve-spend-usd",
                "10.00",
            ),
        )

        assert result.exit_code == 2
        assert expected in result.output.lower()

    assert boundary_calls == 0


def test_eval_live_writes_checkpoint_receipts_and_study_run_atomically(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, _, tasks_path, manifest_path, price_book_path = _live_artifacts(tmp_path)
    output_path = tmp_path / "artifacts" / "run.json"
    checkpoint_path = tmp_path / "state" / "checkpoint.json"
    receipts_path = tmp_path / "artifacts" / "receipts.json"
    atomic_paths = []
    real_atomic_json = eval_cli_module._atomic_json

    async def injected_runner(**kwargs):
        return await execute_live_study(
            **kwargs,
            call_model_func=_successful_live_provider,
        )

    def recording_atomic_json(path, value):
        atomic_paths.append(path)
        real_atomic_json(path, value)

    monkeypatch.setattr(eval_cli_module, "run_live_study", injected_runner, raising=False)
    monkeypatch.setattr(eval_cli_module, "_atomic_json", recording_atomic_json)

    result = runner.invoke(
        app,
        _live_command(
            manifest_path,
            tasks_path,
            price_book_path,
            output_path,
            checkpoint_path,
            receipts_path,
            "--execute",
            "--approve-spend-usd",
            "10.00",
        ),
    )

    assert result.exit_code == 0, result.output
    run_payload = json.loads(output_path.read_text(encoding="utf-8"))
    receipt_payload = json.loads(receipts_path.read_text(encoding="utf-8"))
    assert run_payload["total_planned_runs"] == len(manifest.planned_runs)
    assert receipt_payload["study_id"] == manifest.study_id
    assert receipt_payload["receipts"]
    assert checkpoint_path.exists()
    assert atomic_paths == [output_path, receipts_path]
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_eval_live_resume_uses_existing_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, price_book, tasks_path, manifest_path, price_book_path = _live_artifacts(tmp_path)
    output_path = tmp_path / "run.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    receipts_path = tmp_path / "receipts.json"
    first_run = manifest.planned_runs[0]
    preserved = RunRecord(
        planned_run_id=first_run.planned_run_id,
        outcome="success",
        output="preserved checkpoint result",
        cost_receipt_complete=True,
    )
    bindings = build_checkpoint_bindings(
        manifest,
        price_book,
        hard_cap_usd=Decimal("10.00"),
    )
    write_live_checkpoint(
        checkpoint_path,
        create_live_checkpoint(bindings=bindings, records=(preserved,)),
    )
    provider_calls = 0

    async def counting_provider(name, model_id, messages, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return await _successful_live_provider(name, model_id, messages, **kwargs)

    async def injected_runner(**kwargs):
        assert kwargs["checkpoint_path"] == checkpoint_path
        return await execute_live_study(**kwargs, call_model_func=counting_provider)

    monkeypatch.setattr(eval_cli_module, "run_live_study", injected_runner, raising=False)

    result = runner.invoke(
        app,
        _live_command(
            manifest_path,
            tasks_path,
            price_book_path,
            output_path,
            checkpoint_path,
            receipts_path,
            "--execute",
            "--approve-spend-usd",
            "10.00",
        ),
    )

    assert result.exit_code == 0, result.output
    run_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert run_payload["records"][0]["output"] == "preserved checkpoint result"
    expected_calls = sum(
        len(stage_call_sequence(item.condition_id, roster_size=3))
        for item in manifest.planned_runs[1:]
    )
    assert provider_calls == expected_calls
    resumed_checkpoint = load_live_checkpoint(
        checkpoint_path,
        expected_bindings=bindings,
    )
    assert resumed_checkpoint.records[0] == preserved


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
