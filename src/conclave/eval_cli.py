"""Experimental, offline-only command surface for the H1 evaluation harness."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from .evals.blinding import BlindMap, blind_run_records
from .evals.dataset import load_public_tasks
from .evals.models import EVAL_CONDITION_IDS, StudyManifest, StudyRun
from .evals.protocols import build_study_manifest
from .evals.reporting import write_report_bundle
from .evals.runner import validate_run_records
from .evals.scoring import AdjudicationRecord, GraderJudgment, score_study

app = typer.Typer(
    add_completion=False,
    help=(
        "Experimental H1 evaluation tools (DSE-708). Offline artifacts only; "
        "these commands never call providers."
    ),
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _abort(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _validated(model_type, path: Path):
    try:
        return model_type.model_validate(_read_json(path))
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        _abort(f"invalid {path}: {exc}")


def _validate_run_summary(study_run: StudyRun) -> None:
    records = study_run.records
    expected_counts = dict(sorted(Counter(record.outcome for record in records).items()))
    expected_tokens = sum(record.completion_tokens or 0 for record in records)
    expected_latency = sum(record.latency_ms or 0.0 for record in records)
    if (
        study_run.total_planned_runs != len(records)
        or study_run.outcome_counts != expected_counts
        or study_run.total_completion_tokens != expected_tokens
        or abs(study_run.total_latency_ms - expected_latency) > 1e-9
    ):
        _abort("run artifact summary does not match its immutable records")


@app.command("plan")
def plan_command(
    public_tasks: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Argument()],
    study_id: Annotated[str, typer.Option("--study-id")],
    replicates: Annotated[int, typer.Option("--replicates", min=1)] = 1,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens", min=1)] = 1000,
) -> None:
    """Freeze a complete, equal-budget study manifest from public tasks."""

    try:
        tasks = load_public_tasks(public_tasks)
        manifest = build_study_manifest(
            study_id=study_id,
            tasks=tasks,
            replicates=replicates,
            seed=seed,
            output_token_budgets={
                condition_id: max_output_tokens for condition_id in EVAL_CONDITION_IDS
            },
        )
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        _abort(f"could not build manifest: {exc}")
    _atomic_json(output, manifest.model_dump(mode="json"))
    typer.echo(f"Wrote {len(manifest.planned_runs)} predeclared runs to {output}")


@app.command("run")
def run_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Argument()],
    replay_artifact: Annotated[
        Path | None,
        typer.Option(
            "--replay-artifact",
            exists=True,
            readable=True,
            help="Existing offline StudyRun artifact to validate. Live execution is unavailable.",
        ),
    ] = None,
) -> None:
    """Validate an offline replay artifact; live provider execution is disabled."""

    if replay_artifact is None:
        _abort(
            "an offline replay artifact is required (--replay-artifact); "
            "live provider execution is disabled in DSE-708"
        )
    manifest = _validated(StudyManifest, manifest_path)
    study_run = _validated(StudyRun, replay_artifact)
    _validate_run_summary(study_run)
    if study_run.study_id != manifest.study_id:
        _abort("replay artifact study_id does not match the manifest")
    try:
        validate_run_records(manifest, study_run.records)
    except ValueError as exc:
        _abort(f"replay artifact does not cover the frozen manifest: {exc}")
    if study_run.total_planned_runs != len(manifest.planned_runs):
        _abort("replay artifact total_planned_runs does not match the manifest")
    _atomic_json(output, study_run.model_dump(mode="json"))
    typer.echo(f"Validated offline replay artifact and wrote {output}")


@app.command("blind")
def blind_command(
    run_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    grader_output: Annotated[Path, typer.Argument()],
    blind_map_output: Annotated[Path, typer.Argument()],
    seed: Annotated[int, typer.Option("--seed")],
) -> None:
    """Write grader-visible outputs and a physically separate restricted map."""

    if grader_output.resolve() == blind_map_output.resolve():
        _abort("grader output and blind map must use different paths")
    study_run = _validated(StudyRun, run_path)
    _validate_run_summary(study_run)
    try:
        blinded, blind_map = blind_run_records(study_run.records, seed=seed)
    except ValueError as exc:
        _abort(f"could not blind run artifact: {exc}")
    _atomic_json(grader_output, blinded.model_dump(mode="json"))
    _atomic_json(blind_map_output, blind_map.model_dump(mode="json"))
    typer.echo(f"Wrote {len(blinded.outputs)} blinded outputs and separate map")


def _records(path: Path | None, key: str, model_type) -> tuple:
    if path is None:
        return ()
    try:
        payload = _read_json(path)
        values = payload[key]
        return tuple(model_type.model_validate(value) for value in values)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        _abort(f"invalid {key} artifact {path}: {exc}")


@app.command("report")
def report_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    run_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    judgments_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    json_output: Annotated[Path, typer.Argument()],
    markdown_output: Annotated[Path, typer.Argument()],
    bootstrap_seed: Annotated[int, typer.Option("--bootstrap-seed")],
    adjudications_path: Annotated[
        Path | None, typer.Option("--adjudications", exists=True, readable=True)
    ] = None,
    blind_map_path: Annotated[
        Path | None, typer.Option("--blind-map", exists=True, readable=True)
    ] = None,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=1)] = 1000,
) -> None:
    """Score frozen artifacts and write exploratory JSON plus Markdown reports."""

    manifest = _validated(StudyManifest, manifest_path)
    study_run = _validated(StudyRun, run_path)
    _validate_run_summary(study_run)
    judgments = _records(judgments_path, "judgments", GraderJudgment)
    adjudications = _records(adjudications_path, "adjudications", AdjudicationRecord)
    blind_map = _validated(BlindMap, blind_map_path) if blind_map_path is not None else None
    try:
        report = score_study(
            manifest=manifest,
            study_run=study_run,
            raw_judgments=judgments,
            adjudications=adjudications,
            blind_map=blind_map,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
    except (ValidationError, ValueError) as exc:
        _abort(f"could not score study: {exc}")
    write_report_bundle(report, json_path=json_output, markdown_path=markdown_output)
    typer.echo(f"Wrote exploratory report to {json_output} and {markdown_output}")
