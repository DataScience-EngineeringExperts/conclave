"""Deterministic JSON and Markdown rendering for exploratory eval reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .scoring import StudyScoreReport


def render_markdown_report(report: StudyScoreReport) -> str:
    """Render a compact human report without promoting exploratory evidence."""

    lines = [
        "# Conclave Elite Evaluation — SYNTHETIC / EXPLORATORY",
        "",
        f"Study: `{report.study_id}`",
        "",
        "**Decision status: not yet eligible.** Go / redesign / kill decisions require "
        "confirmatory held-out data.",
        "",
        "## Outcome distribution",
        "",
        "| Outcome | Planned runs |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {outcome} | {count} |"
        for outcome, count in sorted(report.run_outcome_distribution.items())
    )
    lines.extend(
        [
            "",
            "## Failure-inclusive critical-error-free rates",
            "",
            "| Condition | Error-free / planned | Rate | 95% Wilson CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in report.condition_metrics:
        interval = metric.wilson_95_interval
        lines.append(
            f"| {metric.condition_id} | {metric.critical_error_free_count} / "
            f"{metric.planned_runs} | {metric.critical_error_free_rate:.3f} | "
            f"[{interval.lower:.3f}, {interval.upper:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap comparisons",
            "",
            "Elite-minus-baseline task-paired differences; intervals are exploratory.",
            "",
            "| Baseline | Difference | 95% paired bootstrap CI | Tasks |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {item.baseline_condition_id} | {item.estimate:.3f} | "
        f"[{item.lower:.3f}, {item.upper:.3f}] | {item.task_count} |"
        for item in report.paired_differences
    )
    reliability = report.reliability
    kappa = "n/a" if reliability.cohen_kappa is None else f"{reliability.cohen_kappa:.3f}"
    lines.extend(
        [
            "",
            "## Grader reliability",
            "",
            f"- Cohen kappa: {kappa} ({reliability.paired_judgments} paired judgments)",
            f"- Adjudication rate: {reliability.adjudication_rate:.3f} "
            f"({reliability.adjudicated_disagreements}/{reliability.disagreements} disagreements)",
            "",
            "## Raw grader provenance",
            "",
            f"- Raw judgments preserved: {len(report.raw_judgments)}",
            f"- Separate adjudications preserved: {len(report.adjudications)}",
            "- Complete atomic records and resolved-cell provenance are available in the JSON report.",
            "",
            "## Go / redesign / kill",
            "",
            "- Go: not yet eligible",
            "- Redesign: not yet eligible",
            "- Kill: not yet eligible",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def write_report_bundle(
    report: StudyScoreReport, *, json_path: str | Path, markdown_path: str | Path
) -> None:
    """Atomically write matching machine-readable and human-readable reports."""

    _atomic_write(Path(json_path), report.model_dump_json(indent=2) + "\n")
    _atomic_write(Path(markdown_path), render_markdown_report(report))
