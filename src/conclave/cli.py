"""conclave command-line interface.

Two commands:

* ``conclave ask "<prompt>" --council grok,gemini,claude --mode synthesize``
  Modes: ``synthesize`` (default), ``raw``, ``debate`` (``--rounds N``),
  ``adversarial`` (``--proposer NAME``).
* ``conclave providers`` -- show which providers currently have a key (without
  ever printing key values).
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import load_config
from .council import Council
from .models import CouncilResult, StreamEvent
from .registry import DEFAULT_MODELS, key_present, key_source

app = typer.Typer(
    add_completion=False,
    help="Bring-your-own-keys multi-model council. Fan a prompt to N models.",
)
console = Console()
err_console = Console(stderr=True)


def _result_to_dict(result: CouncilResult) -> dict:
    """Serialize a CouncilResult to a JSON-safe dict (no secrets included)."""
    return result.model_dump(mode="json")


def _answer_panel(ans, *, border: str = "cyan") -> Panel:
    """Build a rich panel for a single ModelAnswer (ok or failed)."""
    if ans.ok:
        usage = ans.usage.total_tokens if ans.usage else 0
        subtitle = f"{ans.model_id} · {ans.latency_s:.2f}s · {usage} tok"
        return Panel(
            ans.answer or "",
            title=f"[bold]{ans.name}[/bold]",
            subtitle=subtitle,
            border_style=border,
        )
    return Panel(
        f"[red]{ans.error}[/red]",
        title=f"[bold]{ans.name}[/bold] (failed)",
        subtitle=ans.model_id,
        border_style="red",
    )


def _print_skipped(result: CouncilResult) -> None:
    """Print the skipped-no-key warning line if any members were skipped."""
    if result.skipped:
        err_console.print(f"[yellow]Skipped (no key): {', '.join(result.skipped)}[/yellow]")


def _print_synthesis(result: CouncilResult, title: str = "SYNTHESIS") -> None:
    """Print the synthesis panel, or a warning if synthesis did not run."""
    if result.synthesis is not None:
        console.print(
            Panel(
                result.synthesis,
                title=f"[bold green]{title}[/bold green] "
                f"({result.synthesizer} · {result.synthesizer_model_id})",
                border_style="green",
            )
        )
    elif result.synthesis_error:
        err_console.print(f"[yellow]No {title.lower()}: {result.synthesis_error}[/yellow]")


def _consensus_line(verdict) -> str:
    """Format the consensus as a single heuristic-labeled line (never authoritative).

    The score is a deterministic heuristic over the model's position clustering
    (``position_cluster_ratio_v1``), not a confidence number, so the rendered line
    flags it as a heuristic by name. A ``None`` score (N<2 responders or no
    positioned members) degrades to ``n/a`` rather than printing a bogus 0.0.
    """
    label = verdict.consensus_label or "n/a"
    if verdict.consensus_score is None:
        score = "n/a"
    else:
        score = f"{verdict.consensus_score:.2f}"
    method = verdict.consensus_method or "n/a"
    # No square brackets here: the panel content is rendered with Rich markup
    # enabled, and a literal "[heuristic: ...]" reads as an (unknown) markup tag
    # and is silently dropped. "heuristic:" prose keeps the meaning without the
    # collision; the score is explicitly framed as a heuristic, never authoritative.
    return f"consensus: {label} ({score}) — heuristic: {method}"


def _conflict_providers(verdict, label: str) -> list[str]:
    """Resolve a position label to the providers holding it, from the verdict.

    A conflict references positions by ``label``; the provider lists live on the
    verdict's ``positions``. Returns the first matching position's providers, or
    an empty list when the label has no clean mapping (then the caller shows the
    label alone rather than inventing a provider list).
    """
    for pos in verdict.positions:
        if pos.label == label:
            return pos.providers
    return []


def _print_verdict_absent_note(result: CouncilResult) -> None:
    """Print a single dim note explaining WHY no verdict was produced (stderr).

    Only fires when the verdict is absent AND the manifest carries a reason
    (open-ended prompt, N<2 responders, or extraction failure). Routed to
    ``err_console`` (stderr) and styled ``[dim]`` so it is purely informational
    and never disrupts an existing stdout assertion (e.g. test_cli.py's
    human-render checks read ``result.output`` which CliRunner mixes, but the note
    is dim and additive — it adds no panel/header those tests assert on).
    """
    if result.verdict is not None:
        return
    manifest = result.manifest
    if manifest is None or not manifest.verdict_absent_reason:
        return
    err_console.print(f"[dim]No verdict: {manifest.verdict_absent_reason}[/dim]")


def _print_verdict(result: CouncilResult) -> None:
    """Print the verdict section (headline, recommendation, consensus, conflicts).

    A no-op when ``result.verdict is None`` (raw mode and verdict-absent runs):
    nothing is rendered so the human output is byte-identical to the pre-verdict
    behavior. When a verdict is present it is rendered as a single green-bordered
    :class:`~rich.panel.Panel`, mirroring :func:`_print_synthesis`, so the verdict
    sits visually beside the synthesis rather than as a parallel renderer.

    Each block degrades gracefully: a ``None`` consensus score shows ``n/a``
    (:func:`_consensus_line`); empty ``conflicts`` / ``minority_reports`` are
    simply omitted; a conflict whose position labels do not map to providers shows
    the labels alone (:func:`_conflict_providers`).
    """
    verdict = result.verdict
    if verdict is None:
        return

    lines: list[str] = [
        f"[bold]{verdict.headline}[/bold]",
        "",
        verdict.recommendation,
        "",
        f"[dim]{_consensus_line(verdict)}[/dim]",
    ]

    if verdict.conflicts:
        lines.append("")
        lines.append("[bold]Conflicts[/bold]")
        for conflict in verdict.conflicts:
            held_by: list[str] = []
            for pos_label in conflict.position_labels:
                providers = _conflict_providers(verdict, pos_label)
                if providers:
                    held_by.append(f"{pos_label} ({', '.join(providers)})")
                else:
                    held_by.append(pos_label)
            tension = " vs ".join(held_by) if held_by else "(positions unmapped)"
            lines.append(f"  • [yellow]{conflict.topic}[/yellow]: {tension}")
            if conflict.summary:
                lines.append(f"    {conflict.summary}")

    if verdict.minority_reports:
        lines.append("")
        lines.append("[bold]Minority reports[/bold]")
        for report in verdict.minority_reports:
            who = ", ".join(report.providers) if report.providers else "unattributed"
            lines.append(f"  • [magenta]{who}[/magenta]: {report.claim}")
            if report.why_it_matters:
                lines.append(f"    [dim]{report.why_it_matters}[/dim]")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold green]VERDICT[/bold green] ({verdict.verdict_type})",
            border_style="green",
        )
    )


def _render_human(result: CouncilResult) -> None:
    """Render raw answers + synthesis + verdict (when present) to the terminal."""
    _print_skipped(result)
    for ans in result.answers:
        console.print(_answer_panel(ans))
    _print_synthesis(result)
    _print_verdict(result)
    _print_verdict_absent_note(result)


def _render_debate(result: CouncilResult) -> None:
    """Render a debate: a panel per member per round, then the final synthesis."""
    _print_skipped(result)
    for rnd in result.rounds:
        console.rule(f"[bold]Round {rnd.round_number}[/bold]")
        for ans in rnd.answers:
            console.print(_answer_panel(ans, border="magenta"))
    if result.converged:
        score = result.convergence_score if result.convergence_score is not None else 0.0
        err_console.print(
            f"[green]Converged after {len(result.rounds)} round(s) "
            f"(score {score:.3f}); stopped early.[/green]"
        )
    _print_synthesis(result, title="FINAL SYNTHESIS")


def _render_adversarial(result: CouncilResult) -> None:
    """Render an adversarial run: proposal, critiques, then the verdict."""
    _print_skipped(result)
    adv = result.adversarial
    if adv is None:
        err_console.print("[yellow]No adversarial result produced.[/yellow]")
        return

    console.rule(f"[bold]Proposal — {adv.proposer}[/bold]")
    console.print(_answer_panel(adv.proposal, border="blue"))

    if adv.critiques:
        console.rule("[bold]Critiques[/bold]")
        for crit in adv.critiques:
            console.print(_answer_panel(crit, border="yellow"))

    if adv.verdict is not None:
        console.print(
            Panel(
                adv.verdict,
                title=f"[bold green]VERDICT[/bold green] ({adv.judge} · {adv.judge_model_id})",
                border_style="green",
            )
        )
    elif adv.verdict_error:
        err_console.print(f"[yellow]No verdict: {adv.verdict_error}[/yellow]")


def _stream_to_terminal(council: Council, prompt: str, *, synthesize: bool) -> CouncilResult:
    """Render a live token stream to the terminal and return the final result.

    Drives :meth:`Council.stream_sync`, printing each member's (and the
    synthesizer's) tokens inline as they arrive under a header. A failed member
    is shown in red at its ``member_done`` event. The returned
    :class:`CouncilResult` is the same structure the buffered path produces, so
    the caller applies the usual exit-code contract.

    Headers are tracked per source so a header prints exactly once -- on the
    first delta for a streaming source, or at the done event for a source that
    streamed no live tokens (e.g. a cache hit emits a single delta; a failed
    member emits none).
    """
    # Track which sources have had a header printed so we open each section once.
    started: set[str] = set()

    def _ensure_header(label: str, key: str, style: str) -> None:
        if key not in started:
            started.add(key)
            console.print(f"\n[bold {style}]{label}[/bold {style}]")

    def on_event(event: StreamEvent) -> None:
        if event.type == "member_delta":
            _ensure_header(f"{event.name} ({event.model_id})", f"m:{event.name}", "cyan")
            console.print(event.text or "", end="", soft_wrap=True, highlight=False)
        elif event.type == "member_done":
            ans = event.answer
            if ans is not None and not ans.ok:
                _ensure_header(f"{event.name} (failed)", f"m:{event.name}", "red")
                console.print(f"[red]{ans.error}[/red]")
            else:
                # Close the streamed block with a newline so the next header is clean.
                console.print()
        elif event.type == "synthesis_delta":
            _ensure_header(f"SYNTHESIS ({event.name} · {event.model_id})", "synthesis", "green")
            console.print(event.text or "", end="", soft_wrap=True, highlight=False)
        elif event.type == "synthesis_done":
            ans = event.answer
            if ans is not None and not ans.ok:
                err_console.print(f"\n[yellow]No synthesis: {ans.error}[/yellow]")
            else:
                console.print()

    result = council.stream_sync(prompt, on_event, synthesize=synthesize)
    _print_skipped(result)
    # Surface a synthesis short-circuit reason that produced no synthesis_done
    # event (e.g. synthesizer had no key, or no usable answers to synthesize).
    if synthesize and result.synthesis is None and result.synthesis_error:
        err_console.print(f"[yellow]No synthesis: {result.synthesis_error}[/yellow]")
    return result


# Mode name -> human renderer. JSON output bypasses this via model_dump.
_RENDERERS = {
    "synthesize": _render_human,
    "raw": _render_human,
    "debate": _render_debate,
    "adversarial": _render_adversarial,
}


_VALID_MODES = {"synthesize", "raw", "debate", "adversarial"}

# Threshold used when --converge is passed without an explicit --converge-threshold.
# High by design: an early stop should require answers that are nearly stable
# round-over-round, so the default is conservative and rarely fires spuriously.
_DEFAULT_CONVERGE_THRESHOLD = 0.95


def _resolve_converge_threshold(
    converge: bool | None,
    converge_threshold: float | None,
    config_threshold: float | None,
) -> float | None:
    """Resolve the CLI convergence flags + config into a single threshold.

    Returns the threshold to run with, or ``None`` for early-stop off. Precedence,
    mirroring how ``--cache/--no-cache`` overrides config per invocation:

    * an explicit ``--converge-threshold`` value wins (and implies on);
    * else ``--no-converge`` forces off regardless of config;
    * else ``--converge`` turns on at the default threshold;
    * else (both flags unset) defer to the config value.

    Resolution happens here (not in the council) so ``--no-converge`` can force
    off even when config enables it; the resulting concrete threshold is then
    passed to :meth:`Council.debate_sync`.
    """
    if converge_threshold is not None:
        return converge_threshold
    if converge is False:
        return None
    if converge is True:
        return _DEFAULT_CONVERGE_THRESHOLD
    return config_threshold


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The prompt to send to the council."),
    council: str = typer.Option(
        "default",
        "--council",
        "-c",
        help="Named council or comma-separated friendly names (e.g. grok,claude).",
    ),
    mode: str = typer.Option(
        "synthesize",
        "--mode",
        "-m",
        help="Run mode: synthesize | raw | debate | adversarial.",
    ),
    synthesizer: str | None = typer.Option(
        None, "--synthesizer", "-s", help="Override the synthesizer/judge model name."
    ),
    rounds: int = typer.Option(
        2, "--rounds", "-r", help="Maximum number of debate rounds (debate mode only).", min=1
    ),
    converge: bool | None = typer.Option(
        None,
        "--converge/--no-converge",
        help=(
            "Enable/disable debate early-stop on answer convergence (debate mode "
            "only; off by default, defers to config when unset). --converge with "
            "no --converge-threshold uses a default threshold of "
            f"{_DEFAULT_CONVERGE_THRESHOLD}; --no-converge forces it off."
        ),
    ),
    converge_threshold: float | None = typer.Option(
        None,
        "--converge-threshold",
        help=(
            "Debate early-stop threshold in [0.0, 1.0] (debate mode only). When "
            "set, the debate stops once round-over-round answer stability reaches "
            "this value. Implies --converge. Overrides config; unset defers to it."
        ),
        min=0.0,
        max=1.0,
    ),
    proposer: str | None = typer.Option(
        None,
        "--proposer",
        "-p",
        help="Proposer model name (adversarial mode; defaults to first member).",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the full result as JSON instead of panels."
    ),
    cache: bool | None = typer.Option(
        None,
        "--cache/--no-cache",
        help=(
            "Use the on-disk result cache (off by default; defers to config when "
            "unset). On a hit an identical prior run is returned without calling "
            "the providers. The cache never stores API keys."
        ),
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help=(
            "Stream member (and synthesizer) tokens live to the terminal as they "
            "arrive (synthesize/raw modes only). Ignored with --json. On a cache "
            "hit the cached text is rendered in one shot (no live token stream)."
        ),
    ),
) -> None:
    """Fan PROMPT out to a council and synthesize, debate, or adversarially review.

    Exit codes:

    * 0 -- the run produced at least one usable member answer.
    * 1 -- the run completed but produced zero usable member answers (e.g. no
      council member had an API key, or every member failed). Under ``--json``
      the full JSON result is still emitted to stdout first, so a script can both
      parse the payload and detect the failure via the non-zero exit code.
    * 2 -- a usage/config error (unknown mode, or no members resolved).
    """
    mode_lower = mode.lower()
    if mode_lower not in _VALID_MODES:
        err_console.print(
            f"[red]Unknown mode '{mode}'. Choose one of: {', '.join(sorted(_VALID_MODES))}.[/red]"
        )
        raise typer.Exit(code=2)

    if stream and mode_lower not in ("synthesize", "raw"):
        err_console.print(
            f"[red]--stream is only supported for synthesize/raw modes, not '{mode_lower}'.[/red]"
        )
        raise typer.Exit(code=2)

    cfg = load_config()
    members = cfg.resolve_council(council)
    if not members:
        err_console.print(f"[red]No council members resolved from '{council}'.[/red]")
        raise typer.Exit(code=2)

    c = Council(models=members, synthesizer=synthesizer, config=cfg, cache=cache)

    # Streaming path: live token output (synthesize/raw only, not with --json).
    # It produces the same final CouncilResult, so the exit-code contract below
    # applies identically.
    if stream and not as_json:
        result = _stream_to_terminal(c, prompt, synthesize=(mode_lower == "synthesize"))
        if not result.successful_answers:
            err_console.print(
                "[red]No usable council answers. Run 'conclave providers' to check keys.[/red]"
            )
            raise typer.Exit(code=1)
        return

    if mode_lower == "debate":
        threshold = _resolve_converge_threshold(
            converge, converge_threshold, cfg.converge_threshold
        )
        result = c.debate_sync(prompt, rounds=rounds, converge_threshold=threshold)
    elif mode_lower == "adversarial":
        result = c.adversarial_sync(prompt, proposer=proposer)
    else:
        result = c.ask_sync(prompt, synthesize=(mode_lower == "synthesize"))

    # A run that produced no usable member answers is a failure for scripting
    # purposes regardless of output format. We compute this once and apply the
    # same exit-code contract to both the JSON and human paths.
    no_usable_answers = not result.successful_answers

    if as_json:
        # Always emit valid JSON to stdout so a consumer can parse the payload,
        # then signal failure via the exit code if nothing usable came back.
        console.print_json(json.dumps(_result_to_dict(result)))
        if no_usable_answers:
            raise typer.Exit(code=1)
        return

    if no_usable_answers:
        err_console.print(
            "[red]No usable council answers. Run 'conclave providers' to check keys.[/red]"
        )
        raise typer.Exit(code=1)

    _RENDERERS[result.mode](result)


@app.command()
def providers() -> None:
    """List known providers and whether a key is present (no values shown)."""
    cfg = load_config()
    table = Table(title="conclave providers")
    table.add_column("name", style="bold")
    table.add_column("model id")
    table.add_column("key", justify="center")
    table.add_column("env var")

    for name, model_id in sorted(cfg.models.items()):
        present = key_present(model_id)
        mark = "[green]✓[/green]" if present else "[red]✗[/red]"
        source = key_source(model_id) or "[dim]-[/dim]"
        table.add_row(name, model_id, mark, source)

    console.print(table)
    console.print(f"[dim]synthesizer default: {cfg.synthesizer} · conclave {__version__}[/dim]")


def _builtin_default_note() -> str:
    """Return a short note listing built-in providers (used in --help footer)."""
    return ", ".join(DEFAULT_MODELS.keys())


if __name__ == "__main__":
    app()
