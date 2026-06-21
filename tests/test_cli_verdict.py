"""Tests for CAC-06-CLI: the CLI renders the council verdict (human + --json).

These exercise ``conclave.cli.ask`` through Typer's ``CliRunner`` (no real keys,
no network), reusing the exact harness from ``test_cli.py`` (``CliRunner``,
``_config()``, ``patch_cli_config``, ``patch_call_model``, ``monkeypatch.setenv``)
and the dual-seam verdict pattern from ``test_council_verdict.py`` (a handler that
branches on ``_is_verdict_call(messages)`` and returns extraction JSON for that
call, prose otherwise). Two concerns are pinned here:

* **Human render (verdict section).** A synthesize run that produces a real
  verdict renders the headline, recommendation, consensus label+score (labeled as
  a heuristic), at least one conflict, and a minority report -- in the existing
  rich panel style. A run whose verdict degrades to ``None`` (prose-only handler)
  renders exactly as before (member panels, exit 0, no crash).
* **``--json`` payload.** The existing ``model_dump(mode="json")`` path carries
  the full v2 result, so ``payload["verdict"]`` and ``payload["manifest"]`` are
  present, the verdict carries ``headline``/``recommendation``/``consensus_score``/
  ``consensus_label``, and the hoisted consensus mirror is present on the result.
* **Secret-safety (both paths).** Provider keys set to obvious fake values never
  appear in human OR ``--json`` output, and the serialized JSON carries none of
  the manifest's ``_FORBIDDEN_SUBSTRINGS`` given clean inputs.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from conclave import cli
from conclave.config import ConclaveConfig
from conclave.manifest import _FORBIDDEN_SUBSTRINGS

runner = CliRunner()

# Fake provider key values used by the secret-safety proof. Chosen WITHOUT any
# ``_FORBIDDEN_SUBSTRINGS`` token (no "sk-", "bearer", "authorization", "api_key",
# "x-api-key") so the no-forbidden-substring assertion is meaningful and not
# self-defeating: if one of these distinctive values leaked it would be a real
# secret exposure, and separately the serialized JSON must carry no forbidden
# pattern at all given these clean inputs. The values are still obviously secret
# (a leak would be unmistakable) without tripping the substring scan themselves.
_FAKE_KEYS = {
    "XAI_API_KEY": "FAKE-GROK-SECRET-zzz111",
    "GEMINI_API_KEY": "FAKE-GEMINI-SECRET-zzz222",
    "ANTHROPIC_API_KEY": "FAKE-CLAUDE-SECRET-zzz333",
    "PERPLEXITY_API_KEY": "FAKE-PPLX-SECRET-zzz444",
}


def _config() -> ConclaveConfig:
    """A deterministic config independent of any on-disk ~/.conclave file."""
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": "anthropic/claude-sonnet-4-6",
            "perplexity": "perplexity/sonar-pro",
        },
        councils={"default": ["grok", "gemini", "claude", "perplexity"]},
        synthesizer="claude",
    )


@pytest.fixture
def patch_cli_config(monkeypatch):
    """Make ``conclave.cli.load_config`` return the deterministic test config."""
    monkeypatch.setattr(cli, "load_config", _config)


def _set_fake_keys(monkeypatch) -> None:
    """Set every provider key to an obvious fake value (no forbidden substrings)."""
    for var, val in _FAKE_KEYS.items():
        monkeypatch.setenv(var, val)


def _is_verdict_call(messages) -> bool:
    """True when ``messages`` is the verdict-extraction call (vs a member/prose call)."""
    return bool(messages) and messages[0].get("content", "").startswith(
        "You are the verdict extractor"
    )


# Distinctive strings the human-render assertions look for. Kept here so the
# extraction JSON and the assertions share one source of truth.
_HEADLINE = "Ship behind a flag."
_RECOMMENDATION = "Adopt with a feature flag and a rollback plan."
_CONFLICT_TOPIC = "rollout speed"
_MINORITY_CLAIM = "the migration window is too short for a hard cutover"


def _split_extraction_json() -> str:
    """Build extraction JSON with >=2 positions, a conflict, and a minority report.

    Two members hold ``"yes"`` and one holds ``"no"``: positions are split across
    two labels (so the conflict path renders and the minority report has providers
    to attribute), and the deterministic 2-of-3 vote ratio (0.6667) buckets to
    ``"majority"`` per the agreement table — a real, non-unanimous consensus.
    ``provider_votes[*].provider`` matches the member names so the engine's
    per-member sequence resolves.
    """
    return json.dumps(
        {
            "verdict_applies": True,
            "verdict_type": "decision",
            "headline": _HEADLINE,
            "recommendation": _RECOMMENDATION,
            "positions": [
                {
                    "label": "yes",
                    "summary": "Ship now behind a flag; iterate in prod.",
                    "providers": ["grok", "gemini"],
                    "evidence_answer_ids": ["grok-1", "gemini-1"],
                },
                {
                    "label": "no",
                    "summary": "Hold until the migration window widens.",
                    "providers": ["perplexity"],
                    "evidence_answer_ids": ["perplexity-1"],
                },
            ],
            "provider_votes": [
                {"provider": "grok", "position_label": "yes"},
                {"provider": "gemini", "position_label": "yes"},
                {"provider": "perplexity", "position_label": "no"},
            ],
            "conflicts": [
                {
                    "topic": _CONFLICT_TOPIC,
                    "position_labels": ["yes", "no"],
                    "summary": "Disagreement on whether to cut over now or wait.",
                    "consensus_score": None,
                }
            ],
            "minority_reports": [
                {
                    "providers": ["perplexity"],
                    "claim": _MINORITY_CLAIM,
                    "evidence_answer_ids": ["perplexity-1"],
                    "why_it_matters": "a rushed cutover risks data loss",
                }
            ],
            "caveats": [],
            "dissent_summary": None,
        }
    )


def _verdict_handler(extraction_json: str):
    """A handler that returns extraction JSON for the verdict call, prose otherwise."""

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        if _is_verdict_call(messages):
            return make_response(extraction_json)
        return make_response(f"member answer from {model}")

    return handler


# --------------------------------------------------------------------------- #
# 1. Human render WITH a verdict.
# --------------------------------------------------------------------------- #
def test_human_render_shows_verdict_section(monkeypatch, patch_cli_config, patch_call_model):
    """A synthesize run with a real verdict renders the verdict panel + its content."""
    _set_fake_keys(monkeypatch)
    patch_call_model(_verdict_handler(_split_extraction_json()))

    # A wide console so Rich does not ellipsize the verdict panel's content; the
    # module-level console fixes its width at import, so swap in a wide one (same
    # technique test_cli.py uses for the providers table).
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(width=200))

    result = runner.invoke(
        cli.app, ["ask", "Should we ship?", "--council", "grok,gemini,perplexity"]
    )
    assert result.exit_code == 0

    out = result.output
    # Member panels + synthesis still render (additive section, nothing removed).
    assert "grok" in out
    assert "SYNTHESIS" in out
    # Verdict header + the heuristic-labeled consensus line.
    assert "VERDICT" in out
    assert "consensus" in out
    # The score's deterministic label appears alongside the word consensus. A
    # 2-of-3 vote ratio (0.6667) buckets to "majority" per the agreement table
    # (0.5 < score < 0.75), and the rounded score renders on the same line.
    assert "majority" in out
    assert "0.67" in out
    # The verdict's headline and recommendation render.
    assert _HEADLINE in out
    assert _RECOMMENDATION in out
    # The consensus is clearly flagged as a heuristic (not an authoritative score).
    assert "heuristic" in out.lower()
    # At least one conflict's content renders.
    assert _CONFLICT_TOPIC in out
    # The minority report renders.
    assert _MINORITY_CLAIM in out


# --------------------------------------------------------------------------- #
# 2. --json carries the full verdict + manifest.
# --------------------------------------------------------------------------- #
def test_json_payload_carries_verdict_and_manifest(monkeypatch, patch_cli_config, patch_call_model):
    """--json emits the v2 result: verdict (with consensus fields) + manifest present."""
    _set_fake_keys(monkeypatch)
    patch_call_model(_verdict_handler(_split_extraction_json()))

    result = runner.invoke(
        cli.app,
        ["ask", "Should we ship?", "--council", "grok,gemini,perplexity", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    # Verdict block present and well-formed.
    verdict = payload["verdict"]
    assert verdict is not None
    assert verdict["headline"] == _HEADLINE
    assert verdict["recommendation"] == _RECOMMENDATION
    assert "consensus_score" in verdict
    assert verdict["consensus_label"] is not None
    # The conflict + minority report ride along in the verdict block.
    assert any(c["topic"] == _CONFLICT_TOPIC for c in verdict["conflicts"])
    assert any(m["claim"] == _MINORITY_CLAIM for m in verdict["minority_reports"])

    # Manifest present and a dict (the auditable receipt).
    assert payload["manifest"] is not None
    assert isinstance(payload["manifest"], dict)
    assert payload["manifest"]["mode"] == "synthesize"

    # The hoisted consensus mirror is present on the top-level result.
    assert payload["consensus_score"] == verdict["consensus_score"]
    assert payload["consensus_label"] == verdict["consensus_label"]
    assert payload["consensus_method"] == verdict["consensus_method"]


# --------------------------------------------------------------------------- #
# 3. No-secret proof -- human path.
# --------------------------------------------------------------------------- #
def test_human_output_leaks_no_keys(monkeypatch, patch_cli_config, patch_call_model):
    """No fake key VALUE appears anywhere in the human-rendered verdict output."""
    _set_fake_keys(monkeypatch)
    patch_call_model(_verdict_handler(_split_extraction_json()))

    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(width=200))

    result = runner.invoke(
        cli.app, ["ask", "Should we ship?", "--council", "grok,gemini,perplexity"]
    )
    assert result.exit_code == 0
    for val in _FAKE_KEYS.values():
        assert val not in result.output


# --------------------------------------------------------------------------- #
# 4. No-secret proof -- --json path (key values + forbidden substrings).
# --------------------------------------------------------------------------- #
def test_json_output_leaks_no_keys_or_forbidden_substrings(
    monkeypatch, patch_cli_config, patch_call_model
):
    """--json carries no fake key VALUE and (given clean inputs) no forbidden substring."""
    _set_fake_keys(monkeypatch)
    patch_call_model(_verdict_handler(_split_extraction_json()))

    result = runner.invoke(
        cli.app,
        ["ask", "Should we ship?", "--council", "grok,gemini,perplexity", "--json"],
    )
    assert result.exit_code == 0

    # 4a. No fake key VALUE leaks into the serialized JSON.
    for val in _FAKE_KEYS.values():
        assert val not in result.stdout

    # 4b. Given clean inputs (no secret-shaped content fed in), the serialized
    # JSON carries none of the manifest's forbidden substrings.
    payload = json.loads(result.stdout)
    serialized = json.dumps(payload).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in serialized


# --------------------------------------------------------------------------- #
# 5. No-verdict run renders cleanly (prose-only handler -> verdict degrades None).
# --------------------------------------------------------------------------- #
def test_no_verdict_run_renders_cleanly(monkeypatch, patch_cli_config, patch_call_model):
    """A prose-only handler degrades the verdict to None; human render is unchanged."""
    _set_fake_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        # Prose for EVERY call (incl. the verdict call) -> extraction parse-fails
        # -> result.verdict is None -> no verdict section is rendered.
        return make_response(f"prose answer from {model}")

    patch_call_model(handler)
    result = runner.invoke(
        cli.app, ["ask", "Write me a poem.", "--council", "grok,gemini,perplexity"]
    )
    assert result.exit_code == 0
    out = result.output
    # Member panels + synthesis still present; no crash.
    assert "grok" in out
    assert "gemini" in out
    assert "SYNTHESIS" in out
    # No verdict header rendered when the verdict is absent.
    assert "VERDICT" not in out
    # No fake key value leaks on the degraded path either.
    for val in _FAKE_KEYS.values():
        assert val not in out
