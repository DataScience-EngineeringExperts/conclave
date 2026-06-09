"""Tests for the CLI exit-code contract and HTTP-client lifecycle wiring.

These exercise ``conclave.cli.ask`` through Typer's ``CliRunner`` (no real keys,
no network). Two concerns are pinned here:

* **Exit-code contract (#17).** A run that produces zero *usable* member answers
  exits non-zero (code 1) on both the human and ``--json`` paths, and under
  ``--json`` the full JSON payload is still emitted to stdout so a script can
  parse the result *and* detect the failure via the exit code. A run with at
  least one usable answer exits 0.
* **Pooled-client lifecycle (#20).** The synchronous council wrappers close the
  shared httpx client when the run completes, so ``transport.aclose()`` is
  actually invoked and no client leaks past the CLI command.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from conclave import cli
from conclave.config import ConclaveConfig

runner = CliRunner()


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


def test_no_members_human_exits_one(clear_keys, patch_cli_config):
    """Plain (human) path: zero usable answers -> exit code 1."""
    result = runner.invoke(cli.app, ["ask", "hello"])
    assert result.exit_code == 1
    assert "No usable council answers" in result.output


def test_no_members_json_exits_one_but_emits_json(clear_keys, patch_cli_config):
    """JSON path: zero usable answers -> exit 1, yet valid JSON still on stdout."""
    result = runner.invoke(cli.app, ["ask", "hello", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["answers"] == []
    assert payload["prompt"] == "hello"


def test_all_members_failed_exits_one(monkeypatch, patch_cli_config, patch_call_model):
    """Keys present but every member errors -> zero usable answers -> exit 1."""
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    def handler(model, messages, **kwargs):
        raise RuntimeError("provider down")

    patch_call_model(handler)
    result = runner.invoke(cli.app, ["ask", "hello", "--council", "grok,gemini", "--mode", "raw"])
    assert result.exit_code == 1


def test_successful_run_exits_zero(monkeypatch, patch_cli_config, patch_call_model):
    """At least one usable answer -> exit 0, JSON payload carries the answers."""
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        return make_response(f"answer from {model}")

    patch_call_model(handler)
    result = runner.invoke(
        cli.app,
        ["ask", "hello", "--council", "grok,gemini", "--mode", "raw", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["answers"]) == 2
    assert all(a["answer"].startswith("answer from") for a in payload["answers"])


def test_unknown_mode_exits_two(patch_cli_config):
    """Usage error (bad mode) keeps its distinct exit code 2."""
    result = runner.invoke(cli.app, ["ask", "hello", "--mode", "bogus"])
    assert result.exit_code == 2


def test_unresolved_council_exits_two(patch_cli_config):
    """Usage error (empty council selector resolves to zero members) -> exit 2."""
    result = runner.invoke(cli.app, ["ask", "hello", "--council", " , "])
    assert result.exit_code == 2
    assert "No council members resolved" in result.output


def test_sync_run_closes_pooled_client(monkeypatch, patch_call_model):
    """The sync wrapper invokes transport.aclose() so the client never leaks."""
    import conclave.council as council_mod
    from conclave import Council

    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")

    calls = {"aclose": 0}

    async def fake_aclose():
        calls["aclose"] += 1

    monkeypatch.setattr(council_mod.transport, "aclose", fake_aclose)

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        return make_response("ok")

    patch_call_model(handler)
    Council(models=["grok"], synthesizer="grok", config=_config()).ask_sync(
        "hello", synthesize=False
    )
    assert calls["aclose"] == 1


def test_close_sync_invokes_aclose(monkeypatch):
    """Council.close_sync drives transport.aclose without re-closing recursively."""
    import conclave.council as council_mod
    from conclave import Council

    calls = {"aclose": 0}

    async def fake_aclose():
        calls["aclose"] += 1

    monkeypatch.setattr(council_mod.transport, "aclose", fake_aclose)
    Council(models=["grok"], config=_config()).close_sync()
    # close_sync passes close_client=False, so aclose fires exactly once (the body),
    # not twice (body + finally).
    assert calls["aclose"] == 1


async def test_aclose_really_closes_real_client():
    """End-to-end: a real pooled client gets created then closed by aclose()."""
    from conclave import transport

    client = transport._get_client()
    assert not client.is_closed
    await transport.aclose()
    assert client.is_closed


# --------------------------------------------------------------------------- #
# Human (rich) renderers + the providers command. These cover the panel/table
# rendering paths that the JSON exit-code tests bypass via model_dump.
# --------------------------------------------------------------------------- #


def _all_keys(monkeypatch) -> None:
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")


def test_synthesize_human_render_prints_synthesis(monkeypatch, patch_cli_config, patch_call_model):
    """The default synthesize mode renders member panels and a SYNTHESIS panel."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        # Every call (members and the synthesizer) returns a usable answer, so the
        # synthesis runs and its panel is rendered.
        return make_response(f"ans-{model}")

    patch_call_model(handler)
    result = runner.invoke(cli.app, ["ask", "hello", "--council", "grok,gemini"])
    assert result.exit_code == 0
    # Member panel titles and the synthesis header are present in the rendered output.
    assert "grok" in result.output
    assert "SYNTHESIS" in result.output


def test_raw_human_render_prints_member_panels(monkeypatch, patch_cli_config, patch_call_model):
    """Raw mode renders each member's answer panel and no synthesis header."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        return make_response(f"raw-{model}")

    patch_call_model(handler)
    result = runner.invoke(cli.app, ["ask", "hello", "--council", "grok,gemini", "--mode", "raw"])
    assert result.exit_code == 0
    assert "grok" in result.output
    assert "gemini" in result.output


def test_debate_human_render_prints_rounds(monkeypatch, patch_cli_config, patch_call_model):
    """Debate mode renders a Round rule per round plus the FINAL SYNTHESIS panel."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        return make_response(f"debate-{model}")

    patch_call_model(handler)
    result = runner.invoke(
        cli.app,
        ["ask", "hello", "--council", "grok,gemini", "--mode", "debate", "--rounds", "2"],
    )
    assert result.exit_code == 0
    assert "Round" in result.output
    assert "SYNTHESIS" in result.output


def test_adversarial_human_render_prints_proposal_and_verdict(
    monkeypatch, patch_cli_config, patch_call_model
):
    """Adversarial mode renders the proposal, critiques, and a VERDICT panel."""
    _all_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        return make_response(f"adv-{model}")

    patch_call_model(handler)
    result = runner.invoke(
        cli.app,
        ["ask", "hello", "--council", "grok,gemini", "--mode", "adversarial"],
    )
    assert result.exit_code == 0
    assert "Proposal" in result.output
    assert "VERDICT" in result.output


def test_skipped_members_warning_is_printed(monkeypatch, patch_cli_config, patch_call_model):
    """A member with no key is skipped and surfaced via the yellow warning line."""
    # Only grok has a key; gemini will be skipped for lack of GEMINI_API_KEY.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "dummy-key")

    def handler(model, messages, **kwargs):
        from tests.conftest import make_response

        return make_response(f"ans-{model}")

    patch_call_model(handler)
    result = runner.invoke(cli.app, ["ask", "hello", "--council", "grok,gemini", "--mode", "raw"])
    assert result.exit_code == 0
    assert "Skipped (no key)" in result.output
    assert "gemini" in result.output


def test_cache_flag_serves_second_run_from_cache(monkeypatch, patch_cli_config, tmp_path):
    """`--cache` makes a second identical CLI run a hit (providers not re-called)."""
    import conclave.council as council_mod
    from conclave.models import ModelAnswer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _all_keys(monkeypatch)

    counter = {"n": 0}

    async def fake_call_model(name, model_id, messages, *, temperature=0.7, timeout=120.0):
        counter["n"] += 1
        return ModelAnswer(name=name, model_id=model_id, answer=f"ans-{model_id}")

    monkeypatch.setattr(council_mod, "call_model", fake_call_model)

    args = ["ask", "what is 2+2?", "--council", "grok", "--mode", "raw", "--cache", "--json"]
    first = runner.invoke(cli.app, args)
    assert first.exit_code == 0
    n_after_first = counter["n"]
    assert n_after_first > 0
    assert json.loads(first.stdout)["cached"] is False

    second = runner.invoke(cli.app, args)
    assert second.exit_code == 0
    assert counter["n"] == n_after_first  # no new provider calls -> served from cache
    assert json.loads(second.stdout)["cached"] is True


def test_no_cache_flag_runs_live_each_time(monkeypatch, patch_cli_config, tmp_path):
    """`--no-cache` forces a live run even if config enabled caching."""
    import conclave.council as council_mod
    from conclave.models import ModelAnswer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _all_keys(monkeypatch)

    counter = {"n": 0}

    async def fake_call_model(name, model_id, messages, *, temperature=0.7, timeout=120.0):
        counter["n"] += 1
        return ModelAnswer(name=name, model_id=model_id, answer="ans")

    monkeypatch.setattr(council_mod, "call_model", fake_call_model)

    args = ["ask", "p", "--council", "grok", "--mode", "raw", "--no-cache", "--json"]
    runner.invoke(cli.app, args)
    n = counter["n"]
    runner.invoke(cli.app, args)
    assert counter["n"] == n * 2  # ran live both times


def test_providers_command_lists_keys_without_values(monkeypatch, patch_cli_config):
    """`conclave providers` prints a table marking present/absent keys, no values."""
    monkeypatch.setenv("XAI_API_KEY", "super-secret-value")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["providers"])
    assert result.exit_code == 0
    assert "conclave providers" in result.output
    # Provider names and the env-var column appear; the secret VALUE never does.
    assert "grok" in result.output
    assert "XAI_API_KEY" in result.output
    assert "super-secret-value" not in result.output
