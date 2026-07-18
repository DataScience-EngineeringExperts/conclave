"""Tests for the optional result cache (issue #6 / PDD §9 #4).

All tests run offline. The cache is redirected to a per-test ``tmp_path`` via the
``XDG_CACHE_HOME`` env var so the real ``~/.cache`` is never touched and each test
starts empty.

Pinned behaviors:

* **Off by default** -- two identical runs both execute and nothing is written.
* **On -> miss then hit** -- the second identical run does NOT call the providers
  (asserted by a call counter on the patched call path) and is flagged ``cached``.
* **Key sensitivity** -- changing prompt / council / mode / model id misses.
* **Security** -- no API key VALUE appears in the cache key or the persisted
  on-disk payload, even with a fake key env var set.
* **Graceful degradation** -- a corrupt entry is a miss, the run completes, no
  crash, and the corrupt entry is overwritten with a valid one.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import conclave.council as council_mod
from conclave import Council
from conclave import cache as cache_mod
from conclave.config import ConclaveConfig, CustomEndpoint
from conclave.models import ModelAnswer
from tests.conftest import make_response


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    """Redirect the cache dir into tmp_path and return it."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path / "conclave"


def _config(cache: bool = False) -> ConclaveConfig:
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
        cache=cache,
    )


def _set_keys(monkeypatch) -> None:
    """Set every provider key to a dummy non-empty value."""
    for var in ("XAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"):
        monkeypatch.setenv(var, "dummy-key")


@pytest.fixture
def counting_call_model(monkeypatch):
    """Patch ``conclave.council.call_model`` with a call-counting fake.

    Returns the mutable counter dict ``{"n": int}``. Every member + synthesizer
    call increments it, so a cache HIT is provable by the counter not advancing.
    """
    counter = {"n": 0}

    async def fake_call_model(
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None
    ):
        counter["n"] += 1
        await asyncio.sleep(0)
        # Synthesizer call is the 2-message (system+user) one.
        text = "MERGED" if len(messages) == 2 else f"answer from {model_id}"
        return ModelAnswer(name=name, model_id=model_id, answer=text)

    monkeypatch.setattr(council_mod, "call_model", fake_call_model)
    return counter


# --------------------------------------------------------------------------- #
# Off by default
# --------------------------------------------------------------------------- #


async def test_cache_off_by_default_no_file_no_hit(monkeypatch, counting_call_model, cache_home):
    """Cache OFF (default): two identical runs both execute; nothing is written."""
    _set_keys(monkeypatch)
    council = Council(models=["grok", "perplexity"], synthesizer="claude", config=_config())
    assert council.cache_enabled is False

    r1 = await council.ask("what is 2+2?")
    after_first = counting_call_model["n"]
    r2 = await council.ask("what is 2+2?")

    assert r1.cached is False
    assert r2.cached is False
    # Second run executed again -> counter advanced.
    assert counting_call_model["n"] == after_first * 2
    # No cache artifacts written at all.
    assert not cache_home.exists() or not list(cache_home.glob("*.json"))


# --------------------------------------------------------------------------- #
# On: miss then hit
# --------------------------------------------------------------------------- #


async def test_cache_on_first_miss_then_hit(monkeypatch, counting_call_model, cache_home):
    """Cache ON: first run populates; second identical run is a hit, no provider calls."""
    _set_keys(monkeypatch)
    council = Council(
        models=["grok", "perplexity"], synthesizer="claude", config=_config(), cache=True
    )

    r1 = await council.ask("what is 2+2?")
    calls_after_first = counting_call_model["n"]
    assert calls_after_first > 0
    assert r1.cached is False
    # Exactly one entry written.
    entries = list(cache_home.glob("*.json"))
    assert len(entries) == 1

    r2 = await council.ask("what is 2+2?")
    # The hit must NOT call the providers again.
    assert counting_call_model["n"] == calls_after_first
    assert r2.cached is True
    # Same content served.
    assert r2.synthesis == r1.synthesis
    assert [a.answer for a in r2.answers] == [a.answer for a in r1.answers]


async def test_cache_on_via_config_flag(monkeypatch, counting_call_model, cache_home):
    """Cache enabled through config.cache (no explicit Council arg) also hits."""
    _set_keys(monkeypatch)
    council = Council(models=["grok"], synthesizer="claude", config=_config(cache=True))
    assert council.cache_enabled is True

    await council.ask("hello")
    n = counting_call_model["n"]
    r2 = await council.ask("hello")
    assert counting_call_model["n"] == n
    assert r2.cached is True


async def test_explicit_no_cache_overrides_config(monkeypatch, counting_call_model, cache_home):
    """An explicit cache=False overrides config.cache=True (the --no-cache path)."""
    _set_keys(monkeypatch)
    council = Council(models=["grok"], config=_config(cache=True), cache=False)
    assert council.cache_enabled is False
    await council.ask("hello")
    n = counting_call_model["n"]
    await council.ask("hello")
    assert counting_call_model["n"] == n * 2  # ran again, no hit


# --------------------------------------------------------------------------- #
# Key sensitivity
# --------------------------------------------------------------------------- #


async def test_changing_prompt_misses(monkeypatch, counting_call_model, cache_home):
    _set_keys(monkeypatch)
    council = Council(models=["grok"], synthesizer="claude", config=_config(), cache=True)
    await council.ask("prompt one")
    n = counting_call_model["n"]
    r = await council.ask("prompt two")
    assert counting_call_model["n"] > n
    assert r.cached is False


async def test_changing_council_membership_misses(monkeypatch, counting_call_model, cache_home):
    _set_keys(monkeypatch)
    c1 = Council(models=["grok"], synthesizer="claude", config=_config(), cache=True)
    await c1.ask("same prompt")
    n = counting_call_model["n"]
    c2 = Council(models=["grok", "perplexity"], synthesizer="claude", config=_config(), cache=True)
    r = await c2.ask("same prompt")
    assert counting_call_model["n"] > n
    assert r.cached is False


async def test_changing_mode_misses(monkeypatch, counting_call_model, cache_home):
    _set_keys(monkeypatch)
    council = Council(models=["grok"], synthesizer="claude", config=_config(), cache=True)
    await council.ask("same prompt", synthesize=True)
    n = counting_call_model["n"]
    r = await council.ask("same prompt", synthesize=False)  # raw mode -> different key
    assert counting_call_model["n"] > n
    assert r.cached is False


async def test_elite_cache_hit_preserves_artifacts_and_isolated_mode(
    monkeypatch, counting_call_model, cache_home
):
    _set_keys(monkeypatch)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
        cache=True,
        extract_verdict=False,
    )

    live = await council.elite("same prompt")
    calls_after_live = counting_call_model["n"]
    cached = await council.elite("same prompt")

    assert live.cached is False
    assert live.elite is not None
    assert live.elite.completed is True
    assert live.elite.decision_readiness == "indeterminate"
    assert live.elite.readiness_reasons == ["adjudication.disabled"]
    assert cached.cached is True
    assert cached.elite == live.elite
    assert counting_call_model["n"] == calls_after_live

    normal = await council.ask("same prompt")

    assert normal.cached is False
    assert normal.elite is None
    assert counting_call_model["n"] > calls_after_live
    assert len(list(cache_home.glob("*.json"))) == 2


def test_current_cache_shape_without_readiness_defaults_indeterminate(cache_home):
    """A legacy Elite payload in the current envelope can never replay as ready."""
    from conclave.models import CouncilResult, EliteResult

    key = "legacy-elite-readiness"
    cache_home.mkdir(parents=True, exist_ok=True)
    result = CouncilResult(
        prompt="q",
        mode="elite",
        elite=EliteResult(completed=True),
    ).model_dump(mode="json")
    del result["elite"]["decision_readiness"]
    del result["elite"]["readiness_reasons"]
    envelope = {
        "cache_format_version": cache_mod.CACHE_FORMAT_VERSION,
        "result": result,
    }
    (cache_home / f"{key}.json").write_text(json.dumps(envelope), encoding="utf-8")

    cached = cache_mod.load(key)

    assert cached is not None
    assert cached.elite is not None
    assert cached.elite.completed is True
    assert cached.elite.decision_readiness == "indeterminate"
    assert cached.elite.readiness_reasons == ["adjudication.not_evaluated"]


def test_previous_cache_format_payload_is_a_miss(cache_home):
    """Version 2 identities cannot replay after exact-prompt keying ships."""
    from conclave.models import CouncilResult

    key = "version-two-entry"
    cache_home.mkdir(parents=True, exist_ok=True)
    envelope = {
        "cache_format_version": "2",
        "result": CouncilResult(prompt="q", mode="raw").model_dump(mode="json"),
    }
    (cache_home / f"{key}.json").write_text(json.dumps(envelope), encoding="utf-8")

    assert cache_mod.load(key) is None


async def test_changing_model_id_misses(monkeypatch, counting_call_model, cache_home):
    """Same friendly name but a different resolved model id -> different key."""
    _set_keys(monkeypatch)
    cfg_a = _config()
    await Council(models=["grok"], synthesizer="claude", config=cfg_a, cache=True).ask("p")
    n = counting_call_model["n"]
    cfg_b = _config()
    cfg_b.models["grok"] = "xai/grok-4.3-mini"  # different resolved id
    r = await Council(models=["grok"], synthesizer="claude", config=cfg_b, cache=True).ask("p")
    assert counting_call_model["n"] > n
    assert r.cached is False


# --------------------------------------------------------------------------- #
# Security: no key material on disk or in the key
# --------------------------------------------------------------------------- #


async def test_no_key_value_in_cache_key_or_payload(monkeypatch, counting_call_model, cache_home):
    """A fake key VALUE must not appear in the cache key string or persisted file."""
    secret = "sk-CONCLAVE-SUPER-SECRET-KEY-VALUE-9f8e7d6c"
    for var in ("XAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, secret)

    council = Council(models=["grok"], synthesizer="claude", config=_config(), cache=True)

    # The cache key itself must contain zero key material.
    key = council._cache_key("audit prompt", "synthesize")
    assert secret not in key

    await council.ask("audit prompt")

    entries = list(cache_home.glob("*.json"))
    assert len(entries) == 1
    blob = entries[0].read_text(encoding="utf-8")
    assert secret not in blob
    # Also sanity-check the env var NAMES are absent from the stored payload.
    assert "XAI_API_KEY" not in blob
    assert "ANTHROPIC_API_KEY" not in blob
    # And the filename (the key) carries no secret.
    assert secret not in entries[0].name


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #


async def test_corrupt_entry_is_miss_no_crash(monkeypatch, counting_call_model, cache_home):
    """A corrupt cache file is treated as a miss; the run completes and rewrites it."""
    _set_keys(monkeypatch)
    council = Council(models=["grok"], synthesizer="claude", config=_config(), cache=True)

    # Pre-write a corrupt entry at the exact key the run will use.
    key = council._cache_key("q", "synthesize")
    cache_home.mkdir(parents=True, exist_ok=True)
    (cache_home / f"{key}.json").write_text("{ this is not valid json", encoding="utf-8")

    r = await council.ask("q")  # must not raise
    assert r.cached is False  # corrupt entry was a miss, ran live
    assert counting_call_model["n"] > 0
    # The corrupt entry was overwritten with a valid one -> next run hits.
    n = counting_call_model["n"]
    r2 = await council.ask("q")
    assert r2.cached is True
    assert counting_call_model["n"] == n


async def test_unreadable_payload_schema_is_miss(monkeypatch, cache_home):
    """A JSON file that is not a valid CouncilResult is a miss, not a crash."""
    cache_home.mkdir(parents=True, exist_ok=True)
    key = "deadbeef"
    (cache_home / f"{key}.json").write_text(json.dumps({"not": "a result"}), encoding="utf-8")
    assert cache_mod.load(key) is None


async def test_write_failure_does_not_crash_run(monkeypatch, counting_call_model, cache_home):
    """A failing cache write degrades to a normal live run (no exception)."""
    _set_keys(monkeypatch)

    # Simulate a low-level failure in path resolution: both load() and store()
    # must swallow it and degrade to a normal live run with no caching.
    def raise_oserror(key):
        raise OSError("simulated cache path failure")

    monkeypatch.setattr(cache_mod, "_entry_path", raise_oserror)

    council = Council(models=["grok"], synthesizer="claude", config=_config(), cache=True)
    # Even though path resolution fails inside store(), the run completes.
    r = await council.ask("q")
    assert r.cached is False
    assert counting_call_model["n"] > 0


# --------------------------------------------------------------------------- #
# Cache key direct unit checks
# --------------------------------------------------------------------------- #


def test_make_key_is_deterministic_and_order_sensitive():
    base = dict(
        prompt="hello world",
        mode="synthesize",
        synthesizer="claude",
        synthesizer_model_id="anthropic/claude-sonnet-4-6",
        temperature=0.7,
    )
    k1 = cache_mod.make_key(members=[("a", "x/1"), ("b", "y/2")], **base)
    k2 = cache_mod.make_key(members=[("a", "x/1"), ("b", "y/2")], **base)
    k3 = cache_mod.make_key(members=[("b", "y/2"), ("a", "x/1")], **base)
    assert k1 == k2  # deterministic
    assert k1 != k3  # member order matters (debate/adversarial ordering)
    assert len(k1) == 64  # sha256 hex


def test_make_key_preserves_exact_prompt_whitespace():
    common = dict(
        mode="raw",
        members=[("a", "x/1")],
        synthesizer=None,
        synthesizer_model_id=None,
        temperature=0.7,
    )
    assert cache_mod.make_key(prompt="a\n  b", **common) != cache_mod.make_key(
        prompt="a b", **common
    )


def test_make_key_debate_converge_threshold_differs(monkeypatch):
    """A converged-config debate and a no-converge debate must NOT collide (issue #4).

    Otherwise identical inputs: only ``converge_threshold`` differs. The cache key
    must differ so a converged run (which may stop early) is never served for a
    fixed-rounds request, and vice versa.
    """
    base = dict(
        prompt="hello world",
        mode="debate",
        members=[("a", "x/1"), ("b", "y/2")],
        synthesizer="claude",
        synthesizer_model_id="anthropic/claude-sonnet-4-6",
        temperature=0.7,
        rounds=5,
    )
    k_off = cache_mod.make_key(converge_threshold=None, **base)
    k_on = cache_mod.make_key(converge_threshold=0.9, **base)
    k_on2 = cache_mod.make_key(converge_threshold=0.95, **base)
    assert k_off != k_on  # converge on vs off -> different keys
    assert k_on != k_on2  # different thresholds -> different keys
    # Determinism preserved.
    assert k_off == cache_mod.make_key(converge_threshold=None, **base)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("cache_format_version", "next-cache-format"),
        ("protocol_version", "next-elite-protocol"),
        ("synthesis_prompt_version", "next-synthesis-prompt"),
        ("elite_prompt_version", "next-elite-prompt"),
        ("verdict_schema_version", "next-verdict-schema"),
        ("verdict_prompt_version", "next-verdict-prompt"),
        ("timeout", 45.0),
        ("extract_verdict", False),
        ("source_bundle_digest", "sha256:grounded-source-bundle"),
    ],
)
def test_make_key_varies_for_every_protocol_identity_dimension(override, value):
    """Every output-affecting protocol/version setting must invalidate reuse."""
    base = dict(
        prompt="Should we ship?",
        mode="elite",
        members=[("grok", "xai/grok-4.3"), ("gemini", "gemini/gemini-2.5-pro")],
        synthesizer="claude",
        synthesizer_model_id="anthropic/claude-sonnet-4-6",
        temperature=0.7,
        timeout=120.0,
        extract_verdict=True,
        source_bundle_digest=None,
    )
    changed = dict(base)
    changed[override] = value

    assert cache_mod.make_key(**base) != cache_mod.make_key(**changed)


def test_cache_identity_covers_roster_mode_params_and_safe_custom_endpoint_config():
    """Roster, mode parameters, and endpoint routing all affect identity."""
    base = dict(
        prompt="pick one",
        mode="vote",
        members=[("a", "custom/model-a")],
        synthesizer="judge",
        synthesizer_model_id="custom/judge-a",
        temperature=0.2,
        timeout=30.0,
        extract_verdict=True,
        choices=["A", "B"],
        endpoint_urls={"custom": "https://gateway.example/v1/chat?api-version=2026-01"},
    )

    assert cache_mod.make_key(**base) != cache_mod.make_key(
        **{**base, "members": [("a", "custom/model-b")]}
    )
    assert cache_mod.make_key(**base) != cache_mod.make_key(**{**base, "choices": ["A", "B", "C"]})
    assert cache_mod.make_key(**base) != cache_mod.make_key(
        **{
            **base,
            "endpoint_urls": {"custom": "https://other.example/v1/chat?api-version=2026-01"},
        }
    )
    assert cache_mod.make_key(**base) != cache_mod.make_key(
        **{
            **base,
            "endpoint_urls": {"custom": "https://gateway.example/v1/chat?api-version=2027-01"},
        }
    )


def test_cache_identity_never_contains_or_depends_on_endpoint_credentials():
    """Credential-bearing URL components and API-key values are non-identity."""
    secret = "sk-CONCLAVE-ENDPOINT-SECRET-0123456789"
    safe_url = "https://gateway.example/v1/chat?api-version=2026-01"
    credentialed_url = (
        f"https://user:{secret}@gateway.example/v1/chat?"
        f"api-version=2026-01&api_key={secret}&key={secret}&sig={secret}&"
        f"code={secret}&access_token={secret}&auth={secret}&password={secret}&"
        f"label={secret}#private-fragment"
    )
    common = dict(
        prompt="q",
        mode="elite",
        members=[("custom", "custom/model")],
        synthesizer="custom",
        synthesizer_model_id="custom/model",
        temperature=0.7,
    )

    safe_identity = cache_mod.build_identity(**common, endpoint_urls={"custom": safe_url})
    credentialed_identity = cache_mod.build_identity(
        **common, endpoint_urls={"custom": credentialed_url}
    )
    blob = json.dumps(credentialed_identity, sort_keys=True)

    assert credentialed_identity == safe_identity
    assert secret not in blob
    assert credentialed_url not in blob
    assert "api_key" not in blob


def test_cache_identity_fingerprints_prompt_and_source_inputs():
    """Potentially sensitive prompt/source content never appears in identity."""
    secret = "sk_FAKE_PROMPT_SECRET_0123456789"
    identity = cache_mod.build_identity(
        prompt=f"analyze {secret}",
        mode="elite",
        members=[("a", "x/1")],
        synthesizer="a",
        synthesizer_model_id="x/1",
        temperature=0.7,
        source_bundle_digest=f"malformed-digest-{secret}",
    )

    blob = json.dumps(identity, sort_keys=True)
    assert secret not in blob
    assert "analyze" not in blob
    assert "malformed-digest" not in blob


def test_council_threads_resolved_identity_settings_into_cache_key():
    """Council cache identity includes resolved endpoint and runtime settings."""
    cfg = _config()
    cfg.models["custom"] = "private/model-a"
    cfg.endpoints["private"] = CustomEndpoint(
        completions_url="https://gateway.example/v1/chat?api-version=2026-01",
        env_var="PRIVATE_API_KEY",
    )
    base = Council(
        models=["custom"],
        synthesizer="custom",
        config=cfg,
        timeout=30.0,
        extract_verdict=True,
        source_bundle_digest="bundle-a",
    )
    base_key = base._cache_key("q", "elite")

    changed_timeout = Council(
        models=["custom"],
        synthesizer="custom",
        config=cfg,
        timeout=45.0,
        extract_verdict=True,
        source_bundle_digest="bundle-a",
    )
    changed_verdict = Council(
        models=["custom"],
        synthesizer="custom",
        config=cfg,
        timeout=30.0,
        extract_verdict=False,
        source_bundle_digest="bundle-a",
    )
    changed_source = Council(
        models=["custom"],
        synthesizer="custom",
        config=cfg,
        timeout=30.0,
        extract_verdict=True,
        source_bundle_digest="bundle-b",
    )
    changed_cfg = cfg.model_copy(deep=True)
    changed_cfg.endpoints[
        "private"
    ].completions_url = "https://other.example/v1/chat?api-version=2026-01"
    changed_endpoint = Council(
        models=["custom"],
        synthesizer="custom",
        config=changed_cfg,
        timeout=30.0,
        extract_verdict=True,
        source_bundle_digest="bundle-a",
    )

    assert base_key != changed_timeout._cache_key("q", "elite")
    assert base_key != changed_verdict._cache_key("q", "elite")
    assert base_key != changed_source._cache_key("q", "elite")
    assert base_key != changed_endpoint._cache_key("q", "elite")


def test_malformed_endpoint_port_degrades_to_safe_deterministic_fingerprint():
    """A malformed configured port cannot crash cache identity construction."""
    raw_url = "https://gateway.example:not-a-port/v1/chat?api-version=2026-01"
    identity = cache_mod.build_identity(
        prompt="q",
        mode="elite",
        members=[("custom", "custom/model")],
        synthesizer="custom",
        synthesizer_model_id="custom/model",
        temperature=0.7,
        endpoint_urls={"custom": raw_url},
    )

    blob = json.dumps(identity, sort_keys=True)
    assert raw_url not in blob
    assert len(identity["endpoint_fingerprints"]["custom"]) == 64


def test_old_unversioned_cache_payload_is_a_miss(cache_home):
    """Pre-envelope cache files are not replayed against a new protocol."""
    key = "legacy"
    cache_home.mkdir(parents=True, exist_ok=True)
    legacy = ModelAnswer(name="a", model_id="x/1", answer="old")
    from conclave.models import CouncilResult

    old_result = CouncilResult(prompt="q", answers=[legacy])
    (cache_home / f"{key}.json").write_text(old_result.model_dump_json(), encoding="utf-8")

    assert cache_mod.load(key) is None


async def test_cache_converge_vs_fixed_no_collision(cache_home, monkeypatch, patch_call_model):
    """End-to-end: a converged debate and a fixed debate get distinct cache entries.

    With caching enabled, running the same prompt as a converged debate then as a
    fixed (no-converge) debate must produce two separate cache files -- the second
    run must not be served the first run's result.
    """
    _set_keys(monkeypatch)

    def handler(model, messages, **kwargs):
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if "synthesizer concluding a multi-round" in system:
            return make_response("SYNTH")
        return make_response(f"identical stable answer from {model}")

    patch_call_model(handler)

    cfg = _config(cache=True)
    council = Council(models=["grok", "gemini"], synthesizer="claude", config=cfg, cache=True)

    converged = await council.debate("q", rounds=5, converge_threshold=0.9)
    fixed = await council.debate("q", rounds=5)  # no convergence

    # The converged run stopped early; the fixed run ran all 5 rounds. If they had
    # collided, the fixed run would have been served the converged (2-round) entry.
    assert converged.converged is True
    assert len(converged.rounds) == 2
    assert fixed.converged is False
    assert len(fixed.rounds) == 5
    # Two distinct cache files exist.
    files = list(cache_home.glob("*.json"))
    assert len(files) == 2
