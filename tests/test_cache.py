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
from conclave.manifest import AdjudicationAttempt, ModelHarnessManifest
from conclave.models import CouncilResult, ModelAnswer
from tests.conftest import install_council_script, make_failed_answer, make_ok_answer, make_response


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


def _chain_config() -> ConclaveConfig:
    """A deterministic config for the DSE-1512 adjudication-succession cache tests.

    Mirrors ``tests/test_adjudication.py``'s ``CFG`` so the chain candidate
    friendly names/model ids line up with :func:`tests.conftest.install_council_script`.
    """
    return ConclaveConfig(
        models={"claude": "anthropic/c", "grok": "xai/g", "gemini": "gemini/m"},
        cache=True,
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
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None, **kwargs
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


# --------------------------------------------------------------------------- #
# DSE-1512: chain identity + no-store on successor/exhausted adjudication
# --------------------------------------------------------------------------- #


def test_identity_includes_full_chain():
    """A different successor ladder invalidates a prior entry."""
    base = dict(
        prompt="p",
        mode="synthesize",
        members=[("g", "xai/g")],
        synthesizer="claude",
        synthesizer_model_id="anthropic/c",
        temperature=0.7,
    )
    one = cache_mod.make_key(**base, synthesizer_chain=[("claude", "anthropic/c")])
    two = cache_mod.make_key(
        **base, synthesizer_chain=[("claude", "anthropic/c"), ("grok", "xai/g")]
    )
    assert one != two

    doc = cache_mod.build_identity(
        **base, synthesizer_chain=[("claude", "anthropic/c"), ("grok", "xai/g")]
    )
    assert doc["synthesizer"] == ["claude", "anthropic/c"]  # legacy key kept
    assert doc["synthesizer_chain"] == [["claude", "anthropic/c"], ["grok", "xai/g"]]


def test_identity_chain_defaults_to_primary_when_omitted():
    """A direct caller that never passes ``synthesizer_chain`` still gets a stable value."""
    base = dict(
        prompt="p",
        mode="synthesize",
        members=[("g", "xai/g")],
        synthesizer="claude",
        synthesizer_model_id="anthropic/c",
        temperature=0.7,
    )
    omitted = cache_mod.make_key(**base)
    explicit = cache_mod.make_key(**base, synthesizer_chain=[("claude", "anthropic/c")])
    assert omitted == explicit

    doc = cache_mod.build_identity(**base)
    assert doc["synthesizer_chain"] == [["claude", "anthropic/c"]]


def test_cache_format_version_bumped():
    # Tracks the CURRENT format version, not a frozen historical one -- DSE-1514
    # bumped this from "4" (DSE-1512's synthesizer-chain identity) to "5" (the
    # price-snapshot fingerprint + max_output_tokens cap); see
    # test_cache_format_version_is_five_for_price_identity for the DSE-1514-named
    # assertion of the same fact.
    assert cache_mod.CACHE_FORMAT_VERSION == "5"


async def test_result_adjudicated_by_successor_is_not_stored(monkeypatch, keys, cache_home):
    """A run where the primary failed over to a successor is never persisted.

    gemini (member) ok, claude (primary) quota 429 -> failed over, grok
    (successor) ok -> success. Because the primary failed for an
    infrastructure reason, the run must not be cached: a second identical
    ``ask`` re-calls every provider rather than replaying the successor's
    answer from cache.
    """
    calls = install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "quota", 429),
            "grok": make_ok_answer("grok", "xai/g"),
        },
    )
    c = Council(
        models=["gemini"],
        synthesizer="claude>grok",
        config=_chain_config(),
        cache=True,
        extract_verdict=False,
    )
    r1 = await c.ask("q")
    r2 = await c.ask("q")

    assert r1.cached is False and r2.cached is False  # second run was NOT served from cache
    assert calls.count("gemini") == 2 and calls.count("grok") == 2
    assert not list(cache_home.glob("*.json"))  # nothing was ever written
    assert [a.outcome for a in r2.manifest.adjudication_succession] == [
        "failed_over",
        "success",
    ]


async def test_result_adjudicated_by_primary_is_stored(monkeypatch, keys, cache_home):
    """A run where the primary answers cleanly is cached like any other run."""
    calls = install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_ok_answer("claude", "anthropic/c"),
        },
    )
    c = Council(
        models=["gemini"],
        synthesizer="claude>grok",
        config=_chain_config(),
        cache=True,
        extract_verdict=False,
    )
    r1 = await c.ask("q")
    r2 = await c.ask("q")

    assert r1.cached is False
    assert r2.cached is True
    assert calls.count("gemini") == 1 and calls.count("claude") == 1  # not re-called
    assert [a.outcome for a in r2.manifest.adjudication_succession] == ["success"]


async def test_exhausted_run_is_not_stored(monkeypatch, keys, cache_home):
    """A chain-exhausted run (every candidate failed for an infra reason) is not cached.

    This is a deliberate, narrow behavior change from v1.3.0 (see
    ``Council._cached_run``'s docstring): caching a run whose whole chain was
    down means an operator would get the same failure back from cache after
    the outage clears.
    """
    calls = install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "quota", 429),
            "grok": make_failed_answer("grok", "xai/g", "unavailable", 503),
        },
    )
    c = Council(
        models=["gemini"],
        synthesizer="claude>grok",
        config=_chain_config(),
        cache=True,
        extract_verdict=False,
    )
    r1 = await c.ask("q")
    r2 = await c.ask("q")

    assert r1.cached is False and r1.degraded is True
    assert r2.cached is False and r2.degraded is True
    assert calls.count("gemini") == 2  # ran again on the second call, not from cache
    assert not list(cache_home.glob("*.json"))
    assert [a.outcome for a in r2.manifest.adjudication_succession] == [
        "failed_over",
        "exhausted",
    ]


async def test_terminal_failure_run_is_stored(monkeypatch, keys, cache_home):
    """A terminal-failure run (the model answered, just unusably) is still cached.

    claude returns a bad_request (400): the request was wrong, not an
    infrastructure problem, so the chain never fails over and the degraded
    result is cached exactly like any other content failure (unchanged from
    v1.3.0).
    """
    calls = install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "bad_request", 400),
        },
    )
    c = Council(
        models=["gemini"],
        synthesizer="claude>grok",
        config=_chain_config(),
        cache=True,
        extract_verdict=False,
    )
    r1 = await c.ask("q")
    r2 = await c.ask("q")

    assert r1.cached is False and r1.degraded is True
    assert r2.cached is True and r2.degraded is True
    assert calls.count("claude") == 1  # only the first, live run called claude
    assert [a.outcome for a in r2.manifest.adjudication_succession] == ["terminal_failure"]


async def test_chain_of_one_unkeyed_run_is_not_stored(monkeypatch, cache_home):
    """A chain-of-one unkeyed synthesizer IS primary_failed_over (DSE-1512 review,
    uniform rule) and is therefore NOT cached -- the ledger's lone
    ``skipped_unkeyed`` entry at attempt_index 1 means the primary never
    adjudicated, even though there is no successor to have adjudicated instead.
    Runs with ``extract_verdict=False`` (below) and the default
    ``extract_verdict=True`` (the second test) must agree: the rule does not
    depend on which roles ran.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = install_council_script(monkeypatch, {"gemini": make_ok_answer("gemini", "gemini/m")})
    c = Council(
        models=["gemini"],
        synthesizer="claude",
        config=_chain_config(),
        cache=True,
        extract_verdict=False,
    )
    r1 = await c.ask("q")
    r2 = await c.ask("q")

    assert r1.primary_failed_over is True
    assert r1.cached is False and r2.cached is False
    assert not list(cache_home.glob("*.json"))  # nothing was ever written
    assert calls.count("gemini") == 2  # second run re-called the providers, not cached


async def test_chain_of_one_unkeyed_run_is_not_stored_with_verdict_extraction(
    monkeypatch, cache_home
):
    """Same as above with the default ``extract_verdict=True``."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = install_council_script(monkeypatch, {"gemini": make_ok_answer("gemini", "gemini/m")})
    c = Council(
        models=["gemini"],
        synthesizer="claude",
        config=_chain_config(),
        cache=True,
    )
    r1 = await c.ask("q")
    r2 = await c.ask("q")

    assert r1.primary_failed_over is True
    assert r1.cached is False and r2.cached is False
    assert not list(cache_home.glob("*.json"))
    assert calls.count("gemini") == 2


# --------------------------------------------------------------------------- #
# CouncilResult.primary_failed_over truth table (DSE-1512 review, uniform rule).
# Constructs a ModelHarnessManifest directly -- no council run -- to pin the
# computed field's rule independent of how any particular role's ledger gets
# built.
# --------------------------------------------------------------------------- #


def _attempt(role: str, outcome: str, index: int) -> AdjudicationAttempt:
    """Build one succession-ledger entry for the truth-table test below."""
    return AdjudicationAttempt(
        role=role,
        candidate="claude",
        model_id="anthropic/c",
        attempt_index=index,
        outcome=outcome,
    )


def _manifest_with(attempts: list[AdjudicationAttempt]) -> ModelHarnessManifest:
    return ModelHarnessManifest(
        request_id="r",
        conclave_version="v",
        mode="synthesize",
        adjudication_succession=attempts,
    )


@pytest.mark.parametrize(
    ("attempts", "expected"),
    [
        pytest.param([], False, id="empty_ledger"),
        pytest.param(
            [_attempt("synthesis", "terminal_failure", 1)], False, id="terminal_failure_at_1"
        ),
        pytest.param(
            [_attempt("synthesis", "skipped_unkeyed", 1)], True, id="skipped_unkeyed_at_1"
        ),
        pytest.param(
            [_attempt("synthesis", "skipped_unkeyed", 1), _attempt("synthesis", "success", 2)],
            True,
            id="skipped_unkeyed_then_success",
        ),
        pytest.param(
            [
                _attempt("synthesis", "skipped_unkeyed", 1),
                _attempt("synthesis", "terminal_failure", 2),
            ],
            True,
            id="skipped_unkeyed_then_terminal_failure",
        ),
        pytest.param(
            [
                _attempt("synthesis", "failed_over", 1),
                _attempt("synthesis", "skipped_unkeyed", 2),
            ],
            True,
            id="failed_over_then_skipped_unkeyed",
        ),
        pytest.param([_attempt("synthesis", "success", 1)], False, id="success_at_1"),
        pytest.param([_attempt("synthesis", "exhausted", 1)], True, id="exhausted_at_1"),
        pytest.param(
            [_attempt("synthesis", "success", 1), _attempt("judge", "failed_over", 1)],
            True,
            id="second_role_index_1_failed_over",
        ),
    ],
)
def test_primary_failed_over_truth_table(attempts, expected):
    """primary_failed_over depends only on each role's attempt_index==1 outcome,
    uniformly across every role, independent of whether a council run ever
    actually happened.
    """
    result = CouncilResult(prompt="p", manifest=_manifest_with(attempts))
    assert result.primary_failed_over is expected


# --------------------------------------------------------------------------- #
# DSE-1512 review, Unit A2: ask_stream's cache store must honor the same
# no-store-on-primary-failover rule as the buffered path (_cached_run).
# --------------------------------------------------------------------------- #


def _stream_chain_config() -> ConclaveConfig:
    """Mirrors ``_chain_config`` with a friendly-name roster wide enough for
    both the sole council member (``gemini``) and the ``claude>grok`` chain.
    """
    return ConclaveConfig(
        models={"claude": "anthropic/c", "grok": "xai/g", "gemini": "gemini/m"},
        cache=True,
    )


def _install_stream_script(monkeypatch, script: dict[str, list]) -> list[str]:
    """Patch the streaming ``call_model_stream`` seam with a per-name script.

    Mirrors ``tests/test_streaming.py``'s helper of the same name: council
    members and synthesizer-chain candidates share this one seam, so a script
    entry is needed for every name a test's run touches.
    """
    import conclave.streaming as streaming_mod

    calls: list[str] = []

    async def fake_stream(
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None, **kwargs
    ):
        calls.append(name)
        for item in script[name]:
            yield item

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)
    return calls


async def test_stream_successor_run_is_not_stored(monkeypatch, keys, cache_home):
    """A stream whose primary synthesizer fails over is never persisted.

    claude yields only an errored final (429, no deltas) so it fails over
    before any output; grok then streams cleanly. Because the primary failed
    for an infrastructure reason, the run must not reach the result cache --
    a subsequent buffered ``ask`` re-calls every provider rather than
    replaying the successor's answer from a stale entry.
    """
    _install_stream_script(
        monkeypatch,
        {
            "gemini": ["gemini ", "says yes", make_ok_answer("gemini", "gemini/m")],
            "claude": [make_failed_answer("claude", "anthropic/c", "quota", 429)],
            "grok": ["grok ", "says yes", make_ok_answer("grok", "xai/g")],
        },
    )
    c = Council(
        models=["gemini"],
        synthesizer="claude>grok",
        config=_stream_chain_config(),
        cache=True,
        extract_verdict=False,
    )

    events = [e async for e in c.ask_stream("q")]
    result = events[-1].result
    assert result.cached is False
    assert result.primary_failed_over is True
    assert not list(cache_home.glob("*.json"))  # nothing was ever written

    # A subsequent buffered ask is NOT served from cache: every provider is
    # called again rather than replaying the successor's answer.
    buffered_calls = install_council_script(
        monkeypatch,
        {
            "gemini": make_ok_answer("gemini", "gemini/m"),
            "claude": make_failed_answer("claude", "anthropic/c", "quota", 429),
            "grok": make_ok_answer("grok", "xai/g"),
        },
    )
    r2 = await c.ask("q")
    assert r2.cached is False
    assert buffered_calls == ["gemini", "claude", "grok"]


async def test_stream_primary_run_is_stored(monkeypatch, keys, cache_home):
    """A stream whose primary synthesizer succeeds is cached like any other run."""
    _install_stream_script(
        monkeypatch,
        {
            "gemini": ["gemini ", "says yes", make_ok_answer("gemini", "gemini/m")],
            "claude": ["claude ", "says yes", make_ok_answer("claude", "anthropic/c")],
        },
    )
    c = Council(
        models=["gemini"],
        synthesizer="claude>grok",
        config=_stream_chain_config(),
        cache=True,
        extract_verdict=False,
    )

    first = [e async for e in c.ask_stream("q")]
    assert first[-1].result.cached is False
    assert first[-1].result.primary_failed_over is False
    assert len(list(cache_home.glob("*.json"))) == 1

    r2 = await c.ask("q")
    assert r2.cached is True


def test_cache_format_version_is_five_for_price_identity():
    from conclave import cache as cache_mod

    assert cache_mod.CACHE_FORMAT_VERSION == "5"


def test_identity_carries_the_price_snapshot_fingerprint_and_output_cap():
    from conclave.cache import build_identity

    base = {
        "prompt": "q",
        "mode": "synthesize",
        "members": [("grok", "xai/grok-4.3")],
        "synthesizer": "claude",
        "synthesizer_model_id": "anthropic/claude-sonnet-4-6",
        "temperature": 0.7,
    }
    plain = build_identity(**base)
    assert plain["price_snapshot_fingerprint"] is None
    assert plain["generation"]["max_output_tokens"] is None

    priced = build_identity(**base, price_snapshot_digest="sha256:" + "c" * 64)
    other = build_identity(**base, price_snapshot_digest="sha256:" + "d" * 64)
    assert priced["price_snapshot_fingerprint"] != plain["price_snapshot_fingerprint"]
    assert priced["price_snapshot_fingerprint"] != other["price_snapshot_fingerprint"]
    # The raw digest never enters the inspectable identity document -- it is
    # re-hashed, matching how source_bundle_digest is handled.
    assert "c" * 64 not in str(priced)

    capped = build_identity(**base, max_output_tokens=512)
    assert capped["generation"]["max_output_tokens"] == 512


def test_two_snapshots_never_share_a_cache_key():
    from conclave.cache import make_key

    base = {
        "prompt": "q",
        "mode": "synthesize",
        "members": [("grok", "xai/grok-4.3")],
        "synthesizer": "claude",
        "synthesizer_model_id": "anthropic/claude-sonnet-4-6",
        "temperature": 0.7,
    }
    assert make_key(**base, price_snapshot_digest="sha256:" + "c" * 64) != make_key(
        **base, price_snapshot_digest="sha256:" + "d" * 64
    )
    assert make_key(**base, max_output_tokens=512) != make_key(**base, max_output_tokens=1024)


async def test_a_cached_result_round_trips_its_decimal_ceiling(tmp_path, monkeypatch, keys):
    """A cache hit must return an exact Decimal ceiling, not a float or a string."""
    from decimal import Decimal

    import conclave.council as council_mod
    from conclave.council import Council
    from conclave.models import ModelAnswer, TokenUsage
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3"))

    async def ok(name, model_id, messages, **kwargs):
        return ModelAnswer(
            name=name,
            model_id=model_id,
            answer="ok",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    monkeypatch.setattr(council_mod, "call_model", ok)
    council = Council(models=["grok"], synthesizer="grok", cache=True, extract_verdict=False)
    live = await council.ask("q", synthesize=False)
    hit = await council.ask("q", synthesize=False)

    assert hit.cached is True
    assert isinstance(hit.manifest.cost_ceiling_usd, Decimal)
    assert hit.manifest.cost_ceiling_usd == live.manifest.cost_ceiling_usd
