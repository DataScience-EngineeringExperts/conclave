"""CAC-07 headline secret-safety matrix (sweep item 1): the canary leaks nowhere.

The load-bearing security sweep. A synthetic, key-shaped canary (the keyleak-audit
``PLANTED`` token — gitleaks-allowlisted, contains ``FAKE``) is planted as the LIVE
value of every provider key the council uses, then THREE flows are driven:

* **1a — buffered ``Council.ask(synthesize=True)`` with a real verdict** (offline
  dual-seam harness): no member ever hits the network, so the success path proves
  the canary value is absent from every serialized surface even when redaction was
  never exercised.
* **1b — streamed ``ask_stream(synthesize=True)``** (member stream seam fake +
  ``patch_call_model`` verdict seam): every ``StreamEvent`` AND the terminal
  ``done`` result are scanned.
* **1c — forced provider-error run on the REAL ``call_model`` highway** (NOT the
  offline seam): ``conclave.transport.post_json`` is mocked to return a 401 whose
  body ECHOES the canary (the v1.0 leak class). This is the STRONG negative control
  — a non-redacting code path WOULD surface the canary here. Includes a streaming
  variant via the copied ``mock_stream_client`` httpx ``MockTransport`` fixture.

Each flow asserts BOTH distinct invariants the brief separates:
  (a) the literal canary VALUE (the full ``PLANTED`` string) appears in NONE of the
      enumerated serialized surfaces; AND
  (b) ``manifest.secret_safety == SECRET_SAFETY_VERIFIED`` and
      ``scan_for_secret_material(manifest) is True`` (structurally clean — no
      ``sk-``/``bearer``/``authorization``/``api_key``/``x-api-key`` anywhere).

Note the ``sk-`` overlap: ``PLANTED`` starts with ``sk-``, which is also one of the
manifest's forbidden substrings, so (b) can only pass if redaction genuinely worked
and the canary never reached the manifest.

All tests offline. The ``mock_stream_client`` fixture is reproduced from
``tests/test_keyleak_audit.py`` (test-local there; copied, not imported).
"""

from __future__ import annotations

import json

import httpx
import pytest

from conclave import Council, transport
from conclave.config import ConclaveConfig, clear_config_cache
from conclave.manifest import (
    SECRET_SAFETY_VERIFIED,
    ModelHarnessManifest,
    scan_for_secret_material,
)
from conclave.models import ModelAnswer
from conclave.providers import receipt_from_answer
from tests.conftest import make_response

# The synthesizer/extractor resolved id for the "claude" friendly name below.
_SYNTH_MODEL_ID = "anthropic/claude-sonnet-4-6"

# The synthetic, key-shaped canary (reused verbatim from test_keyleak_audit.py).
# It starts with ``sk-`` (also a forbidden manifest substring) and contains
# ``FAKE`` so gitleaks ignores it. If this value appears in any serialized surface
# below, a leak has occurred.
PLANTED = "sk-FAKEconclaveLEAK0123456789abcdefSECRET"


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Isolate each test from the in-process config memo (mirrors keyleak audit)."""
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
async def mock_stream_client():
    """Install a MockTransport-backed pooled client; restore the global after.

    Reproduced from ``tests/test_keyleak_audit.py`` (and ``test_streaming.py``) so
    the real ``call_model_stream`` -> ``transport.stream_sse`` -> adapter path runs
    against a caller-supplied handler with no network.
    """
    saved = transport._client
    created: list[httpx.AsyncClient] = []

    def use(handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        created.append(client)
        transport._client = client
        return client

    yield use

    for client in created:
        if not client.is_closed:
            await client.aclose()
    transport._client = saved


# --------------------------------------------------------------------------- #
# Test-local harness helpers (copied verbatim from the canonical templates).
# --------------------------------------------------------------------------- #
def _plant_all_keys(monkeypatch) -> None:
    """Plant the canary as the LIVE value of every key the council config uses."""
    for var in (
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(var, PLANTED)


def _config(synthesizer: str = "claude") -> ConclaveConfig:
    """A deterministic config independent of any on-disk ~/.conclave file."""
    return ConclaveConfig(
        models={
            "grok": "xai/grok-4.3",
            "gemini": "gemini/gemini-2.5-pro",
            "claude": _SYNTH_MODEL_ID,
            "perplexity": "perplexity/sonar-pro",
        },
        councils={"default": ["grok", "gemini", "claude", "perplexity"]},
        synthesizer=synthesizer,
    )


def _is_verdict_call(messages) -> bool:
    """True when ``messages`` is the verdict-extraction call (vs a member/prose call)."""
    return bool(messages) and messages[0].get("content", "").startswith(
        "You are the verdict extractor"
    )


def _extraction_json(
    *,
    verdict_applies: bool = True,
    verdict_type: str = "decision",
    members: tuple[str, ...] = ("grok", "gemini", "perplexity"),
    position_label: str = "yes",
) -> str:
    """Build valid verdict-extraction JSON where every member votes the same way.

    Copied from ``tests/test_council_verdict.py`` (per-file helper).
    """
    return json.dumps(
        {
            "verdict_applies": verdict_applies,
            "verdict_type": verdict_type,
            "headline": "Yes.",
            "recommendation": "Proceed with yes.",
            "positions": [
                {
                    "label": position_label,
                    "summary": "All members agree: yes.",
                    "providers": list(members),
                    "evidence_answer_ids": [],
                }
            ],
            "provider_votes": [
                {"provider": name, "position_label": position_label} for name in members
            ],
            "minority_reports": [],
            "conflicts": [],
            "caveats": [],
            "dissent_summary": None,
        }
    )


def _patch_member_stream(monkeypatch, deltas_by_model, errors_by_model=None) -> None:
    """Patch the MEMBER stream seam with canned deltas + a final answer per model.

    Copied verbatim from ``tests/test_streaming_verdict.py``.
    """
    import conclave.streaming as streaming_mod

    errors_by_model = errors_by_model or {}

    async def fake_stream(
        name, model_id, messages, *, temperature=0.7, timeout=120.0, config=None, **kwargs
    ):
        text_parts = deltas_by_model.get(model_id, ["x"])
        for part in text_parts:
            yield part
        err = errors_by_model.get(model_id)
        if err is not None:
            yield ModelAnswer(name=name, model_id=model_id, answer="".join(text_parts), error=err)
        else:
            yield ModelAnswer(name=name, model_id=model_id, answer="".join(text_parts))

    monkeypatch.setattr(streaming_mod, "call_model_stream", fake_stream)


# --------------------------------------------------------------------------- #
# Shared assertion: the canary value leaks into NONE of the enumerated surfaces.
# --------------------------------------------------------------------------- #
def _assert_result_canary_free(result) -> None:
    """Assert the canary VALUE (a) and structural cleanliness (b) over a result.

    (a) the full ``PLANTED`` string is absent from: the whole
        ``result.model_dump(mode="json")`` payload (which is the ``--json`` CLI
        surface), the serialized verdict / conflicts / provider_votes /
        minority_reports / manifest (receipts + redacted_errors), and every
        ``ModelAnswer.error``; AND
    (b) ``manifest.secret_safety == SECRET_SAFETY_VERIFIED`` and
        ``scan_for_secret_material(manifest) is True``.
    """
    # (a) The whole result payload — identical to the CLI ``--json`` surface
    # (``result.model_dump(mode="json")``).
    full = json.dumps(result.model_dump(mode="json"))
    assert PLANTED not in full, "canary leaked into the full result payload (= --json surface)"

    # (a) Each adjudication surface individually, so a leak is attributable.
    assert PLANTED not in json.dumps(result.model_dump(mode="json").get("verdict")), (
        "canary leaked into verdict"
    )
    assert PLANTED not in json.dumps(result.model_dump(mode="json").get("conflicts")), (
        "canary leaked into conflicts"
    )
    assert PLANTED not in json.dumps(result.model_dump(mode="json").get("provider_votes")), (
        "canary leaked into provider_votes"
    )
    assert PLANTED not in json.dumps(result.model_dump(mode="json").get("minority_reports")), (
        "canary leaked into minority_reports"
    )

    # (a) The manifest payload (covers receipts + redacted_errors).
    assert result.manifest is not None
    manifest_blob = result.manifest.model_dump_json()
    assert PLANTED not in manifest_blob, "canary leaked into the manifest"

    # (a) Every member error string is redacted.
    for ans in result.answers:
        if ans.error is not None:
            assert PLANTED not in ans.error, f"canary leaked into {ans.name}.error"

    # (b) The manifest is structurally clean AND stamped VERIFIED.
    assert result.manifest.secret_safety == SECRET_SAFETY_VERIFIED
    assert scan_for_secret_material(result.manifest) is True


def test_receipt_reduces_raw_provider_error_to_secret_free_category() -> None:
    """Receipts never retain endpoint URLs, response bodies, or echoed credentials."""
    answer = ModelAnswer(
        name="grok",
        model_id="xai/grok-4.3",
        error=f"RuntimeError: POST https://provider.example/v1 body={PLANTED}",
    )

    receipt = receipt_from_answer(answer, temperature=0.7, timeout=120.0)
    payload = receipt.model_dump_json()

    assert receipt.outcome == "failed"
    assert receipt.error_category == "provider_error"
    assert receipt.error == "provider_error"
    assert "provider.example" not in payload
    assert PLANTED not in payload


# --------------------------------------------------------------------------- #
# 1a — buffered Council.ask(synthesize=True) WITH a real verdict.
# --------------------------------------------------------------------------- #
async def test_secret_safety_ask_with_verdict(monkeypatch, patch_call_model):
    """1a: the canary (planted as every live key) leaks into no surface of a real verdict run.

    Under ``patch_call_model`` members never hit the network, so the key is never
    echoed on this path — making this the STRONG positive control for the SUCCESS
    path (the dumped result + manifest are scanned). The echo-forced negative
    control is flow 1c.
    """
    _plant_all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response(f"yes from {model}")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("Should we ship?")

    # The verdict actually formed (so the scanned surfaces are non-trivial).
    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    _assert_result_canary_free(result)


# --------------------------------------------------------------------------- #
# 1b — streamed ask_stream(synthesize=True): every event + the done result.
# --------------------------------------------------------------------------- #
async def test_secret_safety_stream_with_verdict(monkeypatch, patch_call_model):
    """1b: no StreamEvent and no terminal done result carries the canary value.

    Members flow through the stream seam, the verdict through ``patch_call_model``.
    Every ``StreamEvent.model_dump(mode="json")`` is scanned, plus the terminal
    ``done`` result via the shared assertion.
    """
    _plant_all_keys(monkeypatch)
    members = ("grok", "gemini", "perplexity")

    _patch_member_stream(
        monkeypatch,
        {
            "xai/grok-4.3": ["yes ", "grok"],
            "gemini/gemini-2.5-pro": ["yes ", "gemini"],
            "perplexity/sonar-pro": ["yes ", "perplexity"],
            _SYNTH_MODEL_ID: ["SYN"],
        },
    )

    def handler(model, messages, **kwargs):
        if _is_verdict_call(messages):
            return make_response(_extraction_json(members=members))
        return make_response("unused prose")

    patch_call_model(handler)
    council = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("Should we ship?", synthesize=True)]

    # No StreamEvent anywhere carries the canary.
    for ev in events:
        dumped = json.dumps(ev.model_dump(mode="json"))
        assert PLANTED not in dumped, f"canary leaked into a {ev.type} event"

    done = events[-1]
    assert done.type == "done"
    result = done.result
    assert result is not None
    # The streamed verdict formed.
    assert result.verdict is not None
    assert result.verdict.consensus_score == 1.0
    _assert_result_canary_free(result)


# --------------------------------------------------------------------------- #
# 1c — forced provider-error on the REAL call_model highway (echoing 401).
# --------------------------------------------------------------------------- #
async def test_secret_safety_forced_error_buffered(monkeypatch):
    """1c (buffered): a 401 echoing the canary is redacted everywhere; manifest VERIFIED.

    The STRONG negative control. We drive the REAL ``call_model`` highway (no
    offline seam): keys are the canary value and ``conclave.transport.post_json``
    returns a 401 whose body ECHOES the canary — exactly the v1.0 leak class. A
    non-redacting path WOULD surface the canary here. ``CONCLAVE_CONFIG`` points at
    a nonexistent path and the config is built directly, mirroring the keyleak
    tests. Every member fails -> each ``ModelAnswer.error`` must be redacted, the
    verdict is absent (N<2), and the manifest stays VERIFIED.
    """
    _plant_all_keys(monkeypatch)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    async def echoing_401(url, headers, json_body, timeout):
        return 401, {"error": {"message": f"invalid api key: {PLANTED}"}}

    monkeypatch.setattr("conclave.transport.post_json", echoing_401)

    members = ("grok", "gemini", "perplexity")
    council = Council(models=list(members), synthesizer="claude", config=_config())
    result = await council.ask("audit prompt", synthesize=True)

    # Every member failed; each error is redacted (canary masked, marker present).
    assert result.answers
    assert all(not a.ok for a in result.answers)
    for ans in result.answers:
        assert ans.error is not None
        assert PLANTED not in ans.error, f"canary leaked into {ans.name}.error"
        assert "[REDACTED]" in ans.error

    # No verdict (every member failed -> N<2), but synthesis-error recorded and the
    # full result + manifest remain canary-free and VERIFIED.
    assert result.verdict is None
    _assert_result_canary_free(result)


async def test_secret_safety_forced_error_elite(monkeypatch):
    """Elite phase artifacts and receipts redact a provider error echoing a key."""
    _plant_all_keys(monkeypatch)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    async def echoing_401(url, headers, json_body, timeout):
        return 401, {"error": {"message": f"invalid api key: {PLANTED}"}}

    monkeypatch.setattr("conclave.transport.post_json", echoing_401)
    council = Council(
        models=["grok", "gemini", "perplexity"],
        synthesizer="claude",
        config=_config(),
    )

    result = await council.elite("audit prompt")

    assert result.elite is not None
    assert result.elite.completed is False
    assert result.elite.critiques == []
    assert result.elite.revisions == []
    assert all(not answer.ok for answer in result.elite.initial_answers)
    _assert_result_canary_free(result)
    assert [receipt.phase for receipt in result.manifest.receipts] == ["initial"] * 3


async def test_secret_safety_forced_error_streaming(monkeypatch, mock_stream_client):
    """1c (streaming): a mid-stream 401 echoing the canary leaks into no event/answer.

    The streaming negative control at the council level. The httpx ``MockTransport``
    handler returns a 401 whose body echoes the canary; the real
    ``call_model_stream`` -> ``stream_sse`` path must surface a redacted error on
    the final ``ModelAnswer`` and emit no ``StreamEvent`` carrying the canary. The
    terminal ``done`` result + manifest stay clean and VERIFIED.
    """
    _plant_all_keys(monkeypatch)
    monkeypatch.setenv("CONCLAVE_CONFIG", "/nonexistent/conclave.yml")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {PLANTED}"}})

    mock_stream_client(handler)

    members = ("grok", "gemini", "perplexity")
    council = Council(models=list(members), synthesizer="claude", config=_config())
    events = [e async for e in council.ask_stream("audit prompt", synthesize=True)]

    # No StreamEvent anywhere carries the canary.
    for ev in events:
        dumped = json.dumps(ev.model_dump(mode="json"))
        assert PLANTED not in dumped, f"canary leaked into a {ev.type} event"

    done = events[-1]
    assert done.type == "done"
    result = done.result
    assert result is not None

    # Every member failed via the echoing 401; each error is redacted.
    assert result.answers
    assert all(not a.ok for a in result.answers)
    for ans in result.answers:
        assert ans.error is not None
        assert PLANTED not in ans.error
        assert "[REDACTED]" in ans.error

    # Final result + manifest clean and VERIFIED.
    _assert_result_canary_free(result)


# --------------------------------------------------------------------------- #
# Direct manifest-scan unit check: a planted canary in a free-text field fails the
# scan (proves the (b) assertion above is a real gate, not vacuously true).
# --------------------------------------------------------------------------- #
def test_scan_rejects_planted_canary_in_manifest():
    """Sanity: a manifest carrying the canary in a free-text field fails the scan.

    Guards against the (b) assertion being vacuous: if ``scan_for_secret_material``
    could not detect the canary (which starts with the forbidden ``sk-``), a VERIFIED
    stamp on a leaking manifest would be meaningless. Here we inject the canary into
    ``redacted_errors`` and assert the scan returns False.
    """
    polluted = ModelHarnessManifest(
        request_id="r1",
        conclave_version="0.0.0",
        mode="synthesize",
        redacted_errors=[f"leaked credential {PLANTED}"],
    )
    assert scan_for_secret_material(polluted) is False


# --------------------------------------------------------------------------- #
# DSE-1514 Task 13: the pricing fields never un-verify the secret-safety stamp,
# and pricing_warnings is a closed vocabulary -- statically (source scan) and
# dynamically (every reachable warning shape across every mode).
# --------------------------------------------------------------------------- #


def test_pricing_fields_never_un_verify_the_stamp():
    """A model id with an awkward substring must not break the stamp."""
    from decimal import Decimal

    from conclave.manifest import (
        SECRET_SAFETY_VERIFIED,
        ModelHarnessManifest,
        ProviderExecutionReceipt,
        verified_secret_safety,
    )

    manifest = ModelHarnessManifest(
        request_id="r",
        conclave_version="1.3.0",
        mode="synthesize",
        model_ids=["deepseek/deepseek-chat", "together/meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        receipts=[
            ProviderExecutionReceipt(
                name="deepseek",
                provider="deepseek",
                model_id="deepseek/deepseek-chat",
                cost_ceiling_usd=Decimal("0.000123"),
                cost_basis="reservation",
                generation_settings={
                    "temperature": 0.7,
                    "timeout": 120.0,
                    "max_output_tokens": 8,
                },
            )
        ],
        cost_ceiling_usd=Decimal("0.000123"),
        price_snapshot_digest="sha256:" + "e" * 64,
        priced_as_of="2026-09-03",
        unpriced_models=["together/meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        unpriced_receipts=0,
        pricing_warnings=[
            "unpriced_models_present",
            "price_snapshot_stale",
            "no_output_cap_configured",
        ],
    )
    assert verified_secret_safety(manifest) == SECRET_SAFETY_VERIFIED


def test_pricing_warnings_are_a_closed_vocabulary_in_the_code():
    """No pricing warning may be built by interpolation.

    Scans BOTH shapes ``Council._price_manifest`` uses to populate
    ``pricing_warnings`` (DSE-1514 QA M5): the normal path's sequence of
    ``warnings.append(...)`` calls, AND the missing-snapshot path's single
    list-literal assignment (``manifest.pricing_warnings = [...]``) -- the
    module's own top-of-file comment on :data:`conclave.council.
    PRICING_WARNING_VOCABULARY` has claimed both are covered since the
    vocabulary was introduced; this test previously only scanned the first.
    The list-literal pattern requires a ``[`` immediately after ``=``, so it
    does not match ``manifest.pricing_warnings = warnings`` (assigning the
    already-scanned variable back onto the manifest) -- only a literal.
    """
    import re
    from pathlib import Path

    import conclave

    source = (Path(conclave.__file__).parent / "council.py").read_text(encoding="utf-8")
    appended = re.findall(r"warnings\.append\((.+?)\)", source)
    assert appended, "the pricing warning appends moved; update this guard"
    for expression in appended:
        assert expression.startswith('"') and expression.endswith('"'), (
            f"pricing warning must be a literal, got {expression}"
        )

    list_literals = re.findall(r"pricing_warnings\s*=\s*(\[[^\]]*\])", source)
    assert list_literals, "the pricing_warnings list-literal assignment moved; update this guard"
    for literal in list_literals:
        elements = [item.strip() for item in literal.strip("[]").split(",") if item.strip()]
        for element in elements:
            assert element.startswith('"') and element.endswith('"'), (
                f"pricing warning must be a literal, got {element}"
            )


async def test_a_fully_populated_priced_manifest_still_stamps_verified(
    monkeypatch, keys, patch_call_model
):
    """A priced manifest with every new field populated -- a stale-snapshot
    warning AND an unpriced model/receipt AND a real ceiling on the priced
    receipt -- still stamps VERIFIED (DSE-1514 review, Task 13).

    Exercises the real run path (``Council.ask``) rather than a hand-built
    manifest, so the shape asserted is one ``_price_manifest`` can actually
    produce, not a synthetic one that happens to pass the scan.
    """
    from datetime import date

    from conclave.manifest import SECRET_SAFETY_VERIFIED
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    # grok is priced but the snapshot is stale-dated; claude (the synthesizer)
    # has NO entry at all -- both a stale-snapshot warning and an
    # unpriced-model/unpriced-receipt shape land on the SAME manifest.
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", captured_at=date(2026, 1, 1)))
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok"], synthesizer="claude", extract_verdict=False)
    manifest = (await council.ask("q")).manifest

    assert manifest.cost_ceiling_usd is None  # all-or-nothing: claude is unpriced
    assert manifest.unpriced_models == ["anthropic/claude-sonnet-4-6"]
    assert manifest.unpriced_receipts >= 1
    assert "price_snapshot_stale" in manifest.pricing_warnings
    assert "unpriced_models_present" in manifest.pricing_warnings
    priced = [r for r in manifest.receipts if r.model_id == "xai/grok-4.3"]
    assert priced and all(r.cost_ceiling_usd is not None for r in priced)
    assert manifest.secret_safety == SECRET_SAFETY_VERIFIED
    assert scan_for_secret_material(manifest) is True


@pytest.mark.parametrize("mode", ["synthesize", "raw", "debate", "adversarial", "vote", "elite"])
async def test_pricing_warnings_stay_within_the_closed_vocabulary(
    monkeypatch, keys, patch_call_model, mode
):
    """DSE-1514 review, Task 13: run every mode against every warning-producing
    shape (no snapshot, an unpriced model, a stale snapshot, an uncapped
    failed call) and prove every emitted ``pricing_warnings`` entry is drawn
    from :data:`conclave.council.PRICING_WARNING_VOCABULARY` -- the dynamic
    complement to the static source-scan guard above.
    """
    from datetime import date

    from conclave.council import PRICING_WARNING_VOCABULARY, Council
    from tests.test_pricing_receipts import _install_snapshot, _snapshot

    async def run(council: Council):
        if mode == "debate":
            return await council.debate("q", rounds=2)
        if mode == "adversarial":
            return await council.adversarial("q")
        if mode == "vote":
            return await council.vote("q", choices=["a", "b"])
        if mode == "elite":
            return await council.elite("q")
        return await council.ask("q", synthesize=(mode == "synthesize"))

    seen: set[str] = set()

    # Shape 1: no snapshot at all -> price_snapshot_unavailable.
    _install_snapshot(monkeypatch, None)
    patch_call_model(lambda model_id, messages: make_response("ok"))
    council = Council(models=["grok", "gemini"], synthesizer="claude", extract_verdict=False)
    seen.update((await run(council)).manifest.pricing_warnings)

    # Shape 2: fully priced but stale -> price_snapshot_stale.
    _install_snapshot(
        monkeypatch,
        _snapshot(
            "xai/grok-4.3",
            "gemini/gemini-2.5-pro",
            "anthropic/claude-sonnet-4-6",
            captured_at=date(2026, 1, 1),
        ),
    )
    council = Council(models=["grok", "gemini"], synthesizer="claude", extract_verdict=False)
    seen.update((await run(council)).manifest.pricing_warnings)

    # Shape 3: the synthesizer/judge/chain model has no snapshot entry ->
    # unpriced_models_present (+ unpriced_receipts_present).
    _install_snapshot(monkeypatch, _snapshot("xai/grok-4.3", "gemini/gemini-2.5-pro"))
    council = Council(models=["grok", "gemini"], synthesizer="claude", extract_verdict=False)
    seen.update((await run(council)).manifest.pricing_warnings)

    # Shape 4: every model priced, every call fails with no usage, no output
    # cap -> unpriced_receipts_present + no_output_cap_configured.
    _install_snapshot(
        monkeypatch,
        _snapshot("xai/grok-4.3", "gemini/gemini-2.5-pro", "anthropic/claude-sonnet-4-6"),
    )

    def failing(model_id, messages):
        raise RuntimeError("boom")

    patch_call_model(failing)
    council = Council(models=["grok", "gemini"], synthesizer="claude", extract_verdict=False)
    seen.update((await run(council)).manifest.pricing_warnings)

    assert seen, "no pricing_warnings were ever produced; the fixture setup is stale"
    assert seen <= PRICING_WARNING_VOCABULARY
