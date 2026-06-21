"""Tests for the disagreement-extraction + verdict-synthesis engine (CAC-05).

Authoritative spec: ``03_DESIGN_DECISIONS_v1.1.md`` DD-1 (consensus arithmetic)
and DD-2 (verdict schema + verdict-absent rule). These tests pin the invariants
that make the verdict *auditable*:

* the consensus number is **never** an LLM-emitted value — it is recomputed
  deterministically from the model's per-member clustering via
  :mod:`conclave.agreement` (DD-1);
* the verdict is **optional** — three absent paths return ``verdict=None`` with a
  recorded reason and never raise (DD-2 verdict-absent rule);
* the extractor is mocked entirely offline via the
  ``conclave.verdict_synthesis.call_model`` seam (no network);
* extractor provenance (``model_id`` + ``prompt_version``) is recorded on EVERY
  path.

The extractor call is mocked with a fake whose signature is
``(name, model_id, messages, **kwargs)`` so it absorbs the ``config`` (and any
``temperature``/``timeout``) kwargs the engine threads through ``call_model``.
``_ScriptedExtractor`` returns successive canned payloads so a single test can
exercise the repair-once-then-X paths. Warning assertions use the shared
``conclave_caplog`` fixture (GOTCHA 1 — the ``conclave`` logger has
``propagate=False``, so bare ``caplog`` goes blind once the logger is configured).
"""

from __future__ import annotations

import json

from conclave import agreement
from conclave.manifest import VerdictExtraction
from conclave.models import ModelAnswer
from conclave.verdict import CouncilVerdict
from conclave.verdict_synthesis import (
    VERDICT_EXTRACTION_PROMPT_VERSION,
    VerdictSynthesisResult,
    extract_verdict,
    verdict_extraction_json_schema,
)

# --------------------------------------------------------------------------- #
# Test scaffolding: synthetic member answers + a scripted offline extractor.
# --------------------------------------------------------------------------- #

SYNTH_NAME = "claude"
SYNTH_MODEL_ID = "anthropic/claude-sonnet-4"


def member(name: str, answer: str | None, *, answer_id: str | None = None) -> ModelAnswer:
    """Build a synthetic council-member :class:`ModelAnswer` (test data only)."""
    return ModelAnswer(
        name=name,
        model_id=f"{name}/model",
        answer=answer,
        answer_id=answer_id,
    )


class _ScriptedExtractor:
    """A fake ``call_model`` that returns scripted ``ModelAnswer``s in order.

    Each scripted entry is either a ready :class:`ModelAnswer` or a ``str`` (raw
    JSON / non-JSON text) that is wrapped into a successful ``ModelAnswer``. The
    last entry is reused once the script is exhausted, so a test that only feeds
    one payload never IndexErrors on an unexpected extra call. Every invocation is
    recorded in :attr:`calls` so a test can assert on call count (repair = 2) and
    inspect the messages the engine built.
    """

    def __init__(self, *scripted: ModelAnswer | str) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    async def __call__(self, name, model_id, messages, **kwargs):  # noqa: ANN001
        self.calls.append(
            {"name": name, "model_id": model_id, "messages": messages, "kwargs": kwargs}
        )
        idx = min(len(self.calls) - 1, len(self._scripted) - 1)
        item = self._scripted[idx]
        if isinstance(item, ModelAnswer):
            return item
        return ModelAnswer(name=name, model_id=model_id, answer=item)


def _payload(**overrides) -> dict:
    """A minimal VALID extraction payload, overridable per-field.

    Defaults to a unanimous two-member ``decision`` so a test can mutate exactly
    the field under test. Deliberately omits ``consensus_score`` etc. — the
    extraction schema has no such field (the engine computes it).
    """
    base = {
        "verdict_applies": True,
        "verdict_type": "decision",
        "headline": "Use option A.",
        "recommendation": "Adopt option A for the stated reasons.",
        "positions": [
            {
                "label": "option a",
                "summary": "Prefer A.",
                "providers": ["alpha", "beta"],
                "evidence_answer_ids": ["alpha-1", "beta-1"],
            }
        ],
        "provider_votes": [
            {"provider": "alpha", "position_label": "option a"},
            {"provider": "beta", "position_label": "option a"},
        ],
        "conflicts": [],
        "minority_reports": [],
        "caveats": [],
    }
    base.update(overrides)
    return base


def _install(monkeypatch, fake) -> None:
    """Patch the engine's module-level ``call_model`` seam with ``fake``."""
    monkeypatch.setattr("conclave.verdict_synthesis.call_model", fake)


def _extract(prompt, members):
    """Run the engine with the standard synthesizer identity (async-run helper)."""
    import asyncio

    return asyncio.run(
        extract_verdict(
            prompt,
            members,
            synthesizer_name=SYNTH_NAME,
            synthesizer_model_id=SYNTH_MODEL_ID,
        )
    )


# --------------------------------------------------------------------------- #
# Extraction schema (the structured-output contract sent to the model).
# --------------------------------------------------------------------------- #


def test_extraction_schema_omits_consensus_and_requires_verdict_applies():
    """The schema adds ``verdict_applies`` (required) and NEVER asks for consensus.

    DD-1: the model must never emit the consensus number. So the extraction schema
    (a) requires ``verdict_applies`` and (b) drops every consensus field that
    :func:`conclave.verdict.verdict_json_schema` carries.
    """
    schema = verdict_extraction_json_schema()
    assert schema["additionalProperties"] is False
    assert "verdict_applies" in schema["properties"]
    assert schema["properties"]["verdict_applies"] == {"type": "boolean"}
    assert "verdict_applies" in schema["required"]
    # The model must never emit any consensus field.
    for forbidden in ("consensus_score", "consensus_method", "consensus_label"):
        assert forbidden not in schema["properties"]
        assert forbidden not in schema["required"]
    # The judgment fields the model DOES own are present.
    for kept in ("verdict_type", "headline", "recommendation", "positions"):
        assert kept in schema["properties"]
    # verdict_type stays an enum (LCD: enum, not oneOf/anyOf).
    assert schema["properties"]["verdict_type"]["enum"] == ["decision", "review", "synthesis"]


def test_extraction_schema_is_lcd_compliant():
    """Every object node sets ``additionalProperties: false`` and uses no $ref."""
    serialized = json.dumps(verdict_extraction_json_schema())
    assert "$ref" not in serialized
    assert "oneOf" not in serialized
    assert "anyOf" not in serialized
    assert "allOf" not in serialized


# --------------------------------------------------------------------------- #
# Happy path: verdict assembled; consensus recomputed from clustering.
# --------------------------------------------------------------------------- #


def test_happy_path_assembles_verdict_with_computed_consensus(monkeypatch):
    """A valid extraction yields a verdict; consensus is recomputed from votes."""
    members = [
        member("alpha", "A is best", answer_id="alpha-1"),
        member("beta", "A is best", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    result = _extract("Should we pick A or B?", members)

    assert isinstance(result, VerdictSynthesisResult)
    assert result.verdict_absent_reason is None
    assert isinstance(result.verdict, CouncilVerdict)
    # Exactly one extraction call on the happy path (no repair).
    assert len(fake.calls) == 1
    # The model's judgment fields carry through verbatim.
    assert result.verdict.verdict_type == "decision"
    assert result.verdict.headline == "Use option A."
    # Consensus is recomputed deterministically from the two votes (both "option a").
    expected = agreement.consensus_score(["option a", "option a"])
    assert result.verdict.consensus_score == expected == 1.0
    assert result.verdict.consensus_method == agreement.CONSENSUS_METHOD
    assert result.verdict.consensus_label == "unanimous"


def test_conflicts_carry_evidence_and_computed_per_conflict_scores(monkeypatch):
    """Conflicts cite evidence_answer_ids and carry a recomputed per-conflict score."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
        member("gamma", "no", answer_id="gamma-1"),
    ]
    payload = _payload(
        positions=[
            {
                "label": "yes",
                "summary": "Do it.",
                "providers": ["alpha", "beta"],
                "evidence_answer_ids": ["alpha-1", "beta-1"],
            },
            {
                "label": "no",
                "summary": "Don't.",
                "providers": ["gamma"],
                "evidence_answer_ids": ["gamma-1"],
            },
        ],
        provider_votes=[
            {"provider": "alpha", "position_label": "yes"},
            {"provider": "beta", "position_label": "yes"},
            {"provider": "gamma", "position_label": "no"},
        ],
        conflicts=[
            {
                "topic": "Should we do it?",
                "position_labels": ["yes", "no"],
                "summary": "Two camps.",
            }
        ],
    )
    fake = _ScriptedExtractor(json.dumps(payload))
    _install(monkeypatch, fake)

    result = _extract("Should we do it?", members)

    v = result.verdict
    assert v is not None
    # Top-level consensus = 2/3 over all three votes → majority.
    assert v.consensus_score == agreement.consensus_score(["yes", "yes", "no"])
    assert v.consensus_label == "majority"
    # The conflict cites the position evidence and recomputes its own sub-ratio.
    assert len(v.conflicts) == 1
    conflict = v.conflicts[0]
    assert conflict.position_labels == ["yes", "no"]
    # Per-conflict rule: largest-cluster ratio over the members voting yes/no.
    expected_conflict = agreement.consensus_score(["yes", "yes", "no"])
    assert conflict.consensus_score == expected_conflict
    # Evidence survives on the positions.
    assert v.positions[0].evidence_answer_ids == ["alpha-1", "beta-1"]
    assert v.positions[1].evidence_answer_ids == ["gamma-1"]


def test_model_supplied_consensus_number_is_ignored_and_recomputed(monkeypatch):
    """A bogus consensus_score in the model's JSON is dropped; the real one is computed.

    Hard invariant (DD-1): consensus is NEVER LLM-emitted. Even if the model
    smuggles a ``consensus_score``/``consensus_label`` into its JSON, the
    extraction model has no such field, so Pydantic ignores it (the schema sets
    ``additionalProperties: false`` and the model has no consensus fields), and the
    engine recomputes the genuine value from ``provider_votes``.
    """
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "no", answer_id="beta-1"),
    ]
    # Two members, two different votes → real score 0.5 (split). The model lies 0.99.
    payload = _payload(
        positions=[
            {
                "label": "yes",
                "summary": "Do it.",
                "providers": ["alpha"],
                "evidence_answer_ids": ["alpha-1"],
            },
            {
                "label": "no",
                "summary": "Don't.",
                "providers": ["beta"],
                "evidence_answer_ids": ["beta-1"],
            },
        ],
        provider_votes=[
            {"provider": "alpha", "position_label": "yes"},
            {"provider": "beta", "position_label": "no"},
        ],
    )
    payload["consensus_score"] = 0.99  # the smuggled lie
    payload["consensus_label"] = "strong"  # also a lie
    fake = _ScriptedExtractor(json.dumps(payload))
    _install(monkeypatch, fake)

    result = _extract("yes or no?", members)

    v = result.verdict
    assert v is not None
    # The lie is gone; the genuine recomputed value stands.
    assert v.consensus_score == 0.5
    assert v.consensus_label == "split"


def test_n2_tie_yields_split_label_end_to_end(monkeypatch):
    """N=2 with two distinct positions → score 0.5 → label ``split`` (DD-1)."""
    members = [
        member("alpha", "left", answer_id="alpha-1"),
        member("beta", "right", answer_id="beta-1"),
    ]
    payload = _payload(
        positions=[
            {
                "label": "left",
                "summary": "Go left.",
                "providers": ["alpha"],
                "evidence_answer_ids": ["alpha-1"],
            },
            {
                "label": "right",
                "summary": "Go right.",
                "providers": ["beta"],
                "evidence_answer_ids": ["beta-1"],
            },
        ],
        provider_votes=[
            {"provider": "alpha", "position_label": "left"},
            {"provider": "beta", "position_label": "right"},
        ],
    )
    fake = _ScriptedExtractor(json.dumps(payload))
    _install(monkeypatch, fake)

    result = _extract("left or right?", members)

    assert result.verdict is not None
    assert result.verdict.consensus_score == 0.5
    assert result.verdict.consensus_label == "split"


def test_member_with_no_vote_is_excluded_from_consensus(monkeypatch):
    """A responding member without a provider_vote contributes a None to the sequence.

    DD-1 excludes no-stance members from the denominator. The engine maps each
    responding member to its vote (or ``None``); ``consensus_score`` drops the
    ``None`` so the denominator is only positioned members.
    """
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
        member("gamma", "i abstain", answer_id="gamma-1"),  # responds, no vote
    ]
    payload = _payload(
        positions=[
            {
                "label": "yes",
                "summary": "Do it.",
                "providers": ["alpha", "beta"],
                "evidence_answer_ids": ["alpha-1", "beta-1"],
            }
        ],
        # gamma responded but cast no vote → excluded from the denominator.
        provider_votes=[
            {"provider": "alpha", "position_label": "yes"},
            {"provider": "beta", "position_label": "yes"},
        ],
    )
    fake = _ScriptedExtractor(json.dumps(payload))
    _install(monkeypatch, fake)

    result = _extract("do it?", members)

    v = result.verdict
    assert v is not None
    # Denominator is 2 (alpha, beta), not 3 → unanimous, not 2/3.
    assert v.consensus_score == 1.0
    assert v.consensus_label == "unanimous"


# --------------------------------------------------------------------------- #
# Absent path 1: N<2 responding members (no LLM call).
# --------------------------------------------------------------------------- #


def test_absent_when_fewer_than_two_responding_members(monkeypatch):
    """N<2 → verdict None, reason recorded, NO extraction call made."""
    members = [
        member("alpha", "only answer", answer_id="alpha-1"),
        member("beta", None, answer_id="beta-1"),  # failed/empty
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    result = _extract("anything", members)

    assert result.verdict is None
    assert result.verdict_absent_reason == "fewer than 2 responding members"
    # No LLM call on the gate path.
    assert len(fake.calls) == 0
    # Provenance still recorded.
    assert result.extraction.model_id == SYNTH_MODEL_ID
    assert result.extraction.prompt_version == VERDICT_EXTRACTION_PROMPT_VERSION


def test_absent_when_blank_answers_dont_count(monkeypatch):
    """Whitespace-only answers do not count toward the N>=2 gate."""
    members = [
        member("alpha", "   ", answer_id="alpha-1"),
        member("beta", "\n\t", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    result = _extract("anything", members)

    assert result.verdict is None
    assert result.verdict_absent_reason == "fewer than 2 responding members"
    assert len(fake.calls) == 0


# --------------------------------------------------------------------------- #
# Absent path 2: verdict_applies == False (open-ended prompt).
# --------------------------------------------------------------------------- #


def test_absent_when_verdict_does_not_apply(monkeypatch):
    """verdict_applies=false → open-ended; verdict None with the open-ended reason."""
    members = [
        member("alpha", "a poem", answer_id="alpha-1"),
        member("beta", "another poem", answer_id="beta-1"),
    ]
    payload = _payload(verdict_applies=False)
    fake = _ScriptedExtractor(json.dumps(payload))
    _install(monkeypatch, fake)

    result = _extract("Write me a poem.", members)

    assert result.verdict is None
    assert result.verdict_absent_reason == "open-ended prompt (no decision/review to adjudicate)"
    # The extraction DID run (one call) and provenance is recorded.
    assert len(fake.calls) == 1
    assert result.extraction.model_id == SYNTH_MODEL_ID
    assert result.extraction.prompt_version == VERDICT_EXTRACTION_PROMPT_VERSION


# --------------------------------------------------------------------------- #
# Absent path 3: extraction fails schema validation after one repair.
# --------------------------------------------------------------------------- #


def test_absent_when_extraction_fails_after_repair(monkeypatch, conclave_caplog):
    """Two bad payloads (invalid JSON, then schema-invalid) → fallback, no raise."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    # First call: not JSON at all. Repair call: JSON but missing required fields.
    fake = _ScriptedExtractor("this is not json", json.dumps({"verdict_applies": True}))
    _install(monkeypatch, fake)

    result = _extract("yes or no?", members)

    assert result.verdict is None
    assert result.verdict_absent_reason == "verdict extraction failed schema validation"
    # Exactly two calls: original + one repair.
    assert len(fake.calls) == 2
    # Provenance recorded even on failure.
    assert result.extraction.model_id == SYNTH_MODEL_ID
    assert result.extraction.prompt_version == VERDICT_EXTRACTION_PROMPT_VERSION
    # A warning was logged (GOTCHA 1: use conclave_caplog, not bare caplog).
    assert any("verdict extraction" in r.message.lower() for r in conclave_caplog.records)


def test_repair_message_includes_validation_errors(monkeypatch):
    """The repair call appends the stringified validation errors to the messages."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor("not json", json.dumps({"verdict_applies": True}))
    _install(monkeypatch, fake)

    _extract("yes or no?", members)

    # The second (repair) call carries strictly more messages than the first, and
    # the appended content references the parse/validation failure.
    first_msgs = fake.calls[0]["messages"]
    repair_msgs = fake.calls[1]["messages"]
    assert len(repair_msgs) > len(first_msgs)
    repair_text = repair_msgs[-1]["content"].lower()
    assert "valid json" in repair_text or "error" in repair_text


def test_repair_once_then_success(monkeypatch):
    """First payload invalid, second valid → verdict assembled after one repair."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor("garbage, not json", json.dumps(_payload()))
    _install(monkeypatch, fake)

    result = _extract("yes or no?", members)

    assert result.verdict is not None
    assert result.verdict_absent_reason is None
    # Original + one repair = exactly two calls.
    assert len(fake.calls) == 2
    assert result.verdict.consensus_label == "unanimous"


def test_absent_when_extractor_returns_error(monkeypatch, conclave_caplog):
    """An extractor ModelAnswer with .error (no answer) → fallback after repair."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    errored = ModelAnswer(name=SYNTH_NAME, model_id=SYNTH_MODEL_ID, error="upstream 500")
    fake = _ScriptedExtractor(errored, errored)
    _install(monkeypatch, fake)

    result = _extract("yes or no?", members)

    assert result.verdict is None
    assert result.verdict_absent_reason == "verdict extraction failed schema validation"
    assert len(fake.calls) == 2
    assert any("verdict extraction" in r.message.lower() for r in conclave_caplog.records)


def test_absent_when_extractor_returns_empty_answer(monkeypatch):
    """An extractor answer that is empty/whitespace → treated as a parse failure."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    empty = ModelAnswer(name=SYNTH_NAME, model_id=SYNTH_MODEL_ID, answer="   ")
    fake = _ScriptedExtractor(empty, empty)
    _install(monkeypatch, fake)

    result = _extract("yes or no?", members)

    assert result.verdict is None
    assert result.verdict_absent_reason == "verdict extraction failed schema validation"
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- #
# Provenance + messaging invariants on every path.
# --------------------------------------------------------------------------- #


def test_provenance_recorded_on_every_path(monkeypatch):
    """extraction.model_id / .prompt_version are populated on success and absence."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)
    success = _extract("decide?", members)
    assert success.extraction == VerdictExtraction(
        model_id=SYNTH_MODEL_ID, prompt_version=VERDICT_EXTRACTION_PROMPT_VERSION
    )

    # Absent (N<2) path still records provenance.
    absent = _extract("decide?", [member("alpha", "only", answer_id="alpha-1")])
    assert absent.extraction.model_id == SYNTH_MODEL_ID
    assert absent.extraction.prompt_version == VERDICT_EXTRACTION_PROMPT_VERSION


def test_messages_label_member_answers_with_answer_ids(monkeypatch):
    """The extraction prompt labels each member answer with its stable answer_id."""
    members = [
        member("alpha", "first answer text", answer_id="alpha-7"),
        member("beta", "second answer text", answer_id="beta-9"),
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    _extract("decide?", members)

    user_content = fake.calls[0]["messages"][-1]["content"]
    assert "alpha-7" in user_content
    assert "beta-9" in user_content
    assert "first answer text" in user_content
    assert "second answer text" in user_content


def test_messages_fall_back_to_name_when_answer_id_missing(monkeypatch):
    """A responding member with answer_id=None is labeled by a stable fallback."""
    members = [
        member("alpha", "answer one", answer_id=None),
        member("beta", "answer two", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    _extract("decide?", members)

    user_content = fake.calls[0]["messages"][-1]["content"]
    # The fallback label (the member name) appears so the model can still cite it.
    assert "alpha" in user_content
    assert "answer one" in user_content


def test_system_prompt_instructs_no_consensus_number(monkeypatch):
    """The system prompt explicitly tells the model NOT to emit a consensus score."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    _extract("decide?", members)

    system_content = fake.calls[0]["messages"][0]["content"].lower()
    assert fake.calls[0]["messages"][0]["role"] == "system"
    assert "consensus" in system_content


def test_only_responding_members_appear_in_messages(monkeypatch):
    """A failed member (answer=None) is not included in the extraction prompt."""
    members = [
        member("alpha", "real answer", answer_id="alpha-1"),
        member("beta", "real answer two", answer_id="beta-1"),
        member("gamma", None, answer_id="gamma-1"),  # failed — excluded
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    _extract("decide?", members)

    user_content = fake.calls[0]["messages"][-1]["content"]
    assert "gamma-1" not in user_content


def test_config_threaded_through_to_call_model(monkeypatch):
    """A passed config reaches call_model (kwarg seam absorbs it)."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    fake = _ScriptedExtractor(json.dumps(_payload()))
    _install(monkeypatch, fake)

    import asyncio

    sentinel = object()
    asyncio.run(
        extract_verdict(
            "decide?",
            members,
            synthesizer_name=SYNTH_NAME,
            synthesizer_model_id=SYNTH_MODEL_ID,
            config=sentinel,
        )
    )
    assert fake.calls[0]["kwargs"].get("config") is sentinel


def test_engine_extracts_json_from_fenced_code_block(monkeypatch):
    """A model that wraps its JSON in a ```json fence still parses (prompt-level SO)."""
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    fenced = f"```json\n{json.dumps(_payload())}\n```"
    fake = _ScriptedExtractor(fenced)
    _install(monkeypatch, fake)

    result = _extract("decide?", members)

    assert result.verdict is not None
    assert result.verdict.consensus_label == "unanimous"
    # Parsed on the first call — fenced JSON is not a failure.
    assert len(fake.calls) == 1


def test_fenced_block_without_braces_falls_through_to_repair(monkeypatch):
    """A code fence with no JSON object inside still routes to the repair path.

    Guards the defensive fall-through in the fence stripper: a fence whose body has
    no ``{...}`` is returned as-is, fails JSON parsing, and triggers repair rather
    than mis-extracting.
    """
    members = [
        member("alpha", "yes", answer_id="alpha-1"),
        member("beta", "yes", answer_id="beta-1"),
    ]
    # First: a fence with no braces at all → parse failure. Repair: valid JSON.
    fake = _ScriptedExtractor("```\nno json here\n```", json.dumps(_payload()))
    _install(monkeypatch, fake)

    result = _extract("decide?", members)

    assert result.verdict is not None
    assert len(fake.calls) == 2
