"""Tests for the CAC-01 result contract v2: verdict/member types + LCD schemas.

These tests pin DD-2 (verdict + member schema) and the lowest-common-denominator
JSON-Schema constraints, the additive/backward-compatible extensions to
``ModelAnswer`` and ``CouncilResult``, and the secret-safety invariant (no key
material in any new type or schema). All tests run offline; no keys required.
"""

from __future__ import annotations

import pytest

from conclave import (
    CouncilResult,
    CouncilVerdict,
    ModelAnswer,
    member_answer_json_schema,
    verdict_json_schema,
)
from conclave.verdict import (
    CONFIDENCE_LEVELS,
    CONSENSUS_METHOD,
    VERDICT_SCHEMA_VERSION,
    VERDICT_TYPES,
    CouncilConflict,
    CouncilPosition,
    MinorityReport,
    ProviderVote,
)

# Keys that, per the LCD constraint (DD-2), must never appear anywhere in either
# schema dict: choice via enum/nullable only, no $ref/recursion, no conditional.
_FORBIDDEN_SCHEMA_KEYS = frozenset(
    {
        "oneOf",
        "anyOf",
        "allOf",
        "$ref",
        "$defs",
        "if",
        "then",
        "dependentRequired",
    }
)


def _walk_objects(node, depth=1):
    """Yield ``(object_node, depth)`` for every JSON-Schema object node.

    ``depth`` counts object nesting: the root object is depth 1, an object that is
    the ``items`` of an array property of the root is depth 2, and so on. Strings
    living inside a depth-3 object's array property are *not* objects and do not
    increase the object depth.
    """
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node, depth
            child_depth = depth + 1
        else:
            child_depth = depth
        for value in node.values():
            yield from _walk_objects(value, child_depth)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_objects(item, depth)


def _assert_no_forbidden_keys(node):
    """Assert no LCD-forbidden key name appears anywhere in ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            assert key not in _FORBIDDEN_SCHEMA_KEYS, f"forbidden schema key: {key!r}"
            _assert_no_forbidden_keys(value)
    elif isinstance(node, list):
        for item in node:
            _assert_no_forbidden_keys(item)


def _collect_enums(node):
    """Yield every ``enum`` value list found anywhere in ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "enum":
                yield value
            else:
                yield from _collect_enums(value)
    elif isinstance(node, list):
        for item in node:
            yield from _collect_enums(item)


@pytest.mark.parametrize("schema_fn", [verdict_json_schema, member_answer_json_schema])
def test_schema_is_well_formed(schema_fn):
    """Each schema is a top-level object with properties, required, and closed."""
    schema = schema_fn()
    assert schema["type"] == "object"
    assert isinstance(schema.get("properties"), dict) and schema["properties"]
    assert isinstance(schema.get("required"), list)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("schema_fn", [verdict_json_schema, member_answer_json_schema])
def test_lcd_constraints_hold(schema_fn):
    """LCD: no forbidden keys, every object closed, depth ≤ 3, enums are non-empty strings."""
    schema = schema_fn()

    # (a) no oneOf/anyOf/allOf/$ref/$defs/if/then/dependentRequired anywhere.
    _assert_no_forbidden_keys(schema)

    # (b) every object node sets additionalProperties: false, and (c) depth ≤ 3.
    object_nodes = list(_walk_objects(schema))
    assert object_nodes, "expected at least the root object node"
    for obj, depth in object_nodes:
        assert obj["additionalProperties"] is False
        assert depth <= 3, f"object nesting depth {depth} exceeds LCD max of 3"

    # (d) every enum is a non-empty list of strings.
    for enum_values in _collect_enums(schema):
        assert isinstance(enum_values, list) and enum_values
        assert all(isinstance(v, str) for v in enum_values)


def test_enum_values_match_dd2():
    """The two DD-2 enums carry exactly the spec values."""
    member = member_answer_json_schema()
    verdict = verdict_json_schema()
    assert member["properties"]["confidence"]["enum"] == ["low", "medium", "high"]
    assert verdict["properties"]["verdict_type"]["enum"] == ["decision", "review", "synthesis"]


def test_member_schema_shape():
    """Member schema: only key_points required, position nullable, no answer_id."""
    member = member_answer_json_schema()
    assert member["title"] == "MemberAnswer"
    assert member["required"] == ["key_points"]
    assert member["properties"]["position"]["type"] == ["string", "null"]
    assert "answer_id" not in member["properties"]
    # key_points is an array of strings.
    assert member["properties"]["key_points"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_verdict_required_set_is_exactly_dd2():
    """Verdict required == the 7 DD-2 required fields; optionals absent from required."""
    verdict = verdict_json_schema()
    assert verdict["title"] == "CouncilVerdict"
    assert set(verdict["required"]) == {
        "verdict_type",
        "headline",
        "recommendation",
        "consensus_score",
        "consensus_method",
        "consensus_label",
        "positions",
    }
    optional_fields = {
        "conflicts",
        "provider_votes",
        "minority_reports",
        "caveats",
        "dissent_summary",
    }
    for field in optional_fields:
        assert field in verdict["properties"], f"{field} should be in properties"
        assert field not in verdict["required"], f"{field} should NOT be required"
    # consensus_score nullable at the top level (DD-1: float|null).
    assert verdict["properties"]["consensus_score"]["type"] == ["number", "null"]


def test_council_verdict_construction():
    """A minimal valid verdict constructs and is self-stamping with method default."""
    verdict = CouncilVerdict(
        verdict_type="decision",
        headline="Use explicit refresh only.",
        recommendation="Add an explicit refresh endpoint; do not auto-refresh.",
    )
    assert verdict.schema_version == "1"
    assert VERDICT_SCHEMA_VERSION == "1"
    assert verdict.consensus_method == "position_cluster_ratio_v1"
    assert verdict.consensus_method == CONSENSUS_METHOD
    # Optional collections default empty; optional scalars default None.
    assert verdict.positions == []
    assert verdict.conflicts == []
    assert verdict.provider_votes == []
    assert verdict.minority_reports == []
    assert verdict.caveats == []
    assert verdict.consensus_score is None
    assert verdict.consensus_label is None
    assert verdict.dissent_summary is None
    # All three verdict_type values are accepted.
    assert VERDICT_TYPES == ("decision", "review", "synthesis")
    for vt in VERDICT_TYPES:
        v = CouncilVerdict(verdict_type=vt, headline="h", recommendation="r")
        assert v.verdict_type == vt


def test_verdict_absent_defaults_on_result():
    """A bare CouncilResult carries no verdict and empty adjudication collections."""
    result = CouncilResult(prompt="x")
    assert result.verdict is None
    assert result.consensus_score is None
    assert result.consensus_method is None
    assert result.consensus_label is None
    assert result.conflicts == []
    assert result.provider_votes == []
    assert result.minority_reports == []
    # member_answers aliases answers (both empty here).
    assert result.member_answers == []
    assert result.member_answers == result.answers


def test_model_answer_backward_compat():
    """ModelAnswer constructs with no new args; new fields default; latency_ms derives."""
    answer = ModelAnswer(name="grok", model_id="xai/grok-4.3", answer="hi")
    assert answer.answer_id is None
    assert answer.warnings == []
    assert answer.ok is True

    answer.latency_s = 0.25
    assert answer.latency_ms == 250.0

    # CouncilResult still auto-stamps prompt_version with no new args.
    result = CouncilResult(prompt="x")
    assert isinstance(result.prompt_version, str) and result.prompt_version


def test_construction_degrades_gracefully_n1_n2_partial():
    """N=1 / N=2 / partial-failure all CONSTRUCT (fields only, no computation)."""
    ok1 = ModelAnswer(name="grok", model_id="xai/grok-4.3", answer="a")
    ok2 = ModelAnswer(name="claude", model_id="anthropic/claude-sonnet-4-6", answer="b")
    errd = ModelAnswer(name="gemini", model_id="gemini/gemini-2.5-pro", error="boom")

    # N=1: single ok answer, no consensus.
    n1 = CouncilResult(prompt="x", answers=[ok1], consensus_score=None)
    assert len(n1.successful_answers) == 1
    assert n1.failed_answers == []
    assert n1.consensus_score is None
    assert n1.conflicts == [] and n1.provider_votes == [] and n1.minority_reports == []

    # N=2: two ok answers.
    n2 = CouncilResult(prompt="x", answers=[ok1, ok2])
    assert len(n2.successful_answers) == 2
    assert n2.failed_answers == []
    assert n2.conflicts == [] and n2.provider_votes == [] and n2.minority_reports == []

    # Partial failure: 1 ok + 1 errored.
    partial = CouncilResult(prompt="x", answers=[ok1, errd])
    assert len(partial.successful_answers) == 1
    assert len(partial.failed_answers) == 1
    assert partial.failed_answers[0].name == "gemini"
    assert partial.conflicts == []


def test_member_answers_is_answers_alias():
    """member_answers returns the same list contents/object as answers."""
    a = ModelAnswer(name="grok", model_id="xai/grok-4.3", answer="hi")
    result = CouncilResult(prompt="x", answers=[a])
    assert result.member_answers == result.answers
    assert result.member_answers is result.answers
    assert result.member_answers[0] is a


def test_optional_types_construct_and_populate():
    """The optional verdict sub-types construct and attach to a verdict + result."""
    position = CouncilPosition(
        label="explicit only",
        summary="No auto-refresh; explicit endpoint.",
        providers=["anthropic", "openai"],
        evidence_answer_ids=["anthropic-1", "openai-1"],
    )
    conflict = CouncilConflict(
        topic="auto-refresh",
        position_labels=["explicit only", "auto-refresh"],
        summary="Disagreement on automatic token refresh.",
        consensus_score=0.5,
    )
    vote = ProviderVote(provider="gemini", position_label="explicit only", confidence="medium")
    minority = MinorityReport(
        providers=["gemini"],
        claim="Auto-refresh is acceptable with rotation.",
        evidence_answer_ids=["gemini-1"],
        why_it_matters="Affects UX latency.",
    )
    verdict = CouncilVerdict(
        verdict_type="review",
        headline="Explicit refresh recommended.",
        recommendation="Ship explicit refresh.",
        consensus_score=0.67,
        consensus_label="majority",
        positions=[position],
        conflicts=[conflict],
        provider_votes=[vote],
        minority_reports=[minority],
        caveats=["Revisit if rotation lands."],
        dissent_summary="One member favors auto-refresh.",
    )
    assert vote.confidence in CONFIDENCE_LEVELS
    result = CouncilResult(
        prompt="x",
        answers=[ModelAnswer(name="claude", model_id="anthropic/c", answer="hi")],
        verdict=verdict,
        consensus_score=0.67,
        consensus_method=CONSENSUS_METHOD,
        consensus_label="majority",
        conflicts=[conflict],
        provider_votes=[vote],
        minority_reports=[minority],
    )
    assert result.verdict is verdict
    assert result.consensus_label == "majority"
    assert result.conflicts[0].consensus_score == 0.5


def test_no_secret_material_in_schemas_or_verdict():
    """Schemas + a constructed verdict carry no key-like material (secret-safety)."""
    blob = (
        str(verdict_json_schema())
        + str(member_answer_json_schema())
        + CouncilVerdict(
            verdict_type="decision",
            headline="h",
            recommendation="r",
        ).model_dump_json()
    )
    lowered = blob.lower()
    assert "sk-" not in blob
    assert "api_key" not in lowered
    assert "authorization" not in lowered
    assert "bearer" not in lowered
