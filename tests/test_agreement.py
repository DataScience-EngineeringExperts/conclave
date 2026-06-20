"""Tests for the CAC-02 deterministic agreement engine + adapter output-contract.

Two pinned contracts:

* ``conclave.agreement`` -- ``position_cluster_ratio_v1`` (DD-1): the ratio
  arithmetic, the label buckets and their exact boundaries, null-position
  exclusion, conditional-counts, case/whitespace-insensitive grouping, and the
  auditability-paradox guard (``00_SCOPE_PLAN.md`` §4.1: no difflib).
* The :class:`conclave.adapters.base.OutputContract` no-op pass-through on all
  three concrete adapters -- accepting the optional trailing param must not alter
  the request body today (provider-native translation is deferred to
  CAC-02-OAI/ANT/GEM).

All tests run offline; no keys required. Existing ``tests/test_adapters.py`` is
left untouched.
"""

from __future__ import annotations

import pytest

from conclave import agreement
from conclave.adapters.anthropic import AnthropicAdapter
from conclave.adapters.base import OutputContract
from conclave.adapters.gemini import GeminiAdapter
from conclave.adapters.openai_compat import OpenAICompatAdapter

# --------------------------------------------------------------------------- #
# consensus_score: ratio arithmetic (DD-1 position_cluster_ratio_v1)
# --------------------------------------------------------------------------- #


def test_consensus_score_three_of_four_is_0_75():
    assert agreement.consensus_score(["yes", "yes", "yes", "no"]) == 0.75


def test_consensus_score_two_of_three_is_two_thirds():
    score = agreement.consensus_score(["yes", "yes", "no"])
    assert score == pytest.approx(2 / 3)


def test_consensus_score_four_of_four_is_unanimous_ratio():
    assert agreement.consensus_score(["yes", "yes", "yes", "yes"]) == 1.0


def test_consensus_score_unanimous_all_same():
    # Any all-agree list -> ratio 1.0 regardless of cluster value.
    assert agreement.consensus_score(["maybe", "maybe", "maybe"]) == 1.0


def test_consensus_score_n1_is_none():
    # A lone vote is not consensus (N<2 -> undefined).
    assert agreement.consensus_score(["yes"]) is None


def test_consensus_score_empty_is_none():
    assert agreement.consensus_score([]) is None


def test_consensus_score_n2_tie_is_half():
    # Two distinct clusters, 1 each -> 0.5 (a tie, never "50% consensus").
    assert agreement.consensus_score(["yes", "no"]) == 0.5


# --------------------------------------------------------------------------- #
# consensus_score: null-position exclusion, conditional counts, normalization
# --------------------------------------------------------------------------- #


def test_null_positions_excluded_from_denominator():
    # The single None is dropped; the 2 remaining agree -> 2/2 = 1.0.
    assert agreement.consensus_score(["yes", "yes", None]) == 1.0


def test_only_nulls_is_none():
    # All members abstained -> nothing positioned -> undefined.
    assert agreement.consensus_score([None, None, None]) is None


def test_one_position_among_nulls_is_none():
    # After dropping nulls only one positioned member remains -> N<2 -> None.
    assert agreement.consensus_score(["yes", None, None]) is None


def test_conditional_counts_in_denominator():
    # "conditional" is an ordinary non-null cluster: largest "yes"=2, total=3.
    score = agreement.consensus_score(["conditional", "yes", "yes"])
    assert score == pytest.approx(2 / 3)
    assert agreement.consensus_label(score) == "majority"


def test_it_depends_counts_in_denominator():
    # "it depends" is likewise a valid position cluster, not an abstention.
    score = agreement.consensus_score(["it depends", "it depends", "yes"])
    assert score == pytest.approx(2 / 3)


def test_case_and_whitespace_insensitive_grouping():
    # "Yes" / " yes " / "YES" all normalize to the same cluster -> 3/3 = 1.0.
    assert agreement.consensus_score(["Yes", " yes ", "YES"]) == 1.0


def test_internal_whitespace_collapsed_in_grouping():
    # Internal whitespace runs collapse so cosmetic spacing differences group.
    assert agreement.consensus_score(["explicit  refresh", "explicit refresh"]) == 1.0


# --------------------------------------------------------------------------- #
# _normalize_label: the single-source grouping rule
# --------------------------------------------------------------------------- #


def test_normalize_label_casefolds_strips_and_collapses():
    assert agreement._normalize_label("  Explicit   Refresh ") == "explicit refresh"


# --------------------------------------------------------------------------- #
# consensus_label: DD-1 buckets + exact boundary handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "score, expected",
    [
        (None, "none"),  # no positioned members / N<2
        (1.0, "unanimous"),  # checked before strong: perfect != merely strong
        (0.76, "strong"),  # 0.75 <= score < 1.0
        (0.75, "strong"),  # INCLUSIVE lower bound for strong
        (0.74, "majority"),  # 0.5 < score < 0.75 (just under the strong floor)
        (0.51, "majority"),  # just above the majority floor
        (0.5, "split"),  # EXACTLY 0.5 -> split (majority is > 0.5, exclusive)
        (0.49, "split"),  # below 0.5
    ],
)
def test_consensus_label_boundaries(score, expected):
    assert agreement.consensus_label(score) == expected


def test_consensus_label_strong_is_inclusive_at_0_75():
    # Pin the inclusive-strong / exclusive-majority boundary explicitly.
    assert agreement.consensus_label(0.75) == "strong"
    assert agreement.consensus_label(0.7499) == "majority"


def test_consensus_label_one_point_zero_is_unanimous_not_strong():
    assert agreement.consensus_label(1.0) == "unanimous"


# --------------------------------------------------------------------------- #
# End-to-end score -> label (the cases DD-1 calls out by name)
# --------------------------------------------------------------------------- #


def test_n2_tie_scores_half_labels_split():
    score = agreement.consensus_score(["yes", "no"])
    assert score == 0.5
    assert agreement.consensus_label(score) == "split"


def test_three_of_four_labels_strong():
    score = agreement.consensus_score(["yes", "yes", "yes", "no"])
    assert score == 0.75
    assert agreement.consensus_label(score) == "strong"


def test_unanimous_end_to_end():
    score = agreement.consensus_score(["yes", "yes"])
    assert score == 1.0
    assert agreement.consensus_label(score) == "unanimous"


def test_none_score_labels_none_end_to_end():
    score = agreement.consensus_score(["yes"])
    assert score is None
    assert agreement.consensus_label(score) == "none"


def test_consensus_convenience_returns_score_and_label():
    assert agreement.consensus(["yes", "yes", "yes", "no"]) == (0.75, "strong")
    assert agreement.consensus(["yes"]) == (None, "none")


# --------------------------------------------------------------------------- #
# Method literal + auditability-paradox guard (§4.1: no difflib)
# --------------------------------------------------------------------------- #


def test_consensus_method_literal_matches_verdict_source_of_truth():
    assert agreement.CONSENSUS_METHOD == "position_cluster_ratio_v1"


def test_agreement_module_does_not_import_difflib():
    """Auditability paradox (§4.1): the deterministic engine must NOT use difflib.

    difflib's ``SequenceMatcher`` ratio is the debate ``convergence_score`` (text
    stability), a FORBIDDEN consensus measure -- using it here would conflate
    "the answers read similarly" with "the council agrees". The robust assertion
    parses the module's own source for an actual IMPORT statement (a bare mention
    of the word in a docstring -- e.g. the module's own "does not import difflib"
    note -- is fine; checking ``sys.modules`` would be flaky since other modules
    legitimately import difflib). We assert no ``import difflib`` / ``from difflib``
    line exists, and that the names never landed in the module namespace.
    """
    with open(agreement.__file__, encoding="utf-8") as fh:
        import_lines = [
            line.strip() for line in fh if line.lstrip().startswith(("import ", "from "))
        ]
    assert not any("difflib" in line for line in import_lines)
    assert not any("SequenceMatcher" in line for line in import_lines)
    # Belt-and-suspenders: the names never leaked into the module namespace.
    assert not hasattr(agreement, "difflib")
    assert not hasattr(agreement, "SequenceMatcher")


# --------------------------------------------------------------------------- #
# CAC-02 adapter contract: OutputContract is a no-op pass-through today
# --------------------------------------------------------------------------- #


def _openai_adapter() -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        prefix="openai",
        completions_url="https://api.openai.com/v1/chat/completions",
        env_vars=("OPENAI_API_KEY",),
    )


# (adapter, model_id, key) tuples covering all three concrete adapters.
_ADAPTER_CASES = [
    (_openai_adapter(), "openai/gpt-4.1", "sk-secret"),
    (AnthropicAdapter(), "anthropic/claude-sonnet-4-6", "sk-ant-secret"),
    (GeminiAdapter(), "gemini/gemini-2.5-pro", "AIza-secret"),
]

_MESSAGES = [{"role": "user", "content": "hi"}]
_REAL_CONTRACT = OutputContract(
    schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    schema_name="MemberAnswer",
    strict=True,
)


@pytest.mark.parametrize("adapter, model_id, api_key", _ADAPTER_CASES)
def test_build_request_output_contract_is_noop_passthrough(adapter, model_id, api_key):
    """Accepting output_contract must not change the built body today (no-op).

    Three call forms must produce identical bodies: no arg, explicit ``None``,
    and a real :class:`OutputContract`. This proves the param is accepted and
    ignored (provider-native translation deferred to CAC-02-OAI/ANT/GEM).
    """
    _u0, _h0, body_default = adapter.build_request(model_id, _MESSAGES, 0.7, 120.0, api_key)
    _u1, _h1, body_none = adapter.build_request(
        model_id, _MESSAGES, 0.7, 120.0, api_key, output_contract=None
    )
    _u2, _h2, body_real = adapter.build_request(
        model_id, _MESSAGES, 0.7, 120.0, api_key, output_contract=_REAL_CONTRACT
    )
    assert body_none == body_default
    assert body_real == body_default


@pytest.mark.parametrize("adapter, model_id, api_key", _ADAPTER_CASES)
def test_stream_request_output_contract_is_noop_passthrough(adapter, model_id, api_key):
    """Same no-op guarantee on the streaming path (param wired through, ignored)."""
    _u0, _h0, body_default = adapter.stream_request(model_id, _MESSAGES, 0.7, 120.0, api_key)
    _u1, _h1, body_none = adapter.stream_request(
        model_id, _MESSAGES, 0.7, 120.0, api_key, output_contract=None
    )
    _u2, _h2, body_real = adapter.stream_request(
        model_id, _MESSAGES, 0.7, 120.0, api_key, output_contract=_REAL_CONTRACT
    )
    assert body_none == body_default
    assert body_real == body_default


def test_output_contract_schema_field_roundtrips():
    """The pydantic ``schema`` field-name collision is resolved (constructs + reads)."""
    contract = OutputContract(schema={"type": "object"})
    assert contract.schema == {"type": "object"}
    # Defaults hold, including repair_attempts == 1.
    assert contract.schema_name is None
    assert contract.strict is False
    assert contract.repair_attempts == 1


def test_output_contract_defaults_all_none_means_no_structured_output():
    contract = OutputContract()
    assert contract.schema is None
    assert contract.schema_name is None
    assert contract.strict is False
    assert contract.repair_attempts == 1
