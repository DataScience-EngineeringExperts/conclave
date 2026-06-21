"""Verdict + member structured-output schema (Result contract v2 — CAC-01).

This module is the single source of truth for the *adjudication* shapes that sit
on top of the raw :class:`conclave.models.ModelAnswer` fan-out:

* the Pydantic types that a downstream consumer reads off
  :class:`conclave.models.CouncilResult` (``verdict``, ``conflicts``,
  ``provider_votes``, ``minority_reports``), and
* the two fixed JSON-Schema dicts that are *sent to providers* to drive
  structured output (:func:`member_answer_json_schema` for each council
  member's answer, :func:`verdict_json_schema` for the synthesized verdict).

Authoritative spec: ``03_DESIGN_DECISIONS_v1.1.md`` DD-2 (verdict + member
schema) and DD-1 (``consensus_score`` method). This module implements the
*shapes* only — it computes nothing. The consensus arithmetic
(``position_cluster_ratio_v1``) and the verdict-extraction step are CAC-05's job;
the manifest is CAC-04's.

LCD constraint (DD-2): the JSON Schemas here are the lowest common denominator of
the three adapter surfaces (OpenAI ``json_schema``, Gemini ``responseSchema``,
Anthropic tool ``input_schema``). They therefore stay shallow (object nesting
≤ 3), express choice as ``enum`` (never ``oneOf``/``anyOf``/``allOf``), avoid
``$ref``/``$defs``/recursion, set ``additionalProperties: false`` on every object,
and make optionality nullable-or-omitted (never conditional-required). The
position object shape is inlined in both the verdict's ``positions`` and each
``conflict`` rather than ``$ref``-shared, on purpose.

This module deliberately does NOT import :mod:`conclave.models`; the dependency
runs the other way (``models`` imports the verdict types) so there is no cycle.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Schema/version stamps. These mirror the existing ``SYNTHESIS_PROMPT_VERSION``
# pattern: a constructed verdict is self-stamping so a downstream eval/regression
# suite can detect a schema bump rather than silently mis-reading an old shape.
VERDICT_SCHEMA_VERSION = "1"

# DD-1 names the deterministic consensus method. Exposed as a module constant so
# callers and CAC-05 share the exact literal instead of re-typing it.
CONSENSUS_METHOD = "position_cluster_ratio_v1"

# The three valid ``verdict_type`` discriminator values (DD-2). Stored on the
# model as a plain ``str`` (validated against this set) and emitted as a JSON
# Schema ``enum`` in :func:`verdict_json_schema`.
VERDICT_TYPES = ("decision", "review", "synthesis")

# Self-reported member confidence enum (DD-2). Recorded, never used in the
# consensus arithmetic (DD-1: self-reported confidence is unreliable).
CONFIDENCE_LEVELS = ("low", "medium", "high")

# Version tag for the verdict-EXTRACTION prompt (CAC-05), distinct from the
# synthesis prompt version (``conclave.prompts.SYNTHESIS_PROMPT_VERSION``). It is
# recorded in the manifest's ``verdict_extraction`` provenance so a downstream
# audit can tell WHICH extractor wording produced a given clustering. Opaque
# string; only equality/inequality is meaningful. Bump on any change to the
# extraction system prompt in :mod:`conclave.verdict_synthesis`.
VERDICT_EXTRACTION_PROMPT_VERSION = "1"


class CouncilPosition(BaseModel):
    """One clustered stance in the verdict (a ``positions[]`` element).

    The verdict-extraction step (CAC-05) clusters semantically-equivalent member
    positions into these. This is the element shape reused by both the verdict's
    ``positions`` and the human-readable side of a conflict. Every cluster carries
    its ``providers`` and ``evidence_answer_ids`` so a human can verify each
    assignment against the member's raw answer (DD-1 invariant 2).

    See DD-2 verdict schema.

    Attributes:
        label: Short normalized stance label (e.g. ``"explicit refresh only"``).
        summary: One-line human-readable summary of the clustered stance.
        providers: Provider names whose answers fall in this cluster (e.g.
            ``["anthropic", "openai"]``). Names only — never key material.
        evidence_answer_ids: Stable ``answer_id`` values backing this cluster
            (e.g. ``["anthropic-1", "openai-1"]``), for human verification.
    """

    label: str
    summary: str
    providers: list[str] = Field(default_factory=list)
    evidence_answer_ids: list[str] = Field(default_factory=list)


class CouncilConflict(BaseModel):
    """A disagreement between two or more positions (DD-2 optional block).

    Present only when there are ≥2 positions. Carries a per-conflict consensus
    ratio so a consumer can see agreement on a *specific* sub-question distinct
    from the top-level ``consensus_score``.

    See DD-2 verdict schema (optional ``conflicts``).

    Attributes:
        topic: What the conflict is about (the contested sub-question).
        position_labels: The ``CouncilPosition.label`` values in tension.
        summary: Optional human-readable summary of the disagreement.
        consensus_score: Optional per-conflict ratio (DD-1). ``None`` when not
            computed.
    """

    topic: str
    position_labels: list[str] = Field(default_factory=list)
    summary: str = ""
    consensus_score: float | None = None


class ProviderVote(BaseModel):
    """One provider's vote for a position (DD-2 optional ``provider_votes``).

    Satisfies the absorbed GH #3 "show me who voted for what" request.

    Attributes:
        provider: Provider name (e.g. ``"gemini"``). Name only — never a key.
        position_label: The ``CouncilPosition.label`` this provider lands on.
        confidence: Optional self-reported confidence (``low``/``medium``/``high``).
            Recorded, never used in the consensus arithmetic (DD-1).
    """

    provider: str
    position_label: str
    confidence: str | None = None


class MinorityReport(BaseModel):
    """A dissenting view worth surfacing (DD-2 optional ``minority_reports``).

    For ``adversarial`` runs this maps to unrefuted critic points.

    Attributes:
        providers: Provider names holding the minority view. Names only.
        claim: The minority claim itself.
        evidence_answer_ids: Stable ``answer_id`` values backing the claim.
        why_it_matters: Optional rationale for surfacing the dissent despite it
            being a minority view.
    """

    providers: list[str] = Field(default_factory=list)
    claim: str
    evidence_answer_ids: list[str] = Field(default_factory=list)
    why_it_matters: str = ""


class CouncilVerdict(BaseModel):
    """The synthesized adjudication of a council run (DD-2 verdict schema).

    Produced by the verdict-extraction step (CAC-05), not by this module. A
    verdict is never an empty shell: the required fields below always carry a
    real adjudication. The optional collections are present only when applicable
    (e.g. ``conflicts`` only when ≥2 positions). When the prompt is open-ended
    generation, N<2 members respond, or extraction fails after one repair, the
    verdict is absent entirely (``CouncilResult.verdict is None``) — see the
    DD-2 verdict-absent rule.

    ``consensus_score`` here is the position-cluster ratio (DD-1,
    :data:`CONSENSUS_METHOD`); for ``debate`` it is distinct from the existing
    ``convergence_score`` (difflib text-stability) and the two are never
    conflated.

    See DD-2 for the field-by-field spec and the ``verdict_type`` emphasis table.

    Attributes:
        verdict_type: One of :data:`VERDICT_TYPES`
            (``decision``/``review``/``synthesis``). Validated on construction.
        headline: One-line answer.
        recommendation: Actionable synthesized answer.
        consensus_score: Position-cluster ratio in ``[0.0, 1.0]``, or ``None``
            (DD-1; ``None`` for N<2 or no positioned members).
        consensus_method: The method literal; defaults to
            :data:`CONSENSUS_METHOD`.
        consensus_label: Deterministic bucket derived from the score
            (``unanimous``/``strong``/``majority``/``split``/``none``), or
            ``None`` until computed.
        positions: Clustered stances (≥1 when a verdict exists; a single element
            when unanimous).
        conflicts: Disagreements between positions (only when ≥2 positions).
        provider_votes: Per-provider votes (GH #3).
        minority_reports: Dissenting views worth surfacing.
        caveats: Cross-cutting caveats on the verdict.
        dissent_summary: Optional prose summary of the dissent.
        schema_version: The verdict schema version stamp; defaults to
            :data:`VERDICT_SCHEMA_VERSION` so a constructed verdict is
            self-stamping.
    """

    # REQUIRED (DD-2).
    verdict_type: str
    headline: str
    recommendation: str
    consensus_score: float | None = None
    consensus_method: str = CONSENSUS_METHOD
    consensus_label: str | None = None
    positions: list[CouncilPosition] = Field(default_factory=list)

    # OPTIONAL (DD-2).
    conflicts: list[CouncilConflict] = Field(default_factory=list)
    provider_votes: list[ProviderVote] = Field(default_factory=list)
    minority_reports: list[MinorityReport] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    dissent_summary: str | None = None

    # Self-stamping version (mirrors SYNTHESIS_PROMPT_VERSION pattern).
    schema_version: str = VERDICT_SCHEMA_VERSION


def member_answer_json_schema() -> dict:
    """Return the fixed JSON Schema sent to providers for a MEMBER answer (DD-2).

    This is the lowest-common-denominator schema each council member targets when
    structured output is on (CAC-02). ``prose`` holds the full natural reasoning
    so structuring never truncates answer quality; ``key_points`` is the only
    required field. ``answer_id`` is intentionally absent — it is assigned by
    conclave, not emitted by the model.

    The returned dict is a plain Python literal (a fresh object on each call) so a
    caller may mutate it without affecting other callers.

    Returns:
        A draft-style JSON Schema ``dict`` describing the member answer object.

    Example:
        >>> schema = member_answer_json_schema()
        >>> schema["required"]
        ['key_points']
        >>> schema["additionalProperties"]
        False
    """
    return {
        "title": "MemberAnswer",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # null = no clean stance; excluded from consensus (DD-1/DD-2).
            "position": {"type": ["string", "null"]},
            # REQUIRED load-bearing claims.
            "key_points": {"type": "array", "items": {"type": "string"}},
            # Recorded, never used in the arithmetic (DD-1).
            "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
            "caveats": {"type": "array", "items": {"type": "string"}},
            # Full natural reasoning (residue).
            "prose": {"type": "string"},
        },
        "required": ["key_points"],
    }


def verdict_json_schema() -> dict:
    """Return the fixed JSON Schema sent to providers for the VERDICT (DD-2).

    This is the lowest-common-denominator schema the verdict-extraction step
    (CAC-05) targets. Only the seven DD-2 required fields appear in ``required``;
    the optional fields are present in ``properties`` but omitted from
    ``required`` (optionality via omission, never conditional-required).

    The position object shape is inlined in both ``positions`` and each
    ``conflict`` rather than ``$ref``-shared, to satisfy the LCD constraint
    (no ``$ref``/``$defs``). Object nesting tops out at depth 3 (root → array
    ``items`` object → that object's array property ``items``), and every object
    node sets ``additionalProperties: false``.

    The returned dict is a plain Python literal (a fresh object on each call).

    Returns:
        A draft-style JSON Schema ``dict`` describing the verdict object.

    Example:
        >>> schema = verdict_json_schema()
        >>> sorted(schema["required"])
        ['consensus_label', 'consensus_method', 'consensus_score', 'headline', 'positions', 'recommendation', 'verdict_type']
    """
    # Inlined position object shape (depth-2 object; its array property strings
    # are depth-3 — no 4th object level). Reused verbatim for positions and the
    # conflict block per the LCD no-$ref rule.
    position_object = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},
            "summary": {"type": "string"},
            "providers": {"type": "array", "items": {"type": "string"}},
            "evidence_answer_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["label", "summary", "providers", "evidence_answer_ids"],
    }

    return {
        "title": "CouncilVerdict",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # REQUIRED (DD-2).
            "verdict_type": {"type": "string", "enum": list(VERDICT_TYPES)},
            "headline": {"type": "string"},
            "recommendation": {"type": "string"},
            "consensus_score": {"type": ["number", "null"]},
            "consensus_method": {"type": "string"},
            "consensus_label": {"type": "string"},
            "positions": {"type": "array", "items": position_object},
            # OPTIONAL (DD-2) — present in properties, absent from required.
            "conflicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "topic": {"type": "string"},
                        "position_labels": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "consensus_score": {"type": ["number", "null"]},
                    },
                    "required": ["topic", "position_labels", "summary"],
                },
            },
            "provider_votes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "provider": {"type": "string"},
                        "position_label": {"type": "string"},
                        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                    },
                    "required": ["provider", "position_label"],
                },
            },
            "minority_reports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "providers": {"type": "array", "items": {"type": "string"}},
                        "claim": {"type": "string"},
                        "evidence_answer_ids": {"type": "array", "items": {"type": "string"}},
                        "why_it_matters": {"type": "string"},
                    },
                    "required": ["providers", "claim", "evidence_answer_ids"],
                },
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
            "dissent_summary": {"type": "string"},
        },
        "required": [
            "verdict_type",
            "headline",
            "recommendation",
            "consensus_score",
            "consensus_method",
            "consensus_label",
            "positions",
        ],
    }


# Consensus fields the EXTRACTION schema strips out: the model must never emit
# any of them (DD-1 — the number is deterministic arithmetic over the model's
# clustering, computed by CAC-05, never asked of the model). Named once here so
# :func:`verdict_extraction_json_schema` and a guard test share the exact set.
_CONSENSUS_FIELDS = ("consensus_score", "consensus_method", "consensus_label")


def verdict_extraction_json_schema() -> dict:
    """Return the JSON Schema sent to the extractor model for verdict EXTRACTION.

    Derived from :func:`verdict_json_schema` (the full LCD verdict schema) as a
    template, then transformed for the CAC-05 extraction step so the model emits
    only its *judgment*, never the consensus arithmetic:

    * **ADDS** ``verdict_applies`` (``{"type": "boolean"}``) as a REQUIRED field —
      the open-ended-vs-decision discriminator the engine reads to decide whether
      a verdict applies at all (DD-2 verdict-absent rule).
    * **REMOVES** every consensus field (:data:`_CONSENSUS_FIELDS`) from both
      ``properties`` and ``required`` — the model must NEVER emit a consensus
      number; CAC-05 computes it deterministically from ``provider_votes`` via
      :mod:`conclave.agreement` (DD-1, the auditability-paradox fix).
    * **KEEPS** the judgment fields the model owns: ``verdict_type`` (enum),
      ``headline``, ``recommendation``, ``positions``, plus the optional
      ``conflicts``, ``provider_votes``, ``minority_reports``, ``caveats``,
      ``dissent_summary``. ``provider_votes[*].position_label`` is what the engine
      maps each member to in order to build the per-member clustering sequence.

    Inherits all LCD constraints from the template (object nesting ≤ 3, ``enum``
    not ``oneOf``/``anyOf``/``allOf``, no ``$ref``/``$defs``,
    ``additionalProperties: false`` on every object, optionality via
    omission-from-required). The returned dict is a fresh object on each call so a
    caller may mutate it freely.

    Returns:
        A draft-style JSON Schema ``dict`` for the verdict-extraction object.

    Example:
        >>> schema = verdict_extraction_json_schema()
        >>> schema["properties"]["verdict_applies"]
        {'type': 'boolean'}
        >>> "consensus_score" in schema["properties"]
        False
        >>> "verdict_applies" in schema["required"]
        True
    """
    schema = verdict_json_schema()
    schema["title"] = "VerdictExtraction"

    # ADD the discriminator (required) — the engine reads it to decide whether the
    # prompt is a decision/review at all.
    schema["properties"]["verdict_applies"] = {"type": "boolean"}

    # REMOVE every consensus field — the model must never emit the number.
    for field in _CONSENSUS_FIELDS:
        schema["properties"].pop(field, None)

    # Rebuild ``required``: drop the consensus fields, add ``verdict_applies``
    # (optionality stays via omission-from-required; we only add the new gate).
    schema["required"] = [r for r in schema["required"] if r not in _CONSENSUS_FIELDS]
    schema["required"].append("verdict_applies")
    return schema


class VerdictExtractionModel(BaseModel):
    """The validated structured output of the verdict-extraction step (CAC-05).

    This is the Pydantic *validator* for what the extractor model returns —
    Pydantic IS the validation gate (no ``jsonschema`` dependency). It mirrors the
    judgment fields of :class:`CouncilVerdict` but, per DD-1, carries **no
    consensus fields**: the engine computes ``consensus_score`` / ``method`` /
    ``label`` deterministically from :attr:`provider_votes` and never reads them
    off the model. Extra keys a model might smuggle in (e.g. a hallucinated
    ``consensus_score``) are ignored — Pydantic's default ``extra="ignore"`` drops
    unknown fields, so a smuggled number simply never reaches the assembled
    verdict.

    The reused element types (:class:`CouncilPosition`, :class:`CouncilConflict`,
    :class:`ProviderVote`, :class:`MinorityReport`) are the same shapes the final
    verdict carries, so the engine assembles a :class:`CouncilVerdict` directly
    from these validated sub-objects plus its own computed consensus values.

    See DD-2 (verdict schema) and DD-1 (consensus is never model-emitted).

    Attributes:
        verdict_applies: ``True`` when the prompt is a decision/review the council
            can adjudicate; ``False`` for open-ended generation (→ verdict absent,
            DD-2 verdict-absent rule). The single discriminator the engine reads
            to gate verdict assembly.
        verdict_type: One of :data:`VERDICT_TYPES`.
        headline: One-line answer.
        recommendation: Actionable synthesized answer.
        positions: Clustered stances (the model's clustering — the one
            LLM-assisted step, DD-1).
        conflicts: Disagreements between positions; the engine recomputes each
            ``consensus_score`` deterministically (any model-supplied value is
            overwritten).
        provider_votes: Per-provider votes. ``position_label`` on each is what the
            engine maps every responding member to in order to build the
            clustering sequence for :mod:`conclave.agreement`.
        minority_reports: Dissenting views worth surfacing.
        caveats: Cross-cutting caveats on the verdict.
        dissent_summary: Optional prose summary of the dissent.
    """

    verdict_applies: bool
    verdict_type: str
    headline: str
    recommendation: str
    positions: list[CouncilPosition] = Field(default_factory=list)
    conflicts: list[CouncilConflict] = Field(default_factory=list)
    provider_votes: list[ProviderVote] = Field(default_factory=list)
    minority_reports: list[MinorityReport] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    dissent_summary: str | None = None
