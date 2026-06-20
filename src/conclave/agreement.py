"""Deterministic consensus arithmetic: ``position_cluster_ratio_v1`` (DD-1).

Authoritative spec: ``03_DESIGN_DECISIONS_v1.1.md`` DD-1. This module is the pure
arithmetic substrate the verdict-extraction step (CAC-05) builds on. It answers
exactly one question: *given each responding member's PRE-ASSIGNED cluster label,
what fraction landed in the largest cluster?* -- and buckets that ratio into the
DD-1 label.

The auditability paradox (``00_SCOPE_PLAN.md`` §4.1) governs what this module may
NOT do. The deterministic consensus measure must never be a text-similarity
score: difflib's :class:`~difflib.SequenceMatcher` ratio is the debate
``convergence_score`` (text-stability between rounds), a FORBIDDEN consensus
measure -- conflating "the answers read similarly" with "the council agrees" is
the exact mistake DD-1 forbids. Therefore **this module does not import difflib**
and a guard test (``tests/test_agreement.py``) asserts the source never gains
that import.

Semantic clustering of free-text positions (deciding that "yes, but cache it" and
"affirmative with caching" are the *same* stance) is an LLM judgment and belongs
to CAC-05, NOT here. The seam is explicit: callers pre-assign a cluster label to
each member and pass the list of labels in. :func:`consensus_score` then does
*categorical/exact* grouping only -- it normalizes labels for case/whitespace and
groups identical strings. It performs no semantic reasoning and no I/O, and it
touches only position strings (never key material or secrets).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

# Single source of truth for the method literal: re-exported from verdict.py so a
# rename there can never silently desync from the engine that implements it.
from .verdict import CONSENSUS_METHOD

__all__ = ["CONSENSUS_METHOD", "consensus_score", "consensus_label", "consensus"]


def _normalize_label(label: str) -> str:
    """Normalize one cluster label for exact/categorical grouping.

    The single place the grouping rule lives, so :func:`consensus_score`'s notion
    of "the same cluster" is defined once and is independently testable. The rule
    is deliberately conservative -- it is for EXACT/categorical labels only (e.g.
    ``"yes"``, ``"no"``, ``"conditional"``), never semantic equivalence -- so it
    only folds away cosmetic differences:

    * case-fold (``"Yes"`` == ``"yes"`` == ``"YES"``),
    * strip leading/trailing whitespace,
    * collapse internal runs of whitespace to a single space.

    Args:
        label: A raw, pre-assigned cluster label string.

    Returns:
        The normalized label used as the grouping key.

    Example:
        >>> _normalize_label("  Explicit   Refresh ")
        'explicit refresh'
    """
    return " ".join(label.split()).casefold()


def consensus_score(positions: Sequence[str | None]) -> float | None:
    """Compute ``position_cluster_ratio_v1``: largest cluster / positioned members.

    Pure arithmetic over PRE-ASSIGNED cluster labels (DD-1). Each element is one
    responding member's cluster label, already assigned by the caller (CAC-05);
    ``None`` means that member expressed no clean stance and is EXCLUDED from both
    numerator and denominator. Labels are normalized via :func:`_normalize_label`
    (case/whitespace-insensitive, exact grouping only -- no semantic clustering),
    then identical normalized labels form a cluster.

    The score is ``|largest cluster| / |members with a non-null position|``. The
    denominator is the count of responding members that expressed a position;
    a ``"conditional"`` / ``"it depends"`` label is an ordinary non-null string,
    so it is a valid cluster and counts in the denominator (DD-1 default: yes --
    no special-casing here). Self-reported ``confidence`` is never an input to
    this arithmetic (DD-1).

    When fewer than two members expressed a position (``N < 2``), agreement is
    undefined and the score is ``None`` -- a lone vote is not consensus, and a
    1-of-2 split is a tie (``0.5``), never "50% consensus".

    Args:
        positions: One pre-assigned cluster label per responding member;
            ``None`` for a member with no clean stance (dropped before counting).

    Returns:
        The consensus ratio in ``(0.0, 1.0]`` when at least two members expressed
        a position, else ``None`` (``N < 2`` after dropping nulls).

    Example:
        >>> consensus_score(["yes", "yes", "yes", "no"])
        0.75
        >>> consensus_score(["yes", "yes", None])
        1.0
        >>> consensus_score(["yes"])  # returns None
    """
    # Drop null positions: members with no clean stance leave both numerator and
    # denominator (DD-1 exclusion rule).
    labels = [_normalize_label(p) for p in positions if p is not None]
    total = len(labels)
    if total < 2:
        # N<2 positioned members -> agreement undefined (DD-1).
        return None
    largest = max(Counter(labels).values())
    return largest / total


def consensus_label(score: float | None) -> str:
    """Bucket a consensus score into the DD-1 label.

    Deterministic buckets straight from the DD-1 table. Boundary handling is
    explicit and verified against the table: ``strong`` is INCLUSIVE at ``0.75``
    (``>= 0.75``), ``majority`` is EXCLUSIVE of ``0.5`` (``> 0.5``), and exactly
    ``0.5`` falls through to ``split``. ``1.0`` is ``unanimous`` (checked before
    ``strong`` so a perfect score never reads as merely "strong").

    +--------------+-------------------------------+
    | label        | rule                          |
    +==============+===============================+
    | ``none``     | score is ``None``             |
    | ``unanimous``| score == 1.0  (N >= 2)        |
    | ``strong``   | 0.75 <= score < 1.0           |
    | ``majority`` | 0.5  <  score < 0.75          |
    | ``split``    | score <= 0.5 (no majority)    |
    +--------------+-------------------------------+

    Args:
        score: A :func:`consensus_score` result, or ``None`` (no positioned
            members / ``N < 2``).

    Returns:
        One of ``"none"``, ``"unanimous"``, ``"strong"``, ``"majority"``,
        ``"split"``.

    Example:
        >>> consensus_label(1.0)
        'unanimous'
        >>> consensus_label(0.75)
        'strong'
        >>> consensus_label(0.5)
        'split'
        >>> consensus_label(None)
        'none'
    """
    if score is None:
        return "none"
    if score == 1.0:
        return "unanimous"
    if score >= 0.75:
        return "strong"
    if score > 0.5:
        return "majority"
    return "split"


def consensus(positions: Sequence[str | None]) -> tuple[float | None, str]:
    """Convenience: return ``(score, label)`` in one call.

    Equivalent to ``(s := consensus_score(positions), consensus_label(s))``, so a
    caller that needs both the ratio and its DD-1 bucket gets a consistent pair
    without re-deriving one from the other.

    Args:
        positions: One pre-assigned cluster label per responding member; ``None``
            for a member with no clean stance.

    Returns:
        A ``(score, label)`` tuple, e.g. ``(0.75, "strong")`` or
        ``(None, "none")``.

    Example:
        >>> consensus(["yes", "yes", "no"])
        (0.6666666666666666, 'majority')
    """
    score = consensus_score(positions)
    return score, consensus_label(score)
