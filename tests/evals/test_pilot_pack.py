from __future__ import annotations

from collections import Counter
from pathlib import Path

from conclave.evals.dataset import load_grader_keys, load_public_tasks

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "studies" / "elite_pilot_v1"

EXPECTED_TASK_IDS = {
    *(f"PRC-{index:02d}" for index in range(1, 5)),
    *(f"REL-{index:02d}" for index in range(1, 5)),
    *(f"PRD-{index:02d}" for index in range(1, 5)),
    *(f"CAP-{index:02d}" for index in range(1, 5)),
    *(f"GOV-{index:02d}" for index in range(1, 5)),
    *(f"STF-{index:02d}" for index in range(1, 5)),
}
EXPECTED_DIMENSIONS = {
    "constraint_recall",
    "conflict_minority_recognition",
    "unsupported_claim_avoidance",
    "recommendation_correctness",
    "completeness_actionability",
    "readiness_calibration",
}
SHARED_PROMPT = (
    "Using only the reference packet, choose the best course of action. Return: "
    "(1) Recommendation, (2) Readiness: ready|not_ready|indeterminate, "
    "(3) Hard-constraint check, (4) Conflicts/minority view, "
    "(5) Next actions and owners, (6) Unknowns. Cite packet IDs. Do not invent facts."
)


def _expected_readiness(required_facts: tuple[str, ...]) -> str:
    matches = [
        fact.removeprefix("expected_readiness:")
        for fact in required_facts
        if fact.startswith("expected_readiness:")
    ]
    assert len(matches) == 1
    return matches[0]


def test_pilot_pack_is_complete_balanced_and_public_private_separated() -> None:
    tasks = load_public_tasks(PACK / "public_tasks.json")
    keys = load_grader_keys(PACK / "grader_keys.json")

    assert len(tasks) == len(keys) == 24
    assert {task.task_id for task in tasks} == EXPECTED_TASK_IDS
    assert {key.task_id for key in keys} == EXPECTED_TASK_IDS
    assert Counter(task.metadata["macro_family"] for task in tasks) == {
        "operational_execution": 12,
        "organizational_stewardship": 12,
    }
    assert Counter(task.metadata["subfamily"] for task in tasks) == {
        "procurement": 4,
        "release_operations": 4,
        "product_experiments": 4,
        "capital_allocation": 4,
        "governance": 4,
        "staffing": 4,
    }
    assert Counter(task.metadata["tier"] for task in tasks) == {
        "tier_1": 8,
        "tier_2": 8,
        "tier_3": 8,
    }
    assert Counter(_expected_readiness(key.required_facts) for key in keys) == {
        "ready": 8,
        "not_ready": 8,
        "indeterminate": 8,
    }

    assert all(task.prompt == SHARED_PROMPT for task in tasks)
    assert all(len(task.reference_packets) >= 3 for task in tasks)
    assert all(set(key.rubric) == EXPECTED_DIMENSIONS for key in keys)
    assert all(len(key.required_facts) >= 4 for key in keys)
    assert all(len(key.critical_errors) >= 3 for key in keys)

    public_text = (PACK / "public_tasks.json").read_text(encoding="utf-8")
    assert "expected_readiness:" not in public_text
    assert "required_facts" not in public_text
    assert "critical_errors" not in public_text


def test_pilot_protocol_freezes_exploratory_boundary_and_holdout_controls() -> None:
    readme = (PACK / "README.md").read_text(encoding="utf-8")
    protocol = (PACK / "pilot_protocol.md").read_text(encoding="utf-8")
    preregistration = (PACK / "confirmatory_preregistration.md").read_text(encoding="utf-8")

    assert "synthetic exploratory" in readme.lower()
    assert "must not support product-quality claims" in readme.lower()
    assert "critical-error-free decision rate" in protocol.lower()
    assert "failures remain in the denominator" in protocol.lower()
    assert "grader-only" in protocol.lower()
    assert "not a completed preregistration" in preregistration.lower()
    assert "new scenario archetypes" in preregistration.lower()
    assert "parameter swaps" in preregistration.lower()
    assert "eight-token" in preregistration.lower()
    assert "freeze" in preregistration.lower()
    assert "unsealing" in preregistration.lower()
