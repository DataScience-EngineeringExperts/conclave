from __future__ import annotations

from conclave.evals.blinding import blind_run_records
from conclave.evals.models import RunRecord


def _records() -> tuple[RunRecord, ...]:
    return (
        RunRecord(planned_run_id="run_" + "a" * 24, outcome="success", output="Alpha"),
        RunRecord(planned_run_id="run_" + "b" * 24, outcome="failed", error_category="error"),
        RunRecord(planned_run_id="run_" + "c" * 24, outcome="abstained"),
    )


def test_blinding_is_seeded_deterministic_opaque_and_separates_the_map() -> None:
    first_outputs, first_map = blind_run_records(_records(), seed=20260717)
    second_outputs, second_map = blind_run_records(_records(), seed=20260717)
    other_outputs, other_map = blind_run_records(_records(), seed=20260718)

    assert first_outputs == second_outputs
    assert first_map == second_map
    assert first_outputs != other_outputs
    assert first_map != other_map
    assert len({item.opaque_output_id for item in first_outputs.outputs}) == 3
    assert all(item.opaque_output_id.startswith("output_") for item in first_outputs.outputs)
    assert {entry.opaque_output_id for entry in first_map.entries} == {
        output.opaque_output_id for output in first_outputs.outputs
    }
    assert {entry.planned_run_id for entry in first_map.entries} == {
        record.planned_run_id for record in _records()
    }


def test_blinded_outputs_exclude_run_condition_provider_and_model_labels() -> None:
    blinded_outputs, blind_map = blind_run_records(_records(), seed=9)
    serialized_outputs = blinded_outputs.model_dump_json()

    assert "planned_run_id" not in serialized_outputs
    assert "condition" not in serialized_outputs
    assert "provider" not in serialized_outputs
    assert "model" not in serialized_outputs
    assert "run_aaaaaaaaaaaaaaaaaaaaaaaa" not in serialized_outputs
    assert "Alpha" in serialized_outputs
    assert "planned_run_id" in blind_map.model_dump_json()
