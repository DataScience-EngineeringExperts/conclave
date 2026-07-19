"""End-to-end sanitized transport replay for the capped live condition matrix."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import conclave.evals.runner as runner_module
import conclave.transport as transport
from conclave.config import ConclaveConfig, CustomEndpoint
from conclave.evals.dataset import load_public_tasks
from conclave.evals.live import (
    build_checkpoint_bindings,
    hash_study_manifest,
    load_live_checkpoint,
)
from conclave.evals.live_protocols import stage_call_sequence
from conclave.evals.models import EVAL_CONDITION_IDS, StudyManifest
from conclave.evals.pricing import PriceBook
from conclave.evals.replay import (
    RecordingPostJson,
    ReplayArtifact,
    ReplayingPostJson,
    ReplayMismatchError,
)
from conclave.evals.runner import LIVE_HARD_CAP_USD, run_live_study
from conclave.providers import call_model
from conclave.verdict import verdict_extraction_json_schema

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures/evals/live_smoke"
FAKE_KEY_ENV = "CONCLAVE_FAKE_TEST_KEY"
FAKE_KEY = "fixture-only-opaque-credential"
CHECKPOINT_SEAL_KEY = bytes(range(32))


def _canonical_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _assert_strict_schema_instance(value: object, schema: dict, path: str = "$") -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        assert isinstance(value, dict), f"{path} must be an object"
        properties = schema["properties"]
        assert set(schema["required"]) <= set(value), f"{path} is missing required fields"
        if schema.get("additionalProperties") is False:
            assert set(value) <= set(properties), f"{path} has undeclared fields"
        for name, item in value.items():
            _assert_strict_schema_instance(item, properties[name], f"{path}.{name}")
    elif schema_type == "array":
        assert isinstance(value, list), f"{path} must be an array"
        for index, item in enumerate(value):
            _assert_strict_schema_instance(item, schema["items"], f"{path}[{index}]")
    elif schema_type == "string":
        assert isinstance(value, str), f"{path} must be a string"
    elif schema_type == "boolean":
        assert isinstance(value, bool), f"{path} must be a boolean"
    elif schema_type == "number":
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{path} must be a number"
        )
    else:
        raise AssertionError(f"unsupported fixture schema type at {path}: {schema_type!r}")
    if "enum" in schema:
        assert value in schema["enum"], f"{path} is outside its enum"


def _deterministic_clock():
    tick = -1

    def perf_counter() -> float:
        nonlocal tick
        tick += 1
        return tick / 1000

    return perf_counter


def _custom_config(manifest: StudyManifest) -> ConclaveConfig:
    model_ids = {
        member.model_id for roster in manifest.frozen_design.rosters for member in roster.members
    }
    return ConclaveConfig(
        endpoints={
            model_id: CustomEndpoint(
                completions_url="https://fictional.invalid/v1/chat/completions",
                env_var=FAKE_KEY_ENV,
            )
            for model_id in model_ids
        }
    )


def _fake_delegate():
    calls: list[tuple[str, str, int]] = []

    async def delegate(url, headers, body, timeout):
        assert url == "https://fictional.invalid/v1/chat/completions"
        assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
        calls.append((url, body["model"], body["max_tokens"]))
        ordinal = len(calls)
        content = f"Fictional decision artifact {ordinal:03d}."
        if any(
            "verdict extractor" in message.get("content", "")
            for message in body.get("messages", [])
        ):
            content = json.dumps(
                {
                    "verdict_applies": True,
                    "verdict_type": "decision",
                    "headline": "Choose the fictional safe route.",
                    "recommendation": "Use Route A with the stated safeguards.",
                    "positions": [
                        {
                            "label": "route-a",
                            "summary": "Route A is preferred.",
                            "providers": ["Model A", "Model B", "Model C"],
                            "evidence_answer_ids": [
                                "fixture-a",
                                "fixture-b",
                                "fixture-c",
                            ],
                        }
                    ],
                    "provider_votes": [
                        {"provider": "Model A", "position_label": "route-a"},
                        {"provider": "Model B", "position_label": "route-a"},
                        {"provider": "Model C", "position_label": "route-a"},
                    ],
                    "conflicts": [],
                    "minority_reports": [],
                    "caveats": [],
                    "dissent_summary": "",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        return 200, {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    return delegate, calls


async def _execute(
    *,
    monkeypatch,
    manifest,
    tasks,
    price_book,
    post_json,
    checkpoint_path,
):
    config = _custom_config(manifest)

    async def configured_call_model(name, model_id, messages, **kwargs):
        return await call_model(name, model_id, messages, config=config, **kwargs)

    monkeypatch.setattr(transport, "post_json", post_json)
    monkeypatch.setattr(
        runner_module,
        "time",
        SimpleNamespace(perf_counter=_deterministic_clock()),
    )
    study_run = await run_live_study(
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        checkpoint_path=checkpoint_path,
        checkpoint_seal_key=CHECKPOINT_SEAL_KEY,
        call_model_func=configured_call_model,
    )
    checkpoint = load_live_checkpoint(
        checkpoint_path,
        expected_bindings=build_checkpoint_bindings(
            manifest,
            price_book,
            hard_cap_usd=LIVE_HARD_CAP_USD,
        ),
        seal_key=CHECKPOINT_SEAL_KEY,
    )
    return study_run, checkpoint


@pytest.mark.asyncio
async def test_live_smoke_replay_executes_all_conditions_with_zero_network_calls(
    tmp_path, monkeypatch, clear_keys
) -> None:
    del clear_keys
    monkeypatch.setenv(FAKE_KEY_ENV, FAKE_KEY)
    tasks = load_public_tasks(FIXTURE_DIR / "public_tasks.json")
    manifest = StudyManifest.model_validate_json(
        (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    price_book = PriceBook.model_validate_json(
        (FIXTURE_DIR / "price_book.json").read_text(encoding="utf-8")
    )
    artifact = ReplayArtifact.model_validate_json(
        (FIXTURE_DIR / "replay.json").read_text(encoding="utf-8")
    )
    base_manifest_hash = hash_study_manifest(manifest)
    assert artifact.base_manifest_hash == base_manifest_hash

    verdict_records = 0
    for record in artifact.records:
        messages = record.request["body"].get("messages", ())
        if not any("verdict extractor" in message.get("content", "") for message in messages):
            continue
        verdict_records += 1
        content = record.response["choices"][0]["message"]["content"]
        _assert_strict_schema_instance(json.loads(content), verdict_extraction_json_schema())
    assert verdict_records == len(manifest.frozen_design.rosters)

    delegate, delegate_calls = _fake_delegate()
    recorder = RecordingPostJson(delegate, base_manifest_hash=base_manifest_hash)
    recorded_run, recorded_checkpoint = await _execute(
        monkeypatch=monkeypatch,
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        post_json=recorder,
        checkpoint_path=tmp_path / "recorded-checkpoint.json",
    )
    assert _canonical_bytes(recorder.artifact()) == _canonical_bytes(artifact)

    replay_one = ReplayingPostJson(artifact, base_manifest_hash=base_manifest_hash)
    first_run, first_checkpoint = await _execute(
        monkeypatch=monkeypatch,
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        post_json=replay_one,
        checkpoint_path=tmp_path / "first-replay-checkpoint.json",
    )
    replay_one.assert_consumed()

    replay_two = ReplayingPostJson(artifact, base_manifest_hash=base_manifest_hash)
    second_run, second_checkpoint = await _execute(
        monkeypatch=monkeypatch,
        manifest=manifest,
        tasks=tasks,
        price_book=price_book,
        post_json=replay_two,
        checkpoint_path=tmp_path / "second-replay-checkpoint.json",
    )
    replay_two.assert_consumed()

    planned_ids = [planned.planned_run_id for planned in manifest.planned_runs]
    record_counts = Counter(record.planned_run_id for record in first_run.records)
    assert record_counts == Counter(dict.fromkeys(planned_ids, 1))
    assert all(record.outcome == "success" for record in first_run.records)
    condition_counts = Counter(planned.condition_id for planned in manifest.planned_runs)
    assert set(condition_counts) == set(EVAL_CONDITION_IDS)
    assert set(condition_counts.values()) == {len(manifest.frozen_design.rosters)}

    roster_by_id = {roster.roster_id: roster for roster in manifest.frozen_design.rosters}
    expected_call_order = []
    for planned in manifest.planned_runs:
        roster = roster_by_id[planned.roster_id]
        expected_call_order.extend(
            (stage, roster.members[index].provider_id, roster.members[index].model_id)
            for stage, index in stage_call_sequence(
                planned.condition_id,
                roster_size=len(roster.members),
            )
            if stage != "verdict_repair"
        )
    actual_call_order = [
        (receipt.stage, receipt.provider_id, receipt.model_id)
        for receipt in first_checkpoint.receipts
    ]
    assert actual_call_order == expected_call_order
    assert len(delegate_calls) == len(expected_call_order) == len(artifact.records)

    def receipt_bytes(checkpoint):
        return _canonical_bytes(
            [receipt.model_dump(mode="json") for receipt in checkpoint.receipts]
        )

    assert _canonical_bytes(recorded_run) == _canonical_bytes(first_run)
    assert _canonical_bytes(first_run) == _canonical_bytes(second_run)
    assert receipt_bytes(recorded_checkpoint) == receipt_bytes(first_checkpoint)
    assert receipt_bytes(first_checkpoint) == receipt_bytes(second_checkpoint)
    assert (tmp_path / "first-replay-checkpoint.json").read_bytes() == (
        tmp_path / "second-replay-checkpoint.json"
    ).read_bytes()

    serialized = b"\n".join(
        (
            (FIXTURE_DIR / "replay.json").read_bytes(),
            (tmp_path / "recorded-checkpoint.json").read_bytes(),
            (tmp_path / "first-replay-checkpoint.json").read_bytes(),
            receipt_bytes(first_checkpoint),
            _canonical_bytes(first_run),
        )
    )
    assert FAKE_KEY.encode() not in serialized

    delegate_count = len(delegate_calls)
    missing = ReplayingPostJson(artifact, base_manifest_hash=base_manifest_hash)
    with pytest.raises(ReplayMismatchError, match="unconsumed record"):
        missing.assert_consumed()

    first_record = artifact.records[0]
    changed_body = {**first_record.request["body"], "temperature": 0.125}
    changed = ReplayingPostJson(artifact, base_manifest_hash=base_manifest_hash)
    with pytest.raises(ReplayMismatchError, match="unmatched request"):
        await changed(
            first_record.request["url"],
            {"Authorization": f"Bearer {FAKE_KEY}"},
            changed_body,
            1,
        )

    with pytest.raises(ReplayMismatchError, match="unmatched request"):
        await replay_one(
            first_record.request["url"],
            {"Authorization": f"Bearer {FAKE_KEY}"},
            first_record.request["body"],
            1,
        )
    assert len(delegate_calls) == delegate_count
