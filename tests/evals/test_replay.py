"""Strict record/replay tests at the buffered transport seam."""

from __future__ import annotations

import json

import pytest

from conclave.evals.replay import (
    REPLAY_SCHEMA_VERSION,
    RecordingPostJson,
    ReplayArtifact,
    ReplayCompatibilityError,
    ReplayingPostJson,
    ReplayMismatchError,
    ReplayRecord,
)

BASE_HASH = "sha256:" + "a" * 64


async def test_record_replay_uses_occurrence_indexes_and_never_calls_network():
    calls = 0

    async def network(url, headers, body, timeout):
        nonlocal calls
        calls += 1
        return 200, {"answer": calls}

    recorder = RecordingPostJson(network, base_manifest_hash=BASE_HASH)
    args = (
        "https://api.example.test/v1/chat?alt=json",
        {"Authorization": "Bearer sk-secret-value"},
        {"model": "m", "messages": [{"role": "user", "content": "same"}]},
        30.0,
    )
    assert await recorder(*args) == (200, {"answer": 1})
    assert await recorder(*args) == (200, {"answer": 2})
    artifact = recorder.artifact()
    assert [record.occurrence_index for record in artifact.records] == [0, 1]

    async def forbidden_network(*args, **kwargs):
        raise AssertionError("replay performed network I/O")

    replay = ReplayingPostJson(artifact, base_manifest_hash=BASE_HASH)
    assert await replay(*args) == (200, {"answer": 1})
    assert await replay(*args) == (200, {"answer": 2})
    replay.assert_consumed()
    assert calls == 2


async def test_artifact_excludes_headers_keys_and_secret_url_parameters():
    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    recorder = RecordingPostJson(network, base_manifest_hash=BASE_HASH)
    await recorder(
        "https://example.test/generate?key=AIzaSecretValue123&alt=json",
        {"x-goog-api-key": "AIzaSecretValue123", "Authorization": "Bearer sk-secret-value"},
        {"model": "m", "api_key": "sk-secret-value", "prompt": "safe"},
        10,
    )
    encoded = json.dumps(recorder.artifact().model_dump(mode="json"), sort_keys=True)
    assert "AIzaSecretValue123" not in encoded
    assert "sk-secret-value" not in encoded
    assert "Authorization" not in encoded
    assert "x-goog-api-key" not in encoded
    assert "api_key" not in encoded
    assert "alt=json" in encoded


async def test_recording_preserves_legitimate_nested_key_token_and_clean_response_strings():
    async def network(url, headers, body, timeout):
        return 200, {"message": "token budgeting uses a key/value table"}

    recorder = RecordingPostJson(network, base_manifest_hash=BASE_HASH)
    await recorder(
        "https://example.test/v1",
        {},
        {
            "metadata": {"key": "topic", "token": "budget-unit"},
            "prompt": "Explain token budgeting.",
        },
        10,
    )

    record = recorder.artifact().records[0]
    assert record.request["body"]["metadata"] == {"key": "topic", "token": "budget-unit"}
    assert record.response == {"message": "token budgeting uses a key/value table"}


async def test_recording_redacts_exact_credentials_from_request_and_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "opaque-env-credential")

    async def network(url, headers, body, timeout):
        return 200, {
            "header_echo": "Bearer opaque-header-credential",
            "body_echo": "opaque-body-credential",
            "url_echo": "opaque-url-credential",
            "env_echo": "opaque-env-credential",
        }

    recorder = RecordingPostJson(network, base_manifest_hash=BASE_HASH)
    await recorder(
        "https://example.test/v1?api_key=opaque-url-credential&alt=json",
        {"Authorization": "Bearer opaque-header-credential"},
        {"api_key": "opaque-body-credential", "prompt": "safe"},
        10,
    )

    encoded = json.dumps(recorder.artifact().model_dump(mode="json"), sort_keys=True)
    for credential in (
        "opaque-header-credential",
        "opaque-body-credential",
        "opaque-url-credential",
        "opaque-env-credential",
    ):
        assert credential not in encoded


def test_replay_rejects_schema_or_base_manifest_hash_drift():
    artifact = ReplayArtifact(base_manifest_hash=BASE_HASH, records=())
    with pytest.raises(ReplayCompatibilityError, match="base manifest hash"):
        ReplayingPostJson(artifact, base_manifest_hash="sha256:" + "b" * 64)

    drifted = artifact.model_copy(update={"schema_version": "future"})
    with pytest.raises(ReplayCompatibilityError, match="schema version"):
        ReplayingPostJson(drifted, base_manifest_hash=BASE_HASH)
    assert artifact.schema_version == REPLAY_SCHEMA_VERSION


async def test_replay_fails_closed_on_missing_mismatch_and_extra_records():
    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    recorder = RecordingPostJson(network, base_manifest_hash=BASE_HASH)
    await recorder("https://example.test/v1", {}, {"model": "m", "prompt": "one"}, 10)
    artifact = recorder.artifact()

    replay = ReplayingPostJson(artifact, base_manifest_hash=BASE_HASH)
    with pytest.raises(ReplayMismatchError, match="unmatched request"):
        await replay("https://example.test/v1", {}, {"model": "m", "prompt": "two"}, 10)

    replay = ReplayingPostJson(artifact, base_manifest_hash=BASE_HASH)
    with pytest.raises(ReplayMismatchError, match="unconsumed record"):
        replay.assert_consumed()

    replay = ReplayingPostJson(artifact, base_manifest_hash=BASE_HASH)
    await replay("https://example.test/v1", {}, {"model": "m", "prompt": "one"}, 10)
    with pytest.raises(ReplayMismatchError, match="unmatched request"):
        await replay("https://example.test/v1", {}, {"model": "m", "prompt": "one"}, 10)


def test_replay_artifact_rejects_stored_request_hash_mismatch():
    request = {"url": "https://example.test/v1", "body": {"prompt": "safe"}}
    record = ReplayRecord(
        request_hash="sha256:" + "b" * 64,
        occurrence_index=0,
        request=request,
        status=200,
        response={"ok": True},
    )

    with pytest.raises(ValueError, match="request hash"):
        ReplayArtifact(base_manifest_hash=BASE_HASH, records=(record,))


async def test_replay_artifact_rejects_noncontiguous_occurrence_indexes():
    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    recorder = RecordingPostJson(network, base_manifest_hash=BASE_HASH)
    await recorder("https://example.test/v1", {}, {"prompt": "safe"}, 10)
    record = recorder.artifact().records[0].model_copy(update={"occurrence_index": 1})

    with pytest.raises(ValueError, match="contiguous"):
        ReplayArtifact(base_manifest_hash=BASE_HASH, records=(record,))


async def test_replay_artifact_rejects_secret_bearing_or_tampered_payloads():
    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    recorder = RecordingPostJson(network, base_manifest_hash=BASE_HASH)
    await recorder("https://example.test/v1", {}, {"prompt": "safe"}, 10)
    valid = recorder.artifact().records[0]

    secret_request = valid.model_copy(
        update={"request": {**valid.request, "authorization": "Bearer opaque-secret"}}
    )
    secret_response = valid.model_copy(update={"response": "Bearer sk-secret-value"})
    for record in (secret_request, secret_response):
        with pytest.raises(ValueError, match="sanitized"):
            ReplayArtifact(base_manifest_hash=BASE_HASH, records=(record,))

    bypassed_validation = recorder.artifact().model_copy(update={"records": (secret_response,)})
    with pytest.raises(ReplayCompatibilityError, match="integrity"):
        ReplayingPostJson(bypassed_validation, base_manifest_hash=BASE_HASH)
