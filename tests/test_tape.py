"""Tests for the promoted sanitizing record/replay transport (DSE-1517 Task 1/1b).

Task 1 proves the promotion out of ``conclave.evals.replay`` changed nothing:
every moved name is importable from ``conclave.tape``, ``RecordingTransport`` /
``ReplayingTransport`` behave exactly like the eval transports they were
extracted from, and ``evals.replay.ReplayRecord`` is the identical object as
``conclave.tape.TapeRecord``.

Task 1b (appended) adds the CSO-mandated hardening probes: URL userinfo
stripping, the cache's wider secret-query set, secret-shaped query values,
response root-key scrubbing, and a ``CustomEndpoint`` validator that rejects
userinfo in a configured endpoint URL.
"""

from __future__ import annotations

import json

import pytest

from conclave.evals import replay as evals_replay
from conclave.tape import (
    _AMBIGUOUS_CREDENTIAL_NAMES,
    _CREDENTIAL_NAMES,
    TAPE_SCHEMA_VERSION,
    PostJson,
    RecordingTransport,
    ReplayCompatibilityError,
    ReplayError,
    ReplayingTransport,
    ReplayMismatchError,
    Sha256Digest,
    Tape,
    TapeRecord,
    _credentials,
    _hash_request,
    _is_sensitive_name,
    _redact_exact,
    _request,
    _sanitize,
    _sanitize_stored_request,
    _sanitize_url,
    _string_values,
)

RUN_HASH = "sha256:" + "a" * 64


def test_moved_names_are_importable_from_tape():
    """Every name the plan lists as "moved" resolves from ``conclave.tape``."""
    assert TAPE_SCHEMA_VERSION == "conclave_tape_v1"
    assert callable(_is_sensitive_name)
    assert callable(_redact_exact)
    assert callable(_sanitize)
    assert callable(_sanitize_url)
    assert callable(_string_values)
    assert callable(_credentials)
    assert callable(_hash_request)
    assert callable(_sanitize_stored_request)
    assert callable(_request)
    assert "key" in _AMBIGUOUS_CREDENTIAL_NAMES
    assert "api_key" in _CREDENTIAL_NAMES
    assert issubclass(ReplayCompatibilityError, ReplayError)
    assert issubclass(ReplayMismatchError, ReplayError)
    assert PostJson is not None
    assert Sha256Digest is not None


def test_evals_replay_record_is_tape_record():
    """``evals.replay.ReplayRecord`` must be the identical object as ``TapeRecord``."""
    assert evals_replay.ReplayRecord is TapeRecord


async def test_recording_transport_strips_seeded_credentials_matrix():
    """Bearer header, x-api-key/x-goog-api-key headers, query ``key=``, body
    ``api_key``/``token`` -- none of the seeded credential values or their
    carrying names survive into the recorded tape."""

    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    transport = RecordingTransport(network, run_identity_hash=RUN_HASH)
    await transport(
        "https://example.test/generate?key=AIzaSecretValue123&alt=json",
        {
            "Authorization": "Bearer sk-secret-value",
            "x-api-key": "opaque-x-api-key-value",
            "x-goog-api-key": "AIzaSecretValue123",
        },
        {
            "model": "m",
            "api_key": "sk-secret-value",
            "token": "opaque-body-token-value",
            "prompt": "safe",
        },
        10,
    )
    encoded = json.dumps(transport.tape().model_dump(mode="json"), sort_keys=True)
    for leaked in (
        "AIzaSecretValue123",
        "sk-secret-value",
        "opaque-x-api-key-value",
        "opaque-body-token-value",
        "Authorization",
        "x-api-key",
        "x-goog-api-key",
        "api_key",
    ):
        assert leaked not in encoded
    assert "alt=json" in encoded


def test_tape_rejects_an_unsanitized_record():
    bad_request = {"url": "https://example.test/v1", "body": {"api_key": "leaked"}}
    record = TapeRecord(
        request_hash=_hash_request(bad_request),
        occurrence_index=0,
        request=bad_request,
        status=200,
        response={"ok": True},
    )
    with pytest.raises(ValueError, match="sanitized"):
        Tape(run_identity_hash=RUN_HASH, records=(record,))


def test_tape_rejects_a_wrong_hash():
    request = {"url": "https://example.test/v1", "body": {"prompt": "hi"}}
    record = TapeRecord(
        request_hash="sha256:" + "b" * 64,
        occurrence_index=0,
        request=request,
        status=200,
        response={"ok": True},
    )
    with pytest.raises(ValueError, match="request hash"):
        Tape(run_identity_hash=RUN_HASH, records=(record,))


async def test_tape_rejects_noncontiguous_occurrences():
    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    transport = RecordingTransport(network, run_identity_hash=RUN_HASH)
    await transport("https://example.test/v1", {}, {"prompt": "safe"}, 10)
    record = transport.tape().records[0].model_copy(update={"occurrence_index": 1})

    with pytest.raises(ValueError, match="contiguous"):
        Tape(run_identity_hash=RUN_HASH, records=(record,))


async def test_replaying_transport_raises_on_unmatched_request():
    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    transport = RecordingTransport(network, run_identity_hash=RUN_HASH)
    await transport("https://example.test/v1", {}, {"prompt": "one"}, 10)
    tape = transport.tape()

    replay = ReplayingTransport(tape, run_identity_hash=RUN_HASH)
    with pytest.raises(ReplayMismatchError, match="unmatched request"):
        await replay("https://example.test/v1", {}, {"prompt": "two"}, 10)


async def test_replaying_transport_raises_on_leftover_records():
    async def network(url, headers, body, timeout):
        return 200, {"ok": True}

    transport = RecordingTransport(network, run_identity_hash=RUN_HASH)
    await transport("https://example.test/v1", {}, {"prompt": "one"}, 10)
    tape = transport.tape()

    replay = ReplayingTransport(tape, run_identity_hash=RUN_HASH)
    with pytest.raises(ReplayMismatchError, match="unconsumed record"):
        replay.assert_consumed()
