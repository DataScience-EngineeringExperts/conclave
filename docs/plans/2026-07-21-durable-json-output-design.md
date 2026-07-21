# Durable JSON Output Design

## Problem

`conclave ask --json` writes its result only to stdout after the full council run.
Long adversarial runs can outlive a supervising shell or agent tool's output session,
so the council succeeds but the caller loses the final JSON. Increasing a caller
timeout is brittle, and a wrapper cannot guarantee that every integration handles
process detachment correctly.

## Decision

Add an opt-in `--json-output PATH` option to `conclave ask`. After a council result
is complete, Conclave serializes the same payload used by `--json` and writes it to
the requested path before applying the existing exit-code contract. The option does
not require `--json`: callers may retain human terminal output while polling the
durable file. Existing stdout behavior remains unchanged.

The file is written atomically through a securely created temporary sibling and
replaced only after the complete JSON has been flushed. It uses user-private
temporary-file semantics (`0600` on POSIX). The parent directory must already exist;
Conclave will not create directory trees or silently redirect output. A write failure
is reported without exposing the result body, preserves normal stdout, and exits 1.

Alternatives rejected:

- Raising orchestration timeouts still loses output when the supervisor detaches.
- A repository-specific shell wrapper duplicates lifecycle and security behavior.
- Always persisting results would violate Conclave's current no-storage default.

## Data Flow and Safety

The council runs exactly as it does today. The completed `CouncilResult` is converted
once with `_result_to_dict`, encoded as JSON, optionally written to the owner-only
path, and then rendered to stdout or Rich panels. API keys remain outside the result
model and are never written. Because prompts and model answers may be sensitive,
persistence stays explicit and the documentation warns operators to select a secure
local path.

Streaming remains unchanged: `--json-output` is rejected with `--stream`, because the
streaming branch currently returns before constructing the shared rendering path.
Supporting both would enlarge the fix without solving the detached adversarial-run
failure.

## Verification and Rollback

Regression tests cover successful durable output, user-private permissions, identical
stdout/file payloads under `--json`, persistence-failure stdout, portability, and
pre-call rejection of unsupported destinations or streaming. Existing CLI and full
offline suites must remain green, followed by lint and secret scanning. Rollback is a
single option/helper removal; no persisted format, cache, config, or public Python API
changes.
