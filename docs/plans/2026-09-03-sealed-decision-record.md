# Sealed Decision Record Implementation Plan (DSE-1517)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `conclave ask --record decision.ccr` seals a run (prompt, roster, manifest, verdict, and the scrubbed transport tape) under one content digest; `conclave replay decision.ccr` re-derives the whole `CouncilResult` offline with zero network and zero keys; `conclave verify decision.ccr` proves the bundle is intact and that the consensus numbers are arithmetic over the recorded clustering.

**Architecture:** (1) Promote the eval subsystem's sanitizing record/replay transport into a product module (`conclave.tape`) with the evals delegating to it — a pure refactor. (2) Add one replay seam: a `contextvars` override consulted at the top of `transport.post_json`, plus a replay-mode branch in `providers.call_model` that reproduces recorded key presence without reading any env var. (3) A `conclave.record` module owns the bundle format (zip, deterministic), the canonical-form comparison (equality modulo a declared volatile set), the seal gate (whole-bundle forbidden-substring scan), and the three `verify` checks. (4) The bundle binds to the existing secret-free run identity (`cache.build_identity`), never to a new hash. (5) CLI: `--record` on `ask` (buffered modes only), new `replay` and `verify` commands, exit `5` for a failed verification.

**Tech Stack:** Python 3.11+, pydantic v2, stdlib `zipfile` / `hashlib` / `contextvars`, typer, pytest (offline).

**Execution locality:** edit + git on the laptop worktree `~/dev/worktrees/conclave-dse-1517`; run every test/lint on the builder via `~/.claude/scripts/builder-run.sh conclave-dse-1517 '<cmd>'` (rsyncs first). Never run pytest/ruff/pip on the laptop. Shorthand below: `BR='~/.claude/scripts/builder-run.sh conclave-dse-1517'`.

---

## Sequencing and shared surfaces

This ticket lands **third**. The bundle digests `result.json`, which embeds the manifest, so the manifest schema must be frozen first:

- DSE-1512 (PR #63) adds `manifest.adjudication_succession`, `ModelAnswer.failure_category` / `http_status`, `CouncilResult.primary_failed_over`, `CACHE_FORMAT_VERSION = "4"`.
- DSE-1514 adds `manifest.cost_ceiling_usd`, `price_snapshot_digest`, `priced_as_of`, `unpriced_models`, `unpriced_receipts`, `pricing_warnings`, receipt `cost_ceiling_usd` / `cost_basis`, `CACHE_FORMAT_VERSION = "5"`, `max_output_tokens` in `generation_settings`, CLI exit code `4`.

**Before Task 1:** rebase this branch onto `main` after both have merged (`git rebase origin/main`), re-run the full suite on the builder, and record the new baseline in this file's header. Nothing in this plan changes the manifest schema; it only reads it.

## Release classification

**Security-specific for real, not just by vocabulary.** The diff touches one credential path (`providers._resolve_key` gains a replay-mode branch) and creates a new protected-data artifact designed to be handed to third parties. Per `release-control.md`: Round 2 (CSO threat review of this plan) gates Round 3; Round 5 (CSO secret-safety audit of the built artifact) gates Round 6; merge needs one human approval receipt at the exact head SHA. Read-only reviews are unrestricted.

---

## Ground rules (read before Task 1)

- **Never modify** `redact()`, `scan_for_secret_material()`, `verified_secret_safety()`, `_receipt_error_category()`, `registry.py` (`key_present` / `key_source` / `required_env_vars`), or the transport raise sites and their `from None` / `__context__ = None` discipline. The replay seam is *additive*: with the contextvar unset, `post_json` and `call_model` are byte-for-byte the same code path as today.
- **The only credential-path touch** is `providers._resolve_key`: when a replay context is active it returns a fixed non-secret sentinel (`"replay"`). The sentinel is placed in request headers by the adapter exactly like a real key would be, and the replaying transport ignores headers entirely (it hashes `url` + `body` only), so the sentinel never leaves the process. State this in the docstring and in `SECURITY.md`.
- **Zero network, zero env-var reads on replay/verify.** Enforced by a test that unsets every `*_API_KEY`, stubs `transport.post_json` and `transport.stream_sse` to raise, and still replays.
- **A bundle contains the prompt and every answer.** It is secret-free with respect to API keys, not with respect to content. Say so in README and SECURITY.md; never imply otherwise in code comments.
- **Seal gate is fail-closed.** Any forbidden substring anywhere in the serialized bundle members (same set as `manifest._FORBIDDEN_SUBSTRINGS`, `[REDACTED]` allowed) → refuse to seal, exit `1`, nothing written. Any tape request containing a `headers` key or any `_is_sensitive_name` key → refuse.
- **`_AMBIGUOUS_CREDENTIAL_NAMES` (`key`, `token`) stay fail-closed with no opt-out** — that is how `evals/replay.py` already behaves; do not add a flag.
- **Buffered only.** `--stream --record` is rejected before any provider call (exit `2`). `stream_sse` raises in replay mode.
- **Equality modulo the volatile set, nothing else.** See "Volatile fields" below. `replay` names the first diverging JSON path.
- **Digests prove integrity, not authorship.** No signing in v1; document the limitation.
- **TDD.** Failing test → run → minimal code → run → full suite → commit. ruff 0.16 formats Python blocks inside Markdown — keep every ```python block in docs format-clean (100 cols, double quotes, trailing commas); use `text` fences for dedented method fragments.
- Do not touch `cache.py` except to *call* `build_identity` (Task 6 factors a `Council.run_identity()` accessor out of `_cache_key` — behaviour of `_cache_key` unchanged).

---

## Volatile fields (`VOLATILE_FIELDS`)

Canonical form = `result.model_dump(mode="json")` with these removed, recursively, before comparison and before hashing for `verify`'s consensus/provenance checks (the *integrity* check hashes the raw member bytes, not the canonical form):

| JSON path / key | Why it cannot be reproduced |
|---|---|
| `manifest.request_id` | `uuid4().hex` per run |
| any key named `latency_s` | wall-clock (`ModelAnswer`, in every phase collection: `answers`, `rounds[*].answers`, `adversarial.proposal` / `critiques`, `elite.initial_answers` / `critiques` / `revisions`) |
| any key named `latency_ms` | wall-clock (`ProviderExecutionReceipt`) |
| `manifest.total_latency_ms` | sum of the above |
| `manifest.pricing_warnings` | DSE-1514 emits a snapshot-staleness warning that depends on *today's* date relative to `priced_as_of` |
| `cached` | always `False` on a live run and on replay; excluded so a future cache-served record cannot masquerade |

Everything else is deterministic given the tape: answer text and `answer_id` (`stable_answer_id` over name/model/text), usage (from the tape), synthesis, verdict, positions, consensus (arithmetic), receipts (minus latency), the succession ledger, pricing ceilings (from reported usage × a frozen snapshot — the snapshot digest is part of `run_identity` and must match at replay time), `degraded`, `primary_failed_over`.

`VOLATILE_FIELDS` is a module constant in `conclave.record`, written into `bundle.json` so a reader knows what was ignored, and `verify` prints it.

---

## Bundle format (`.ccr`, zip, `conclave_bundle_v1`)

Container: **zip** via stdlib `zipfile`, `ZIP_STORED` (no compressor variance), `ZipInfo.date_time = (1980, 1, 1, 0, 0, 0)`, members written in a fixed order. Digests are over **member contents**, so the zip framing never enters a digest.

| Member | Content |
|---|---|
| `bundle.json` | `{"schema_version": "conclave_bundle_v1", "created_at": <ISO-8601 UTC>, "conclave_version": ..., "mode": ..., "mode_params": {...}, "prompt_fingerprint": <sha256>, "roster": [[name, model_id], ...], "synthesizer_chain": [[name, model_id], ...], "keyed_model_ids": [...], "endpoints": {prefix: sanitized_completions_url}, "generation": {"temperature": ..., "timeout": ..., "max_output_tokens": ...}, "extract_verdict": bool, "run_identity": <the cache.build_identity document>, "run_identity_hash": "sha256:...", "volatile_fields": [...], "member_digests": {"result.json": "sha256:...", "tape.json": "sha256:..."}, "evidence_bundle_digest": null}` |
| `result.json` | exactly the bytes `--json-output` writes (`json.dumps(result.model_dump(mode="json"))` + `\n`) |
| `tape.json` | `{"schema_version": "conclave_tape_v1", "run_identity_hash": "sha256:...", "records": [ {request_hash, occurrence_index, request: {url, body}, status, response}, ... ]}` — sanitized exactly as `evals/replay.py` sanitizes today |
| `DIGEST` | `sha256:` over `"\n".join(f"{name} {digest}" for name, digest in sorted(member_digests.items()))` where the members are `bundle.json` (with `member_digests` filled and `DIGEST` absent), `result.json`, `tape.json` |

`keyed_model_ids` records which model ids had a key at record time (names only — derived via `registry.key_present`, which never returns a value). Replay uses it to reproduce skips and "no API key" errors without the environment.

`endpoints`: for every custom `config.endpoints` prefix used by the roster or chain, the completions URL with **userinfo and query string removed** (`urlsplit` → scheme, netloc-without-userinfo, path). Replay needs it to rebuild the adapter and therefore the request URL that the tape was hashed on. **Threat-model decision (CSO rules in Round 2):** this discloses a custom endpoint's *host and path* to whoever holds the bundle. Alternative: store only `cache._endpoint_fingerprint(url)` and hash tape requests on `(fingerprint, body)`; replay of a custom-endpoint run would then require the operator's own `endpoints:` config to be present and matching. Default in this plan: **record the sanitized URL** (replay works anywhere; host disclosure is documented as a limitation and is far weaker than the prompt content the bundle already carries). If the CSO overrules, Task 5 switches to the fingerprint variant — the seam is one function (`record._endpoint_for_bundle`).

---

## Threat model (input to Round 2)

**Assets:** provider API keys; the prompt and every member/synthesizer answer; custom endpoint hosts; model ids and friendly names; the integrity of the verdict numbers.

**Trust boundary:** a bundle is created by the operator and *handed to a third party*. Assume it is published. Nothing in it is confidential except by the operator's choice to share content.

| Attacker goal | Control | Proof (test) |
|---|---|---|
| Recover an API key from a bundle | Tape stores `{url, body}` only — never headers; exact credential values seen in headers/query/body are replaced with `[REDACTED]` then `redact()`-scrubbed (existing `evals/replay.py` logic, promoted unchanged); seal gate refuses on any forbidden substring or any sensitive-named key anywhere in the bundle | seeded credentials in header / query / body / response body are absent from the sealed bytes (`tests/test_record_secret_safety.py`); `scan`-equivalent over the whole bundle |
| Learn a custom endpoint's credentials | URL userinfo and query stripped before storage; `_AMBIGUOUS_CREDENTIAL_NAMES` dropped | endpoint with `?key=…` and `user:pass@` → neither in the bundle |
| Learn a custom endpoint host | **Accepted, documented** (default) — see bundle format | README/SECURITY.md state it |
| Forge a verdict that passes `verify` | `verify` recomputes `consensus_score` / `consensus_label` and every conflict score from `provider_votes` over responding members via `agreement.consensus()`; a hand-edited score fails check 2 | tamper test: edit `consensus_score` in `result.json` → exit 5, check 2 named |
| Pass `verify` on a tampered bundle | member digests + `DIGEST` over the members; any byte change in `result.json` or `tape.json` fails check 1 | tamper tests: tape edit, result edit, truncation, member removal |
| Make `replay` hit the network or read keys | contextvar transport override consulted before any httpx call; `stream_sse` refuses in replay mode; `_resolve_key` returns a sentinel in replay mode; keyed-ness comes from `keyed_model_ids` | isolation test: all `*_API_KEY` unset, `post_json` / `stream_sse` stubbed to raise → replay succeeds |
| Replay a bundle against a different conclave to make it "lie" | requests are hashed on the sanitized request the *current* code builds; a prompt/schema version change changes the body → unmatched request → loud failure naming the recorded `run_identity.versions` vs the running versions | version-drift test |
| Use `replay` as a key-free way to call providers | replay never reaches httpx; the override *is* the transport | same isolation test |

**Residual risks (documented, accepted):** the bundle discloses prompt + answers by design; digests do not establish authorship (no signing); a custom endpoint host/path is disclosed under the default decision.

---

## Command surface

| Command | Exit codes |
|---|---|
| `conclave ask … --record PATH` | as today for the run (`0` / `1` / `2` / `3` / `4`); `2` if `--stream`, if the parent directory does not exist, or if `PATH` exists and is a directory; `1` if sealing is refused (the run's normal stdout output still happens first, like `--json-output`) |
| `conclave replay PATH [--json]` | `0` replay equals the record modulo volatile fields; `5` divergence / unmatched request / leftover record; `2` unreadable or incompatible bundle |
| `conclave verify PATH [--json]` | `0` all three checks pass; `5` any check fails (each check reported); `2` unreadable or incompatible bundle |

`--record` composes with `--json-output` (both write). `replay --json` prints the re-derived `CouncilResult`; `verify --json` prints `{"integrity": ..., "consensus": ..., "provenance": ..., "volatile_fields": [...]}`.

---

## Round 3 — record + replay

### Task 1: Promote the sanitizing tape into `conclave.tape` (pure refactor)

**Files:**
- Create: `src/conclave/tape.py`
- Modify: `src/conclave/evals/replay.py` (delegate; public names unchanged)
- Test: `tests/test_tape.py` (new); `tests/evals/test_replay.py` must pass **unchanged**

**What moves:** `_CREDENTIAL_NAMES`, `_AMBIGUOUS_CREDENTIAL_NAMES`, `_is_sensitive_name`, `_redact_exact`, `_sanitize`, `_sanitize_url`, `_string_values`, `_credentials`, `_hash_request`, `_sanitize_stored_request`, `_request`, and the `PostJson` alias. `ReplayRecord` moves too (product name `TapeRecord`, identical fields; `evals.replay.ReplayRecord = TapeRecord` alias). `ReplayArtifact` (bound to `base_manifest_hash`, schema `conclave_replay_v1`) stays in evals, importing the helpers. Add the product artifact:

```python
TAPE_SCHEMA_VERSION = "conclave_tape_v1"


class Tape(BaseModel):
    """Sanitized request/response records bound to one run identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["conclave_tape_v1"] = TAPE_SCHEMA_VERSION
    run_identity_hash: Sha256Digest
    records: tuple[TapeRecord, ...]
    # validate_integrity: same rules as ReplayArtifact (sanitized, hashes match, contiguous occurrences)
```

`RecordingTransport(delegate, *, run_identity_hash)` and `ReplayingTransport(tape, *, run_identity_hash)` are the product twins of `RecordingPostJson` / `ReplayingPostJson` (same `__call__` signature as `transport.post_json`; `assert_consumed()`; `ReplayMismatchError` / `ReplayCompatibilityError` re-exported from evals for compatibility). `Sha256Digest` moves to `tape.py` with `evals.models.Sha256Digest` re-exporting it.

**Step 1 (failing tests, `tests/test_tape.py`):** import the moved names from `conclave.tape`; seeded-credential matrix (header bearer, `x-api-key`, query `key=`, body `api_key`, body `token`, url userinfo) → absent from `RecordingTransport` output; `Tape.model_validate` rejects an unsanitized record, a wrong hash, and non-contiguous occurrences; `ReplayingTransport` raises on an unmatched request and on `assert_consumed()` with leftovers; `evals.replay.ReplayRecord is conclave.tape.TapeRecord`.

**Step 2:** `$BR '.venv/bin/python -m pytest -q tests/test_tape.py'` → FAIL (`ImportError`).

**Step 3:** move the code; `evals/replay.py` becomes imports + `ReplayArtifact` + the two eval classes delegating to the product ones (or subclassing). No behaviour change.

**Step 4:** `$BR '.venv/bin/python -m pytest -q tests/test_tape.py tests/evals/test_replay.py'` → PASS; full suite → 0 failures; `git diff --stat -- tests/evals` empty; ruff clean.

**Step 5:** `git commit -m "refactor(tape): promote the sanitizing record/replay transport out of evals (DSE-1517)"`

---

### Task 2: The replay seam in `transport` and `providers`

**Files:**
- Modify: `src/conclave/transport.py` (contextvar + `post_json` head + `stream_sse` guard)
- Modify: `src/conclave/providers.py` (`_resolve_key`, `call_model`)
- Test: `tests/test_replay_seam.py` (new)

**Implementation:**

```python
# transport.py
@dataclass(frozen=True)
class ReplayContext:
    """Active only inside ``conclave replay``. Never constructed on a live run."""

    post_json: PostJson
    keyed_model_ids: frozenset[str]
    key_sentinel: str = "replay"


_REPLAY: ContextVar[ReplayContext | None] = ContextVar("conclave_replay", default=None)


def replay_context() -> ReplayContext | None:
    return _REPLAY.get()


@contextmanager
def replaying(context: ReplayContext) -> Iterator[None]:
    token = _REPLAY.set(context)
    try:
        yield
    finally:
        _REPLAY.reset(token)
```

At the very top of `post_json`: `ctx = _REPLAY.get(); if ctx is not None: return await ctx.post_json(url, headers, json_body, timeout)` — before `_get_client()`, so no httpx client is created. At the top of `stream_sse`: `if _REPLAY.get() is not None: raise TransportError("streaming is not available during replay", category="unexpected")` (buffered-only contract). Everything below the two guards is untouched.

`providers.py`:
```text
def _resolve_key(adapter):
    ctx = transport.replay_context()
    if ctx is not None:
        # Replay: the replaying transport IS the network; the sentinel is placed
        # in the request headers by the adapter and ignored by the replayer
        # (it hashes url + body only). No environment variable is read.
        return ctx.key_sentinel
    ... existing loop unchanged ...
```
In `call_model` (and `call_model_stream`), immediately before `api_key = _resolve_key(adapter)`:
```text
    ctx = transport.replay_context()
    if ctx is not None and model_id not in ctx.keyed_model_ids:
        # Reproduce the recorded "no key" outcome without touching the environment.
        names = " or ".join(adapter.env_vars) or "(none)"
        msg = f"no API key in environment (set {names})"
        return ModelAnswer(name=name, model_id=model_id, latency_s=..., error=msg, failure_category="unkeyed")
```
(the message must be byte-identical to the live branch — factor the existing string into a helper used by both).

**Tests:** seam unset → `post_json` reaches the (stubbed) httpx client exactly as before; seam set → override called, `_get_client` never invoked (patch `transport._get_client` to raise); `stream_sse` raises in replay mode; `_resolve_key` returns the sentinel in replay mode and reads no env var (patch `os.environ` with a `MagicMock` that raises on `get`); `call_model` reproduces the unkeyed message for a model id outside `keyed_model_ids`; the security suites (`tests/test_keyleak_audit.py`, `tests/test_transport.py`) unchanged and green.

**Commit:** `feat(transport): contextvar replay seam; replay-mode key sentinel (DSE-1517)`

---

### Task 3: `Council` replay awareness

**Files:**
- Modify: `src/conclave/council.py` (`__init__(replay: ReplayContext | None = None)`, `_key_present`, `run_identity`, `_available_members`, `adjudicate`, `_chain_unkeyed_error`)
- Test: `tests/test_council_replay.py` (new)

- `Council._key_present(model_id)` → `model_id in self._replay.keyed_model_ids` when replaying, else `registry.key_present(model_id)`. Replace the three `key_present(...)` call sites in `council.py` with `self._key_present(...)`. `registry.key_present` itself is untouched.
- `Council.run_identity(prompt, mode, *, rounds=None, proposer=None, converge_threshold=None, choices=None) -> dict` — factor the argument assembly out of `_cache_key` so both call `cache.build_identity` with identical arguments; `_cache_key` = `sha256(canonical json of run_identity(...))` exactly as before (a test asserts the key is unchanged for a fixed input vs. the pre-refactor value).
- `Council.keyed_model_ids()` → sorted list of model ids in roster ∪ chain for which `registry.key_present` is true (names only).
- Replay councils force `cache=False` (never read or write the cache during replay).

**Tests:** key presence comes from the context in replay mode (env fully unset); `_cache_key` unchanged; `keyed_model_ids` returns names only (assert no env value appears in the output even when the key value is set to a distinctive string).

**Commit:** `feat(council): replay-aware key presence, run_identity accessor (DSE-1517)`

---

### Task 4: `conclave.record` — canonical form, bundle writer, seal gate

**Files:**
- Create: `src/conclave/record.py`
- Test: `tests/test_record_bundle.py` (new)

```python
BUNDLE_SCHEMA_VERSION = "conclave_bundle_v1"
BUNDLE_SUFFIX = ".ccr"
VOLATILE_KEYS = frozenset({"latency_s", "latency_ms"})
VOLATILE_PATHS = (
    ("manifest", "request_id"),
    ("manifest", "total_latency_ms"),
    ("manifest", "pricing_warnings"),
    ("cached",),
)


def canonical_result(payload: dict) -> dict:
    """Return a deep copy with every volatile field removed (see the plan table)."""


def first_divergence(a: dict, b: dict) -> str | None:
    """Return the first differing JSON path (``answers[1].answer``) or ``None``."""


def sanitized_endpoint(url: str) -> str:
    """scheme://host/path — userinfo and query removed. Never raw."""


def seal_bundle(*, path: Path, result_payload: dict, tape: Tape, bundle_meta: dict) -> None:
    """Write PATH atomically (mkstemp + fsync + os.replace, 0600) after the seal gate."""


class SealRefused(RuntimeError):
    """Raised when the bundle would contain secret-shaped material."""


def read_bundle(path: Path) -> LoadedBundle:
    """Parse and structurally validate; never partial-load; raises BundleError."""
```

Seal gate (inside `seal_bundle`, before any byte is written): serialize each member; lower-case; refuse if any of `manifest._FORBIDDEN_SUBSTRINGS` appears (import the tuple — do not copy it — so the two scans can never drift) outside the literal `[REDACTED]` marker; refuse if any tape record's `request` has a `headers` key or any key for which `tape._is_sensitive_name` is true. Digests: `sha256:` hex over member bytes; `DIGEST` as specified. Write order: `bundle.json`, `result.json`, `tape.json`, `DIGEST`.

**Tests:** canonical form strips exactly the volatile set (build a `CouncilResult` with latencies and a request id; assert the stripped keys and that nothing else changed); `first_divergence` names nested paths; two seals of identical inputs produce identical bytes; `0600` on POSIX; `read_bundle` round-trips; seal refused on a seeded `sk-…` in an answer, on a `bearer` substring in a response body, on a `headers` key in a tape request; `sanitized_endpoint("https://u:p@h/v1/chat?key=abc")` → `https://h/v1/chat`.

**Commit:** `feat(record): bundle format, canonical form, seal gate (DSE-1517)`

---

### Task 5: `--record` on `conclave ask`

**Files:**
- Modify: `src/conclave/cli.py` (`ask` option + wiring), `src/conclave/council.py` (a `recording(...)` helper is NOT needed — recording is done by installing a `RecordingTransport` through the Task 2 seam with `keyed_model_ids = council.keyed_model_ids()` and the live `transport.post_json` as delegate)
- Test: `tests/test_cli_record.py` (new)

Wiring in `ask`: validate (`--stream` → exit 2; parent dir; not a directory); build `run_identity = c.run_identity(prompt, mode, …)` and its hash; run the mode inside `transport.replaying(ReplayContext(post_json=RecordingTransport(live_post_json, run_identity_hash=…), keyed_model_ids=frozenset(c.keyed_model_ids())))` — note this is the *recording* use of the same seam: keyed-ness is the live one, the delegate is the live transport; then after the normal stdout rendering and `--json-output`, `seal_bundle(...)`; on `SealRefused` print the reason to stderr and exit `1` (stdout output already happened, mirroring the `--json-output` failure contract).

`bundle_meta` = everything in the "Bundle format" table: mode, mode params, roster pairs, chain pairs, keyed ids, sanitized endpoints for used prefixes, generation settings (incl. `max_output_tokens`), `extract_verdict`, `run_identity`, `run_identity_hash`, `volatile_fields`, `conclave_version`, `created_at`.

**Tests (offline, council seam scripted; transport seam: the recording transport wraps a fake `post_json`):** each mode (`synthesize`, `raw`, `debate`, `adversarial`, `vote`, `elite`) writes a bundle whose `result.json` bytes equal `--json-output`'s; a degraded run and an incomplete Elite run record fine; `--stream --record` → exit 2 before any call; `--record` + `--json-output` both written; without `--record` the CLI output is byte-identical (snapshot test); seal refusal → exit 1 with stdout intact.

**Commit:** `feat(cli): conclave ask --record seals a decision record (DSE-1517)`

---

### Task 6: `conclave replay`

**Files:**
- Modify: `src/conclave/record.py` (`replay_bundle(path) -> ReplayOutcome`), `src/conclave/cli.py` (new command)
- Test: `tests/test_replay_command.py` (new)

`replay_bundle`: `read_bundle` → rebuild `ConclaveConfig(models=dict(roster + chain), synthesizer=chain[0], synthesizer_chain=[names], endpoints={prefix: CustomEndpoint(completions_url=sanitized, env_var="CONCLAVE_REPLAY_UNUSED")})` → `Council(models=[names], synthesizer=[chain names], config=cfg, temperature=…, timeout=…, cache=False, extract_verdict=…, max_output_tokens=…, replay=ReplayContext(post_json=ReplayingTransport(tape, run_identity_hash=…), keyed_model_ids=frozenset(keyed)))` → assert `council.run_identity(...) == bundle.run_identity` (else `BundleError("run identity mismatch: recorded versions {…} vs running {…}")`) → run the recorded mode with the recorded params (`ask_sync` / `debate_sync` / `adversarial_sync` / `vote_sync` / `elite_sync`) under `transport.replaying(...)` → `transport.assert_consumed()` → compare `canonical_result(replayed)` to `canonical_result(recorded)` → `ReplayOutcome(result, divergence: str | None)`.

CLI: `replay PATH [--json]` — prints the re-derived result (renderer of the recorded mode, or JSON), exit `0` when `divergence is None`, else prints `replay diverged at <path>` and exits `5`; `ReplayMismatchError` / leftover records → exit `5` with the call named; `BundleError` → exit `2`.

**Tests:** offline isolation (unset every `*_API_KEY`; `monkeypatch.setattr(transport, "_get_client", raise)`; stub `stream_sse` to raise) → replay of each mode's bundle exits 0; a bundle whose `result.json` was edited (answer text) → exit 5 naming `answers[0].answer`; a bundle recorded with an extra tape record → leftover → exit 5; a missing record → unmatched → exit 5; edited `run_identity.versions.synthesis_prompt` → exit 2 with the version message; schema version `conclave_bundle_v0` → exit 2, nothing else attempted.

**Commit:** `feat(cli): conclave replay re-derives a decision record offline (DSE-1517)`

---

## Round 4 — verify

### Task 7: Public consensus re-derivation

**Files:**
- Modify: `src/conclave/verdict_synthesis.py` (promote `_conflict_score` → `conflict_score`; add `recompute_consensus`)
- Test: `tests/test_recompute_consensus.py` (new)

```python
def recompute_consensus(
    verdict: CouncilVerdict, responders: Sequence[ModelAnswer]
) -> tuple[float | None, str, list[float | None]]:
    """Re-derive the top-level score/label and every conflict score from
    ``verdict.provider_votes`` over ``responders`` (DD-1). Pure arithmetic."""
    votes = {v.provider: v.position_label for v in verdict.provider_votes}
    sequence = [votes.get(a.name) for a in responders]
    score, label = agreement.consensus(sequence)
    return score, label, [conflict_score(sequence, c.position_labels) for c in verdict.conflicts]
```
`_assemble_verdict` keeps its behaviour (a test asserts `recompute_consensus(assembled, responders)` reproduces the assembled numbers exactly). Responders for a result = `_responding(result.answers)` (promote as `responding_answers`).

**Tests:** reproduces `_assemble_verdict`'s numbers on the existing fixtures; detects a hand-edited `consensus_score`; detects an edited conflict score; `None` handling for N<2.

**Commit:** `feat(verdict): public recompute_consensus for third-party verification (DSE-1517)`

---

### Task 8: `conclave verify`

**Files:**
- Modify: `src/conclave/record.py` (`verify_bundle(path) -> VerifyReport`), `src/conclave/cli.py` (new command, `_VERIFY_FAILED_EXIT_CODE = 5`)
- Test: `tests/test_verify_command.py`, `tests/test_record_tamper.py` (new)

`VerifyReport{integrity: CheckResult, consensus: CheckResult, provenance: CheckResult, volatile_fields: list[str]}` with `CheckResult{ok: bool, detail: str}` (bounded, no provider text).

1. **Integrity:** recompute every member digest and `DIGEST`; any mismatch or missing member → fail, naming the member. Runs first and alone if it fails (a corrupt bundle is not parsed further).
2. **Consensus:** load `result.json`; if `verdict` is `None` → ok with detail `"no verdict recorded (reason: <manifest.verdict_absent_reason>)"`; else `recompute_consensus(verdict, responding_answers(result))` must equal `verdict.consensus_score/label`, the hoisted `result.consensus_*` mirrors, and each `conflict.consensus_score` — else fail naming the first mismatching field and the expected vs found values.
3. **Provenance:** when a verdict is present, `manifest.verdict_extraction.model_id` and `.prompt_version` are non-null and `manifest.consensus_method == "position_cluster_ratio_v1"`; always: `manifest.secret_safety == "verified_no_secrets"`, `manifest.request_id` present, `run_identity_hash` in `bundle.json` equals `sha256` of the canonical `run_identity` document.

CLI: `verify PATH [--json]` prints the three lines (`integrity: ok`, …) and the ignored volatile set; exit `0` iff all ok, else `5`; unreadable → `2`.

**Tamper suite (`tests/test_record_tamper.py`):** build a good bundle once (fixture), then for each: rewrite `result.json` with `consensus_score` changed → check 2 fails; change one conflict score → check 2; flip a `provider_votes[*].position_label` → check 2 (score no longer matches); edit a tape response → check 1; delete `tape.json` → check 1; truncate the zip → exit 2; change `schema_version` → exit 2; set `secret_safety` to `unverified` → check 3.

**Commit:** `feat(cli): conclave verify — integrity, consensus re-derivation, provenance (DSE-1517)`

---

### Task 9: Cross-mode and secret-safety coverage

**Files:** `tests/test_record_secret_safety.py` (new); extend `tests/test_keyleak_audit.py` and `tests/test_secret_safety_matrix.py` to the bundle surface.

- Seeded-credential matrix through the REAL `call_model` path with a fake `post_json` that echoes request fragments into the response body: `Authorization: Bearer sk-live-…`, `x-api-key`, query `?key=`, body `api_key`, body `token`, url userinfo — none present in any bundle member; seal passes because the scrub happened; then a second variant where the *answer text* contains `sk-…` → seal refused (exit 1).
- Custom endpoint round-trip: record with `endpoints: {"together": {completions_url: "https://u:p@api.together.xyz/v1/chat/completions?key=abc"}}` → bundle stores `https://api.together.xyz/v1/chat/completions`; replay succeeds with no `endpoints:` config present.
- `replay` and `verify` never read `os.environ` for keys (patch `os.environ.get` to fail for any name ending in `_API_KEY`).
- Every mode × {clean, degraded} × {record → replay → verify} end to end.

**Commit:** `test(record): seeded-credential matrix, custom endpoint, cross-mode round trips (DSE-1517)`

---

### Task 10: Documentation

- **README.md** — new "Decision records" section: the three commands, what a bundle contains (*the prompt and every answer* — secret-free w.r.t. keys, not content), the volatile-set rule, exit codes, the endpoint-host note, "digests prove integrity, not authorship".
- **SECURITY.md** — a "Decision records (`.ccr`)" subsection under the threat model: the seal gate, the replay-mode sentinel and why it never leaves the process, the one credential-path touch (`_resolve_key`), the accepted limitations (content disclosure; endpoint host; no signing), and that `replay`/`verify` are network- and key-free by construction.
- **docs/PRODUCT_DESIGN_DOCUMENT.md** — §4a: "Sealed decision records (v1.5)" — the verifiable-verdict claim becomes third-party checkable; §9: H2 substrate delivered early, `evidence_bundle_digest` reserved.
- **CHANGELOG.md** `[Unreleased]` — Added (three commands, `conclave.tape`, `conclave.record`, `recompute_consensus`, exit code 5), Changed (evals delegate to `conclave.tape`, no behaviour change), Security (the `_resolve_key` replay branch; seal gate), Not changed (no signing; streaming not recordable).
- **DOCUMENTATION_INDEX.md** — link this plan. `config.example.yml` — nothing (no new config).

**Commit:** `docs: decision records — README, SECURITY.md threat model, PDD §4a/§9, changelog (DSE-1517)`

---

## Ship

1. Push; `gh pr create` (title `feat: sealed decision record — record / replay / verify (DSE-1517)`), body: summary, bundle format, threat-model table, the one credential-path touch, exit codes, "not changed", `Closes DSE-1517`.
2. CI green (py3.11/3.12/3.13, ruff, pip-audit, gitleaks).
3. Round 5: `chief-security-officer` seeded-credential audit against the *built* artifact (the Task 9 matrix plus their own probes); blocking.
4. `release_control.py classify` → `security-specific`; one human receipt at the exact head SHA; `release_control.py merge --method squash`.
5. Linear DSE-1517 → Done with the merge SHA.
