# Sealed Decision Record Implementation Plan (DSE-1517) — v2, amended per the Round 2 CSO threat review

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `conclave ask --record decision.ccr` seals a run (prompt, roster, manifest, verdict, and the scrubbed transport tape) under one content digest; `conclave replay decision.ccr` re-derives the whole `CouncilResult` offline with zero network and zero keys; `conclave verify decision.ccr` proves the bundle is internally consistent — intact members, consensus numbers that are arithmetic over the recorded clustering, and complete provenance.

**Architecture:** (1) Promote the eval subsystem's sanitizing record/replay transport into a product module (`conclave.tape`) — a pure refactor — then **harden it** (URL userinfo, the wider secret-query set the cache already uses, response root keys). (2) One transport seam: a `ContextVar` consulted at the top of `transport.post_json`, with **two distinct context types** — `RecordingContext` (live keys, live network, wraps `post_json`) and `ReplayContext` (offline: no keys, no network) — plus a strict process-global backstop so a thread hop can never turn a replay into a live call. (3) `conclave.record` owns the bundle format (zip, deterministic, hardened parser), the canonical-form comparison (volatile fields *replaced by a sentinel*, never deleted), the two-tier seal gate, and the `verify` checks. (4) The bundle binds to the existing secret-free run identity (`cache.build_identity`); custom endpoints are recorded as **fingerprints by default**. (5) CLI: `--record` on `ask` (buffered modes, cache forced off), new `replay` and `verify` commands, exit `5` for a failed verification.

**Tech Stack:** Python 3.11+, pydantic v2, stdlib `zipfile` / `hashlib` / `contextvars`, typer, pytest (offline).

**Execution locality:** edit + git on the laptop worktree `~/dev/worktrees/conclave-dse-1517`; run every test/lint on the builder via `~/.claude/scripts/builder-run.sh conclave-dse-1517 '<cmd>'` (rsyncs first). Never run pytest/ruff/pip on the laptop. Shorthand: `BR='~/.claude/scripts/builder-run.sh conclave-dse-1517'`.

**Round 2 outcome (2026-09-04):** CSO verdict YELLOW — Round 3 may start after the amendments now folded into this v2 (findings F1–F16 and three rulings). The two CSO probes that could not run in Round 2 are Task 1b's first two tests.

---

## Sequencing and shared surfaces

This ticket lands **third**. The bundle digests `result.json`, which embeds the manifest, so the manifest schema must be frozen first:

- DSE-1512 (PR #63) adds `manifest.adjudication_succession`, `ModelAnswer.failure_category` / `http_status`, `CouncilResult.primary_failed_over`, `CACHE_FORMAT_VERSION = "4"`.
- DSE-1514 adds `manifest.cost_ceiling_usd`, `price_snapshot_digest`, `priced_as_of`, `unpriced_models`, `unpriced_receipts`, `pricing_warnings`, receipt `cost_ceiling_usd` / `cost_basis`, `CACHE_FORMAT_VERSION = "5"`, `max_output_tokens` in `generation_settings`, CLI exit code `4`.

**Tasks 1 and 1b touch only `evals/replay.py` and new files** and may run before the rebase. **Before Task 2:** rebase onto `main` after both PRs have merged, re-run the full suite on the builder, and record the new baseline here. Nothing in this plan changes the manifest schema; it only reads it.

## Release classification

**Security-specific for real.** The diff touches one credential path (`providers._resolve_key` gains an *offline-context* branch) and creates a new protected-data artifact designed to be handed to third parties. Round 2 (CSO plan review) — done, YELLOW, amendments applied. Round 5 (CSO secret-safety audit of the built artifact) gates Round 6; merge needs one human approval receipt at the exact head SHA.

---

## Ground rules (read before Task 1)

- **Never modify** `redact()`, `scan_for_secret_material()`, `verified_secret_safety()`, `_receipt_error_category()`, `registry.py`, or the transport raise sites and their `from None` / `__context__ = None` discipline. With no transport context set, `post_json`, `stream_sse`, `_resolve_key`, and `call_model` are byte-for-byte today's code path.
- **Two context types, one seam.** `RecordingContext` (`offline=False`): live keys, live network; only wraps `post_json` to capture the tape. `ReplayContext` (`offline=True`): `_resolve_key` returns the fixed non-secret sentinel `"replay"`; `stream_sse` refuses; key presence comes from `result.json`. The sentinel branch is reachable **only** when `ctx.offline` is true — a test asserts `_resolve_key` reads the real env var under a `RecordingContext`. The sentinel is placed in headers by the adapter like a real key would be and the replaying transport ignores headers (it hashes `url` + `body`), so it never leaves the process. There is no adapter that puts a credential in the URL or body (verified: Gemini uses the `x-goog-api-key` header, not `?key=`).
- **Strict backstop.** `replaying(ctx)` also increments a process-global `_OFFLINE_STRICT` counter; while it is non-zero and the current context has no `ReplayContext`, `post_json` / `stream_sse` **raise** instead of reaching httpx. The CLI `replay` command runs strict; `replay_bundle(..., strict=False)` exists for the embedded/concurrent case and is documented as the caller's responsibility.
- **Zero network, zero env-var reads on replay/verify.** Enforced by a test that unsets every `*_API_KEY`, patches `transport._get_client` to raise, stubs `stream_sse` to raise, and still replays; and by the strict backstop test (a `run_in_executor` hop inside a strict replay raises).
- **A bundle contains the prompt, every answer, and up to 500 chars of redacted provider error text per failed call.** It is secret-free with respect to API keys, not with respect to content. README and SECURITY.md say so; `--record` prints a stderr warning when any answer carries an error.
- **Seal gate is two-tier and fail-closed (Ruling 2).** Tier A refuses; Tier B warns. Details in Task 4. There is no bypass flag — the remedy for a Tier A hit is to re-run or change the source.
- **`_AMBIGUOUS_CREDENTIAL_NAMES` (`key`, `token`) stay fail-closed at body root and query, with no opt-out.** Nested ambiguous names are *not* dropped (dropping `token` inside `messages[*].content` would corrupt the tape) — `_is_sensitive_name` still applies at every depth.
- **Custom endpoints: fingerprint by default (Ruling 1).** Built-in prefixes record their public constant URL verbatim. Custom prefixes record `{fingerprint, scheme, port}` only; replay needs the operator's own `endpoints:` config or `replay --endpoint <prefix>=<url>`, checked by recomputing `cache._endpoint_fingerprint(url)` against `run_identity.endpoint_fingerprints`. `--disclose-endpoints` opts into the sanitized `scheme://host[:port]/path` with a stderr notice naming each host. Document that the fingerprint is sha256 over a low-entropy preimage: it defeats bulk scraping, it is not confidentiality.
- **`--record` forces `cache=False`** (F8). A decision record describes a run that happened; a cache hit records nothing.
- **Equality modulo the volatile set, values replaced by a sentinel, never deleted** (F9). `cached` is **not** volatile.
- **Digests prove integrity, not authorship.** No signing in v1. The docs carry the two "what `verify` proves" sentences verbatim (Task 10) and state: run `verify` first, then `replay`; `replay` alone is not tamper detection.
- **TDD.** Failing test → run → minimal code → run → full suite → commit. ruff 0.16 formats Python blocks inside Markdown — keep every ```python block format-clean; `text` fences for dedented fragments.
- Do not touch `cache.py` except to *call* `build_identity` / `_endpoint_fingerprint` and to *import* `_SECRET_QUERY_KEYS`, `_SECRET_QUERY_PARTS`, `_SECRET_VALUE_MARKERS` (Task 1b).

---

## Volatile fields (`VOLATILE_FIELDS`)

Canonical form = `result.model_dump(mode="json")` with each volatile value **replaced** by the sentinel `"<volatile>"` (scalars) or `["<volatile>"]` (lists) — so a structural edit (a missing key, a different list shape) still diverges. Used for the `replay` comparison only; `verify`'s integrity check hashes raw member bytes.

| JSON path / key | Why it cannot be reproduced |
|---|---|
| `manifest.request_id` | `uuid4().hex` per run |
| any key named `latency_s` | wall-clock (`ModelAnswer`, in every phase collection) |
| any key named `latency_ms` | wall-clock (`ProviderExecutionReceipt`) |
| `manifest.total_latency_ms` | sum of the above |
| `manifest.pricing_warnings` | DSE-1514's snapshot-staleness warning depends on *today's* date; if 1514 ships a bounded category alongside the string, canonicalize on the category instead and keep it in the comparison |

`cached` stays in the comparison (F9). Everything else is deterministic given the tape. `VOLATILE_FIELDS` is a constant in `conclave.record`, written into `bundle.json`, printed by `verify`, and documented as *unverified content*.

---

## Bundle format (`.ccr`, zip, `conclave_bundle_v1`)

Container: **zip**, `ZIP_STORED`, `ZipInfo.date_time = (1980, 1, 1, 0, 0, 0)`, members in fixed order. Digests are over member **bytes**.

| Member | Content |
|---|---|
| `bundle.json` | `schema_version: "conclave_bundle_v1"`, `created_at` (ISO-8601 UTC), `conclave_version`, `mode`, `mode_params`, `prompt_fingerprint` (`cache._digest(prompt)`), `roster: [[name, model_id], …]`, `synthesizer_chain: [[name, model_id], …]`, `endpoints: {prefix: {"kind": "builtin", "url": …} \| {"kind": "custom", "fingerprint": "sha256:…", "scheme": …, "port": …} \| {"kind": "custom_disclosed", "url": sanitized}}`, `generation` (`temperature`, `timeout`, `max_output_tokens`), `extract_verdict`, `run_identity` (the `cache.build_identity` document), `run_identity_hash`, `volatile_fields`, `member_digests` (`result.json`, `tape.json`), `evidence_bundle_digest: null` (reserved) |
| `result.json` | exactly the bytes `--json-output` writes |
| `tape.json` | `{"schema_version": "conclave_tape_v1", "run_identity_hash": …, "records": [{request_hash, occurrence_index, request: {url, body}, status, response}, …]}` |
| `DIGEST` | see the pinned spec |

**Pinned `DIGEST` spec (F7 — goes into `SECURITY.md` verbatim with a worked example):** every JSON member is serialized as `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"` in UTF-8. `member_digests[name] = "sha256:" + hex(sha256(member_bytes))` for `result.json` and `tape.json`; these two are written into `bundle.json` before it is serialized; then `member_digests["bundle.json"]` is computed over the final `bundle.json` bytes. `DIGEST` (the member) contains `"sha256:" + hex(sha256("\n".join(f"{name} {digest}" for name, digest in sorted(all_three.items())).encode("utf-8")))` + `"\n"`.

**Key presence at replay** is derived from `result.json` (F10): a model id is "unkeyed" iff it appears in `manifest.providers_skipped[*]` (by name → model id via the roster) or any answer for it carries `failure_category == "unkeyed"`. No `keyed_model_ids` field.

---

## Threat model (Round 2 — amended)

**Assets:** provider API keys; the prompt, every answer, and redacted provider error text; custom endpoint hosts (and any credential an operator embedded in an endpoint URL); model ids and friendly names; the integrity of the verdict numbers.

**Trust boundary:** a bundle is created by the operator and *handed to a third party*. Assume it is published.

| Attacker goal | Control | Proof (test) |
|---|---|---|
| Recover an API key from a bundle | tape stores `{url, body}` only; exact credential values seen in headers/query/body replaced with `[REDACTED]` then `redact()`; **Tier A seal gate**: `redact(serialized) != serialized` refuses (the redactor masks the *live* env-var values, which are present at record time by definition), key-prefix list, structural checks | seeded credentials in header / query / body / response / **URL userinfo** absent from the sealed bytes; Tier A refusal on a seeded live key value inside an answer |
| Learn a credential embedded in a custom endpoint URL | **userinfo stripped** (Task 1b), the cache's wider `_SECRET_QUERY_*` set applied to tape URLs, Tier A refuses any tape URL whose netloc contains `@`; `CustomEndpoint` rejects userinfo at config load | Task 1b tests + Task 9 |
| Learn a custom endpoint host | fingerprint by default; disclosure only with `--disclose-endpoints` and a stderr notice | Task 5 tests |
| Detect a *partial* edit — a score changed without its votes | `verify` recomputes the consensus score/label and every conflict score from `provider_votes`; cross-member binding (prompt fingerprint, mode, roster, identity hash) | tamper suite |
| Detect accidental corruption, truncation, member removal, extra members | member digests + `DIGEST`; exact member-set check; size / ratio / encryption / decoding rejections | tamper + hostile-zip suite |
| Make `replay` hit the network or read keys | contextvar override before any httpx call; `stream_sse` refuses; `_resolve_key` sentinel only under an offline context; strict backstop for thread hops | isolation + strict tests |
| Replay a bundle against a different conclave to make it "lie" | requests hashed on the sanitized request the *current* code builds; a prompt/schema change → unmatched request → loud failure naming recorded vs running versions | version-drift test |
| **Forge a bundle that passes `verify`** — re-derive the votes and re-seal every digest | **Not detected in v1 — no signing.** Unsigned digests are not tamper-evidence against anyone able to rewrite the whole bundle. Stated verbatim in the docs. | — |

**Residual risks (documented, accepted):** content disclosure by design; up to 500 chars of redacted provider error text per failed call (`--record` warns); a low-entropy endpoint fingerprint; no signing — treat a `.ccr` as authoritative only when it reached you over a channel you already trust (`evidence_bundle_digest` is the reserved slot).

---

## Command surface

| Command | Exit codes |
|---|---|
| `conclave ask … --record PATH [--disclose-endpoints]` | run codes as today; `2` if `--stream`, parent dir missing, `PATH` is a directory or a symlink, or `PATH.parent` is not a directory; `1` if sealing is refused (stdout output already happened) |
| `conclave replay PATH [--json] [--endpoint PREFIX=URL]… [--no-strict]` | `0` equal modulo volatile; `5` divergence / unmatched request / leftover record; `2` unreadable, incompatible, identity mismatch, or a custom endpoint neither configured nor supplied |
| `conclave verify PATH [--json]` | `0` integrity + consensus (or n/a) + provenance pass; `5` any failed check; `2` unreadable / incompatible |

`verify` renders `consensus: n/a (no verdict recorded)` when there is no verdict — never `ok` for a check that did not run (F5b).

---

## Round 3 — record + replay

### Task 1: Promote the sanitizing tape into `conclave.tape` (pure refactor)

**Files:** create `src/conclave/tape.py`; modify `src/conclave/evals/replay.py` (delegate, public names unchanged); test `tests/test_tape.py` (new); `tests/evals/test_replay.py` must pass **unchanged**.

**What moves:** `_CREDENTIAL_NAMES`, `_AMBIGUOUS_CREDENTIAL_NAMES`, `_is_sensitive_name`, `_redact_exact`, `_sanitize`, `_sanitize_url`, `_string_values`, `_credentials`, `_hash_request`, `_sanitize_stored_request`, `_request`, `PostJson`, `ReplayRecord` (product name `TapeRecord`; `evals.replay.ReplayRecord = TapeRecord`), `Sha256Digest` (evals re-exports). `ReplayArtifact` stays in evals. Product artifact + transports:

```python
TAPE_SCHEMA_VERSION = "conclave_tape_v1"


class Tape(BaseModel):
    """Sanitized request/response records bound to one run identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["conclave_tape_v1"] = TAPE_SCHEMA_VERSION
    run_identity_hash: Sha256Digest
    records: tuple[TapeRecord, ...]
    # validate_integrity: identical rules to ReplayArtifact
```

`RecordingTransport(delegate, *, run_identity_hash)` / `ReplayingTransport(tape, *, run_identity_hash)` mirror the eval classes (`__call__` has `post_json`'s signature; `assert_consumed()`; the two error classes live here and evals import them).

**Tests (`tests/test_tape.py`):** moved names import; seeded-credential matrix (bearer header, `x-api-key`, `x-goog-api-key`, query `key=`, body `api_key`, body `token`) absent from `RecordingTransport` output; `Tape` rejects an unsanitized record, a wrong hash, non-contiguous occurrences; `ReplayingTransport` raises on an unmatched request and on leftovers; `evals.replay.ReplayRecord is conclave.tape.TapeRecord`.

Run: `$BR '.venv/bin/python -m pytest -q tests/test_tape.py tests/evals/test_replay.py'` → PASS; full suite → 0 failures; `git diff --stat -- tests/evals` empty; ruff clean.

**Commit:** `refactor(tape): promote the sanitizing record/replay transport out of evals (DSE-1517)`

---

### Task 1b: Harden the promoted sanitizer (CSO F2, F3, F15)

**Files:** modify `src/conclave/tape.py`, `src/conclave/config.py` (`CustomEndpoint` validator); test `tests/test_tape.py` (append).

**Step 1 — the two CSO probes as failing tests:**

```python
def test_urlsplit_keeps_userinfo_in_netloc_and_sanitize_url_must_strip_it():
    from urllib.parse import urlsplit

    assert urlsplit("https://u:p@h/x").netloc == "u:p@h"
    assert _sanitize_url("https://svc:hunter2@litellm.internal:4000/v1/chat?x=1") == (
        "https://litellm.internal:4000/v1/chat?x=1"
    )


@pytest.mark.parametrize(
    "query",
    ["access_key=S", "code=S", "sig=S", "auth=S", "passwd=S", "signature=S", "credential=S"],
)
def test_sanitize_url_drops_the_cache_secret_query_set(query):
    assert "S" not in _sanitize_url(f"https://h/v1?{query}&alt=json")
    assert "alt=json" in _sanitize_url(f"https://h/v1?{query}&alt=json")


def test_sanitize_url_drops_secret_shaped_query_values():
    assert "sk-abc" not in _sanitize_url("https://h/v1?whatever=sk-abc123")


def test_response_root_ambiguous_names_are_dropped():
    out = _sanitize(
        {"token": "T", "choices": [{"message": {"content": "token"}}]}, (), body_root=True
    )
    assert "token" not in out and out["choices"][0]["message"]["content"] == "token"


def test_custom_endpoint_rejects_userinfo():
    with pytest.raises(ValueError):
        CustomEndpoint(completions_url="https://u:p@h/v1/chat/completions", env_var="X")
```

**Step 3 — implement:** `_safe_netloc(parts)` (host[:port], userinfo removed; `urlsplit(...).hostname` / `.port`, port `ValueError` → none) used by `_sanitize_url`; import `_SECRET_QUERY_KEYS`, `_SECRET_QUERY_PARTS`, `_SECRET_VALUE_MARKERS` from `..cache` and union them into `_sanitize_url`'s name test and add a value-marker test (drop the param when its value contains a marker); `RecordingTransport.__call__` sanitizes the response with `body_root=True`; `CustomEndpoint` gains a `field_validator("completions_url")` rejecting a netloc containing `@`. Existing eval tests assert only that `alt=json` survives and `key=` / `api_key=` are removed — confirm they stay green unchanged.

**Commit:** `fix(tape): strip URL userinfo, apply the cache's secret-query set, root-scrub responses (DSE-1517 / CSO F2 F3 F15)`

---

### Task 2: The transport seam — two context types and a strict backstop (F1, F12)

**Files:** modify `src/conclave/transport.py`, `src/conclave/providers.py`; test `tests/test_replay_seam.py` (new).

```python
# transport.py
@dataclass(frozen=True)
class RecordingContext:
    """Live keys, live network; ``post_json`` is wrapped to capture the tape."""

    post_json: PostJson
    offline: ClassVar[bool] = False


@dataclass(frozen=True)
class ReplayContext:
    """No keys, no network. Active only inside ``conclave replay``."""

    post_json: PostJson
    key_sentinel: str = "replay"
    offline: ClassVar[bool] = True


_TRANSPORT: ContextVar[RecordingContext | ReplayContext | None] = ContextVar(
    "conclave_transport", default=None
)
_OFFLINE_STRICT = 0  # process-global refcount; > 0 means "no live call anywhere"


def transport_context() -> RecordingContext | ReplayContext | None:
    return _TRANSPORT.get()


@contextmanager
def recording(context: RecordingContext) -> Iterator[None]: ...  # set/reset only


@contextmanager
def replaying(
    context: ReplayContext, *, strict: bool = True
) -> Iterator[None]: ...  # set/reset the contextvar; if strict, increment/decrement _OFFLINE_STRICT
```

Top of `post_json`: `ctx = _TRANSPORT.get()`; if `ctx is not None` → `return await ctx.post_json(url, headers, json_body, timeout)` (before `_get_client()`); elif `_OFFLINE_STRICT > 0` → `raise TransportError("live call attempted during a strict offline replay", category="unexpected")`. Top of `stream_sse`: if `ctx is not None and ctx.offline` or `_OFFLINE_STRICT > 0` → raise the same way (recording keeps streaming live — but `--record` is buffered-only, so it is never reached under a `RecordingContext`).

`providers._resolve_key`: `ctx = transport.transport_context(); if ctx is not None and ctx.offline: return ctx.key_sentinel` — then the existing loop unchanged. `call_model` / `call_model_stream`: no key-presence branch here (F10 — replay key presence is decided upstream by the council from `result.json`; when a recorded-unkeyed model would be called, the council-level check yields the same "no API key" outcome before `call_model`). Keep the unkeyed message factored so the council can reuse it byte-for-byte.

**Tests:** seam unset → `post_json` reaches the (stubbed) client exactly as before; `ReplayContext` → override called, `_get_client` never invoked; `RecordingContext` → `_resolve_key` reads the **real** env var (F1 guard) and `post_json` wraps the live call; `stream_sse` raises under `ReplayContext`; `_resolve_key` returns the sentinel only under `ReplayContext`; **task isolation**: `asyncio.gather(replay_task, live_task)` created outside `replaying()` — the live task still hits the stub client; **strict backstop**: inside `replaying(strict=True)`, `loop.run_in_executor(None, …)` calling `post_json` raises; with `strict=False` it reaches the client; security suites unchanged and green.

**Commit:** `feat(transport): recording/replay transport contexts with a strict offline backstop (DSE-1517)`

---

### Task 3: `Council` replay awareness and the identity accessor

**Files:** modify `src/conclave/council.py`; test `tests/test_council_replay.py` (new).

- `Council.__init__(…, replay: ReplaySpec | None = None)` where `ReplaySpec{unkeyed_model_ids: frozenset[str]}` is derived by `record.replay_key_presence(result_payload, roster)` from `manifest.providers_skipped` + answers with `failure_category == "unkeyed"` (F10). `_key_present(model_id)` → `model_id not in spec.unkeyed_model_ids` when replaying, else `registry.key_present(model_id)`. Replace the council's `key_present` call sites; `registry` untouched. Replay councils force `cache=False`.
- `Council.run_identity(prompt, mode, *, rounds=None, proposer=None, converge_threshold=None, choices=None) -> dict` factored out of `_cache_key`; `_cache_key` = sha256 of its canonical JSON exactly as before (a fixed-input test proves the key is unchanged).

**Tests:** replay key presence from the spec with the env fully unset; a recorded-unkeyed verdict-extraction candidate reproduces the byte-identical "no API key" error at replay; `_cache_key` unchanged.

**Commit:** `feat(council): replay-aware key presence from the record, run_identity accessor (DSE-1517)`

---

### Task 4: `conclave.record` — canonical form, bundle writer, seal gate, hardened reader

**Files:** create `src/conclave/record.py`; test `tests/test_record_bundle.py`, `tests/test_record_hostile_zip.py` (new).

```python
BUNDLE_SCHEMA_VERSION = "conclave_bundle_v1"
BUNDLE_MEMBERS = ("bundle.json", "result.json", "tape.json", "DIGEST")
VOLATILE_SENTINEL = "<volatile>"
VOLATILE_KEYS = frozenset({"latency_s", "latency_ms"})
VOLATILE_PATHS = (
    ("manifest", "request_id"),
    ("manifest", "total_latency_ms"),
    ("manifest", "pricing_warnings"),
)
MEMBER_CAP = 32 * 1024 * 1024
TOTAL_CAP = 64 * 1024 * 1024
MAX_RATIO = 100
_EXTRA_KEY_PREFIXES = re.compile(r"\b(?:gsk_|sk_|sk-ant-|ghp_|AKIA|ya29\.|hf_|r8_|nvapi-)")
_BARE_BEARER = re.compile(r"\bbearer\s+(?!\[REDACTED\])\S+", re.IGNORECASE)


def canonical_result(payload: dict) -> dict:
    """Deep copy with every volatile VALUE replaced by the sentinel (never deleted)."""


def first_divergence(a: dict, b: dict) -> str | None: ...


def replay_key_presence(result_payload: dict, roster: list[tuple[str, str]]) -> frozenset[str]:
    """Model ids recorded as unkeyed (F10)."""


class SealRefused(RuntimeError): ...


def seal_gate(members: dict[str, bytes]) -> list[str]:
    """Tier A: raise SealRefused. Tier B: return warning strings (bounded, path-only)."""


def seal_bundle(*, path: Path, members: dict[str, bytes]) -> list[str]:
    """Run the gate, write atomically (mkstemp + fsync + os.replace, 0600), return Tier B warnings."""


class BundleError(RuntimeError): ...


def read_bundle(path: Path) -> LoadedBundle:
    """Hardened parse; never partial-load."""
```

**Seal gate, Tier A (refuse):** (1) `redact(text) != text` over the concatenated serialized members; (2) `_EXTRA_KEY_PREFIXES` hit outside `[REDACTED]`; (3) `_BARE_BEARER` hit; (4) any tape record `request` with a `headers` key or any key at any depth for which `tape._is_sensitive_name` is true; (5) any tape URL whose netloc contains `@`. **Tier B (warn, path-named):** bare `authorization` / `api_key` / `api key` with no adjacent secret-shaped token. No flag disables Tier A.

**`read_bundle` hardening (F6):** `set(namelist) == set(BUNDLE_MEMBERS)` and `len(namelist) == 4` (kills extras and duplicates; nothing is ever extracted to disk); reject `flag_bits & 0x1`; reject `file_size > MEMBER_CAP`, sum > `TOTAL_CAP`, `file_size / max(compress_size, 1) > MAX_RATIO`; bounded read `zf.open(n).read(MEMBER_CAP + 1)` refusing on overflow; strict UTF-8 decode; `json.loads` wrapped for `(ValueError, RecursionError)`; schema version checked before anything else is interpreted.

**Write path (F13):** refuse when `path` exists and is a symlink or not a regular file, or `path.parent` is not a directory; mkstemp in `path.parent`, fsync, `os.replace`; fix the helper's double `os.close(fd)` and guard the `unlink` on whether `replace` succeeded (apply the same fix to `cli._write_json_output` — it is the reused helper).

**Tests:** canonical form replaces exactly the volatile set with the sentinel and keeps `cached`; a deleted `latency_s` key diverges; `first_divergence` names nested paths; identical inputs → identical bytes; 0600; round-trip; Tier A refusals (live key value seeded in an answer with that value in the env; `gsk_…` in a response; `bearer xyz`; a `headers` key; `u:p@` URL); Tier B warning for a bare "authorization" in prose; hostile zips (fifth member, duplicate member, DEFLATE ratio 1000:1, encrypted flag, oversized header, non-UTF-8 member, 100k-deep JSON) each → `BundleError`, never a crash.

**Commit:** `feat(record): bundle format, two-tier seal gate, hardened reader (DSE-1517)`

---

### Task 5: `--record` on `conclave ask`

**Files:** modify `src/conclave/cli.py`; test `tests/test_cli_record.py` (new).

Validation: `--stream` → 2; path rules per Task 4 → 2. Council built with `cache=False` (F8). Run inside `transport.recording(RecordingContext(post_json=RecordingTransport(live_post_json, run_identity_hash=…)))`. After stdout rendering and `--json-output`: build `bundle.json` per the format (endpoints per Ruling 1; `--disclose-endpoints` prints `recording endpoint host for <prefix>: <host>` to stderr per custom prefix); if any answer has a non-null `error` print `warning: N answer(s) carry redacted provider error text; it is included in the bundle`; `seal_bundle(...)`; print Tier B warnings; on `SealRefused` print the reason (bounded) and exit 1.

**Tests:** every mode writes a bundle whose `result.json` equals `--json-output`'s bytes; degraded run and incomplete Elite run seal (the error-text warning fires); `--stream --record` → 2 before any call; `--record` + `--json-output` both written; a cache-enabled config still records a live run (`cache=False` forced; no cache file written); custom endpoint → fingerprint entry by default, sanitized URL with `--disclose-endpoints` + notice; without `--record` CLI output byte-identical; seal refusal → 1 with stdout intact.

**Commit:** `feat(cli): conclave ask --record seals a decision record (DSE-1517)`

---

### Task 6: `conclave replay`

**Files:** modify `src/conclave/record.py` (`replay_bundle`), `src/conclave/cli.py`; test `tests/test_replay_command.py` (new).

`replay_bundle(path, *, endpoints: dict[str, str] | None = None, strict: bool = True)`: `read_bundle` → rebuild `ConclaveConfig(models=dict(roster + chain), synthesizer=chain[0], synthesizer_chain=[names], endpoints={…})` where each custom prefix's URL comes from `endpoints` (CLI `--endpoint`) else the operator's loaded config, and **must** satisfy `cache._endpoint_fingerprint(url) == run_identity.endpoint_fingerprints[prefix]` (else `BundleError` naming the prefix); `custom_disclosed` entries supply their own URL; the prompt comes from `result.json`'s `prompt` (F16) → `Council(…, cache=False, replay=ReplaySpec(replay_key_presence(...)))` → assert `council.run_identity(...) == bundle.run_identity` (else `BundleError("run identity mismatch: recorded versions {…} vs running {…}")`) → run the recorded mode with recorded params under `transport.replaying(ReplayContext(post_json=ReplayingTransport(tape, …)), strict=strict)` → `assert_consumed()` → compare canonical forms → `ReplayOutcome(result, divergence)`.

CLI: `replay PATH [--json] [--endpoint PREFIX=URL]… [--no-strict]`; exit 0 / 5 / 2 as in the command table.

**Tests:** offline isolation per the ground rules for each mode's bundle; strict backstop (a patched executor hop raises); edited answer text → 5 naming `answers[0].answer`; extra tape record → 5 (leftover); missing record → 5 (unmatched); edited `run_identity.versions.synthesis_prompt` → 2 with the version message; `conclave_bundle_v0` → 2 with nothing else attempted; custom endpoint: replay fails 2 without `--endpoint`/config, succeeds with the right URL, fails 2 with a URL whose fingerprint differs.

**Commit:** `feat(cli): conclave replay re-derives a decision record offline (DSE-1517)`

---

## Round 4 — verify

### Task 7: Public consensus re-derivation

**Files:** modify `src/conclave/verdict_synthesis.py` (`conflict_score`, `responding_answers`, `recompute_consensus`); test `tests/test_recompute_consensus.py` (new).

```python
def recompute_consensus(
    verdict: CouncilVerdict, responders: Sequence[ModelAnswer]
) -> tuple[float | None, str, list[float | None]]:
    votes = {v.provider: v.position_label for v in verdict.provider_votes}
    sequence = [votes.get(a.name) for a in responders]
    score, label = agreement.consensus(sequence)
    return score, label, [conflict_score(sequence, c.position_labels) for c in verdict.conflicts]
```

**Tests:** reproduces `_assemble_verdict`'s numbers on the existing fixtures; detects an edited score, an edited conflict score, and a flipped vote; N<2 → `None`.

**Commit:** `feat(verdict): public recompute_consensus for third-party verification (DSE-1517)`

---

### Task 8: `conclave verify` (F5)

**Files:** modify `src/conclave/record.py` (`verify_bundle`), `src/conclave/cli.py` (`_VERIFY_FAILED_EXIT_CODE = 5`); test `tests/test_verify_command.py`, `tests/test_record_tamper.py` (new).

`CheckResult{state: Literal["ok", "failed", "not_applicable"], detail: str}` — bounded strings only. `VerifyReport{integrity, consensus, provenance, volatile_fields}`.

1. **Integrity:** every member digest + `DIGEST` per the pinned spec; any mismatch → `failed` naming the member; if it fails, stop (a corrupt bundle is not parsed further).
2. **Consensus:** no verdict → `not_applicable` with `no verdict recorded (reason: <manifest.verdict_absent_reason>)`; else `recompute_consensus(verdict, responding_answers(result))` must equal the verdict's score/label, the hoisted `result.consensus_*` mirrors, and every conflict score → else `failed` naming the first mismatch (expected vs found).
3. **Provenance + cross-member binding (F5a/c):** when a verdict is present, `manifest.verdict_extraction.model_id` and `.prompt_version` non-null and `manifest.consensus_method == "position_cluster_ratio_v1"`; always: `manifest.request_id` present; `bundle.run_identity_hash == sha256(canonical run_identity)`; `tape.run_identity_hash == bundle.run_identity_hash`; `bundle.prompt_fingerprint == run_identity.prompt_fingerprint == cache._digest(result.prompt)`; `bundle.mode == result.mode`; `bundle.mode_params == run_identity.mode_params`; roster names == `manifest.providers_considered` and roster model ids ⊇ `manifest.model_ids`; `manifest.secret_safety == "verified_no_secrets"` reported with the detail `manifest self-scan clean (manifest only — not the answers or the tape)`.

CLI: `verify PATH [--json]` prints `integrity: ok|failed …`, `consensus: ok|failed|n/a …`, `provenance: …`, the ignored volatile set, and the two verbatim "what verify proves" sentences as a footer; exit 0 iff no check `failed`; else 5; unreadable → 2.

**Tamper suite:** score edit → consensus failed; conflict score edit → consensus failed; vote flip → consensus failed; consistent vote+score rewrite without re-sealing → integrity failed (and a doc note that a full re-seal is **not** detected); tape edit / `tape.json` removed / fifth member / truncation → integrity failed or 2; schema version → 2; `secret_safety` un-verified → provenance failed; `bundle.mode` edited → provenance failed; verdict nulled → `consensus: n/a`, exit 0 only if the other checks pass, and the CLI output makes the n/a visible.

**Commit:** `feat(cli): conclave verify — integrity, consensus re-derivation, provenance + cross-member binding (DSE-1517)`

---

### Task 9: Cross-mode and secret-safety coverage (F14)

**Files:** `tests/test_record_secret_safety.py` (new); extend `tests/test_keyleak_audit.py` and `tests/test_secret_safety_matrix.py`.

- Seeded-credential matrix through the REAL `call_model` path with a fake `post_json` echoing request fragments into the response: bearer, `x-api-key`, `x-goog-api-key`, query `key=`, body `api_key`, body `token`, **URL userinfo** — none in any bundle member; variant with a live key value inside an answer → Tier A refusal (exit 1).
- **Adapter header-name binding (F14):** for every adapter, `build_request` with a sentinel key → every header whose value equals the sentinel satisfies `tape._is_sensitive_name(name)`.
- Custom endpoint round-trip: record with `endpoints: {"together": {"completions_url": "https://api.together.xyz/v1/chat/completions?access_key=abc"}}` → fingerprint entry only, `access_key` nowhere; replay with `--endpoint together=<same url>` succeeds; with a different URL → 2.
- `replay` and `verify` read no `*_API_KEY` (patch `os.environ.get` to fail for those names).
- Every mode × {clean, degraded} × record → verify → replay, in that order.

**Commit:** `test(record): seeded-credential matrix, adapter header binding, custom endpoint, cross-mode round trips (DSE-1517)`

---

### Task 10: Documentation

- **README.md** — "Decision records": the three commands; *run `verify` first, then `replay`*; what a bundle contains (prompt, every answer, redacted provider error text); volatile fields are unverified content; endpoints are fingerprinted unless `--disclose-endpoints`; exit codes; the two verbatim sentences below.
- **SECURITY.md** — "Decision records (`.ccr`)": the two-tier seal gate; the pinned `DIGEST` spec with a worked example; the recording/replay contexts and the sentinel (the one credential-path touch); the strict backstop; the hostile-zip rejections; accepted limitations (content; error text; low-entropy fingerprint; no signing — treat a `.ccr` as authoritative only when it reached you over a channel you already trust).
- **docs/PRODUCT_DESIGN_DOCUMENT.md** — §4a "Sealed decision records (v1.5)"; §9 H2 substrate, `evidence_bundle_digest` reserved.
- **CHANGELOG.md** — Added / Changed (evals delegate to `conclave.tape`; `_sanitize_url` now strips userinfo and the wider query set — a hardening of the eval replay path too) / Security / Not changed.
- **DOCUMENTATION_INDEX.md** — link this plan.

The two sentences (verbatim in README, SECURITY.md, and the `verify` footer):

> `conclave verify` proves three things and only three: that the bundle's members hash to the digests the bundle itself declares, that the recorded consensus score and label are the correct arithmetic over the recorded `provider_votes` and responding members, and that the recorded manifest declares the expected consensus method and verdict-extraction provenance.

> It does **not** prove that the recorded votes, answers, or tape came from real provider calls, and — because the bundle is unsigned — it does not prove authorship or detect a forger who rebuilt every digest after editing; a green `verify` means the bundle is internally consistent, not that it is true.

**Commit:** `docs: decision records — README, SECURITY.md threat model + DIGEST spec, PDD, changelog (DSE-1517)`

---

## Ship

1. Push; `gh pr create` (`feat: sealed decision record — record / replay / verify (DSE-1517)`); body: summary, bundle format, threat-model table incl. the "not detected" row, the credential-path touch, exit codes, "not changed", `Closes DSE-1517`.
2. CI green.
3. Round 5: `chief-security-officer` audit of the *built* artifact — the Task 9 matrix plus their own probes; blocking.
4. `release_control.py classify` → `security-specific`; one human receipt at the exact head SHA; `release_control.py merge --method squash`.
5. Linear DSE-1517 → Done.
