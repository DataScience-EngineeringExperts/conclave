# Security Policy

conclave is a **bring-your-own-keys** multi-model council. It calls foundation-model
APIs with the user's own credentials, read from the environment by **variable name
only**. The most security-sensitive surface in the project is therefore key
handling: a weakness that causes a real API key to be stored, logged, serialized,
or echoed back to the user breaks the core trust promise. We treat reports against
that surface as the highest priority.

## Threat model

This section is the honest, current map of what conclave's key handling **does**
and **does not** defend against. It backs the headline BYO-keys claim. The threat
we model is **credential leakage**: a real provider API key escaping the process
into a result object, a log line, a serialized payload, an on-disk cache file, or
terminal output. Accuracy here is the product — we document accepted limitations
rather than overclaim.

### Trust boundary

The boundary conclave defends is **"the user's key value never leaves the in-flight
HTTPS request to the provider, except where the user themselves directs it."**

```
   environment ──(read by NAME, at call time)──▶ adapter builds request
        │                                              │
        │                                       headers carry the key
        │                                              ▼
        └────────────── INSIDE the boundary ──▶ httpx → provider (TLS)
                                                       │
   ════════════ TRUST BOUNDARY ══════════════         │ response / error
                                                       ▼
   redact() scrubs every error/diagnostic string conclave produces
        │
        ▼
   CouncilResult / logs / cache / stdout ◀── OUTSIDE the boundary (must be key-free)
```

Inside the boundary the key value legitimately exists: in the environment, in the
local variable that reads it, and in the request headers handed to httpx. Outside
the boundary — anything conclave returns, logs, caches, serializes, or prints —
must be free of key material. Everything below is about keeping that second set
clean.

### What IS protected

- **Name-only key handling.** Keys are referenced by env var **name** in config
  and code. The value is read from the environment **at call time** in
  `providers._resolve_key`, used only to build the request, and **never assigned
  to any object, cached field, or model**. `registry.key_present` /
  `key_source` report only *whether* a var is set and its *name* — never its value.
- **`redact()` scope.** Every error/diagnostic string conclave surfaces passes
  through `conclave.adapters.base.redact()` before it reaches a result field, a
  log, or stdout. `redact()` scrubs, in order: (1) the live **value** of every env
  var conclave knows a name for — built-in providers **and** custom-endpoint
  `env_var` names declared in config (this catches a BYO key of *any* shape);
  (2) `x-api-key` / `x-goog-api-key` header echoes; (3) `Authorization: Bearer …`
  tokens; (4) standalone provider-shaped key tokens (`sk-…`, `xai-…`, `pplx-…`,
  `AIza…`). `ProviderError` redacts **on construction**; the provider call path
  redacts again at capture (belt-and-suspenders).
- **No key persistence — including the cache.** conclave never writes a key to
  disk. The optional result cache (`conclave.cache`, off by default) stores only
  the already-redacted `CouncilResult` (`model_dump(mode="json")`), and its cache
  key is a SHA-256 over prompt + mode + member/synthesizer **names** + model ids +
  params — **no env var name or value** is read when computing it. So neither a
  cache file, a cache filename, nor the cache key can carry a secret.
- **Streaming path.** Streamed text deltas carry only parsed answer **content**.
  A mid-stream provider error is captured and **redacted** on the final
  `ModelAnswer` exactly like the buffered path; the error path emits no text
  delta, so a key echoed in an error reaches neither a streamed event nor the
  final answer.
- **Partial-failure isolation.** One member failing never aborts a run, and each
  member's error is independently redacted, so a leak in one provider's error
  response cannot smear into another member's result. The defense-in-depth
  catch-alls in `Council.fan_out` and `streaming._drive_member` (which only fire on
  an *unexpected* raise escaping the already-redacting provider call) also run
  their exception text through `redact()`, so the "every surfaced error string is
  scrubbed" invariant holds even on those paths.
- **`repr` / `str` safety.** No config, adapter, or result object stores a key, so
  none can render one in a `repr`/`str` or a traceback frame that references it.
  The transient request `headers` dict does carry the key (it must, to authenticate),
  but it is built inside the adapter and handed straight to the transport — it is
  not retained on any object.
- **Exception cause-chain hardening.** The transport raises its `TransportError`
  with the cause chain dropped, so the surfaced error retains **no** reference to
  the underlying httpx exception. That httpx exception's `.request.headers` holds
  the live auth header; had it survived as `__cause__`/`__context__` it would be one
  cause-chain hop from the surfaced error and would leak the key via
  `traceback.format_exception`, `logging.exception`, or a cause-chain `repr` of a
  transport error. `raise … from None` clears `__cause__` and sets
  `__suppress_context__` (so no standard formatter renders the httpx exception or
  its headers), and the transport additionally nulls `__context__` at a boundary so
  even a direct `err.__context__` attribute walk finds no header-bearing exception.
- **CLI.** `conclave providers` prints key **presence** and the env var **name**,
  never a value; `--json` serializes the same redacted `CouncilResult`.

These guarantees are pinned by the regression suite in
[`tests/test_keyleak_audit.py`](tests/test_keyleak_audit.py) (one test class per
vector below), plus the redaction/cache/streaming tests in `tests/test_providers.py`,
`tests/test_cache.py`, and `tests/test_streaming.py`.

### What redact() does NOT cover — accepted limitations

`redact()` is a defense for the strings **conclave itself** produces. It is not a
universal egress filter, and we do not claim it is. Known gaps, accepted for 1.0:

- **httpx / httpcore DEBUG logging (out of band — guarded by default).** httpx and
  httpcore have their own loggers. At **DEBUG** level httpcore logs full request
  headers, including the `Authorization` / `x-api-key` value, to whatever handler
  the host application configured. This bypasses `redact()` entirely — it never
  sees those records. The guard against it is now **default-on, opt-out**:
  constructing a `Council` automatically calls `conclave.guard_transport_logging()`,
  which installs a filter that drops httpx/httpcore **DEBUG** records (the only
  level that emits header content) while leaving INFO+ diagnostics intact. The
  guard is scoped to the `httpx`/`httpcore` loggers only — it never touches the
  host application's root logger or any other logger. Opt out with
  `Council(…, allow_transport_debug_logging=True)` for the rare case where you
  need that DEBUG band in a process that holds no real keys; you remain responsible
  for it then. Consumers using the provider functions directly **without** a
  `Council` can install the same guard by calling `conclave.guard_transport_logging()`
  once at startup (it is idempotent). Either way, the standing guidance remains:
  do not enable httpx/httpcore DEBUG logging in a process that holds real provider
  keys (e.g. avoid `logging.basicConfig(level=logging.DEBUG)` process-wide).
- **Partial / URL-encoded / transformed key fragments.** `redact()` masks the
  exact env-var value and a fixed set of known key *shapes*. It does **not** catch
  a key that a provider has split, truncated, URL-encoded, base64-wrapped, or
  otherwise transformed before echoing it back, **unless** that transformed form
  still equals the live env-var value (the value-based pass) or matches a known
  shape. A novel provider error that leaks `<first-12-chars>…` of a key, or a
  percent-encoded form, can slip the pattern pass. The value-based pass is the
  primary defense; the shape patterns are best-effort secondary.
- **Anything the user explicitly logs or prints.** If a consumer reads a key from
  the environment themselves, or logs/prints the request headers, the raw
  `os.environ`, or their own constructed Authorization header, that is outside
  conclave's control. conclave only governs the strings it returns and logs.
- **The in-flight request and the provider side.** The key is, by necessity,
  present in the request headers and transmitted to the provider over TLS. What the
  provider does with it (its logs, its breach posture) is outside scope. Memory
  inspection of the running process (a local attacker with debugger access) is also
  out of scope — the env var value is in process memory by design.
- **Dependencies.** Vulnerabilities in httpx, pydantic, typer, pyyaml, or rich
  themselves are upstream; report conclave's *use* of them to us if exploitable,
  but the libraries' own CVEs belong upstream. CI runs gitleaks on every push.

### Key-leak vector map (what a reviewer probes day 1)

| # | Vector | Risk | Status |
|---|--------|------|--------|
| 1 | Cache write path ordering | HIGH if pre-redaction | **Protected** — cache stores only the redacted `CouncilResult`; key never in file/filename/key. Test: V1. |
| 2 | Streaming chunk path | MED | **Protected** — deltas are answer content; mid-stream errors redacted on the final answer, never streamed. Test: V2. |
| 3 | config/transport `__repr__` in tracebacks | MED | **Protected** — no object stores a key; transient headers are not retained. Test: V3. |
| 4 | Provider 400/422 echoing request fragments | MED | **Protected** — error capture runs through `redact()` (and `ProviderError` redacts on construction). Test: V4. |
| 5 | httpx/httpcore DEBUG logging | HIGH (bypasses redact) | **Default-on guard (Council installs it) + opt-out** — `Council.__init__` calls `guard_transport_logging()` automatically; opt out with `allow_transport_debug_logging=True`. Tests: V5, V9. |
| 6 | redact() misses URL-encoded / partial fragments | MED | **Accepted limitation** — documented above; value-based pass is primary, shape patterns best-effort. |
| 7 | Test fixtures with key-shaped strings | LOW | **Protected** — all fixtures use obviously-fake `…FAKE…` patterns; `.gitleaks.toml` allowlists the test tree only. |
| 8 | Partial-failure catch-all error construction (audit-found; not in original map) | LOW | **Protected** — `fan_out` / `_drive_member` catch-alls now `redact()` the raw exception text too. Test: V7. |
| 9 | TransportError cause chain retaining the httpx exception (header-bearing `.request`) | HIGH (leaks via traceback/`logging.exception`) | **Protected** — transport raises `… from None`; surfaced error keeps no `__cause__`/`__context__` ref to the httpx exception, so its auth header cannot leak via traceback/cause-chain repr. Test: V8. |

## Reporting a vulnerability

**Do not open a public GitHub issue, pull request, or discussion for a security
vulnerability.** Public disclosure before a fix is available puts every user's
credentials at risk.

Report privately through **either** channel (GitHub Security Advisories is
preferred because it keeps the report, the fix, and the CVE in one place):

1. **GitHub Security Advisories** — go to the repository's **Security** tab and
   click **"Report a vulnerability"**
   (<https://github.com/ernestprovo23/conclave/security/advisories/new>). This
   opens a private advisory visible only to you and the maintainers.
2. **Email** — `ernest@thedataexperts.us`. Use a clear subject line such as
   `[conclave security]`. If you want to encrypt, say so in a first plaintext
   email and we will arrange a key.

### What to include

A good report lets us reproduce and triage fast:

- The conclave version or commit SHA.
- The surface involved (`conclave ask` / `conclave providers`, or the library
  entry points `Council.ask` / `debate` / `adversarial`), the mode, and the
  relevant flags or config.
- A minimal repro: the `~/.conclave/config.yml` (with key **names**, never
  values), the provider/endpoint, and the input that triggers it. Strip or fake
  any real credentials first.
- The expected vs. actual behavior and the security impact — e.g. "a real key
  appears unredacted in `ModelAnswer.error`", "a key value is written to a log
  line", "`CouncilResult` JSON serialization leaks a credential", "the CLI prints
  a key value".

Reports that demonstrate a **credential leak** — a real key escaping into any
result field, log, serialized payload, or terminal output — are the highest
priority. The key-handling contract is: keys are read from the environment by
name at call time, never stored on objects, never logged, never serialized; and
provider error strings are scrubbed by `redact()` before they reach a result.

## Supported versions

Security fixes are issued for the latest minor series. Older series are not
patched — upgrade to a supported release.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

> Pre-1.0 note: the public surface is still evolving. The supported series will
> advance with each minor release; only the most recent `0.x` minor receives
> security patches.

## Response window

We aim to:

- **Acknowledge** your report within **3 business days**.
- Provide an **initial assessment** (accepted / needs-info / not-a-vuln, with a
  severity estimate) within **7 business days**.
- Ship a fix or a documented mitigation for accepted, validated reports within
  **30 days** of acknowledgement for high/critical severity, and on a best-effort
  basis for lower severities.

These are targets for a small maintainer team, not contractual SLAs. If a report
stalls, a polite nudge to `ernest@thedataexperts.us` is welcome.

## Disclosure & credit

We follow coordinated disclosure. We will work with you on a disclosure timeline,
publish a GitHub Security Advisory (and request a CVE where warranted) once a fix
is available, and credit you in the advisory unless you ask to remain anonymous.

## Scope notes for this repository

- conclave never persists credentials. If you find a path where a key is written
  to disk, a log, a result object, or stdout/stderr, that is in scope.
- Reports about the upstream model providers themselves (OpenAI, Anthropic,
  Google, xAI, Perplexity) or about third-party dependencies (httpx, pydantic,
  typer) are best filed upstream — but tell us too if conclave's *use* of them is
  exploitable.
- conclave is a council aggregator, not a security control. The *content* a model
  returns is not adjudicated for safety; that is out of scope. A leak of the
  user's own credentials is the security boundary we defend.
