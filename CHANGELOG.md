# Changelog

All notable changes to conclave are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Elite Decision Protocol (unreleased).** A quality-first `elite` mode for consequential
  decisions: independent member answers → concurrent council-wide evidence audits → concurrent
  member revisions → the existing synthesis and canonical auditable verdict. It is available
  through `--mode elite`, `Council.elite()`, and `elite_sync()` in source; no version, tag, or
  package publication is claimed here.
- **Fixed three-success phase gate.** Each of Elite's `initial`, `critique`, and `revision`
  phases requires three successful responders. Larger councils may survive partial failures
  while three remain. A failed gate stops later calls and returns an incomplete result with a
  phase-specific reason, attempted artifacts, no synthesis/verdict, and CLI exit code 1.
- **Phase-auditable results.** `EliteResult` preserves initial answers, critiques, and revisions;
  the manifest records a separate redacted receipt with `phase` provenance for every attempted
  member call, aggregates phase usage/latency, and retains the existing secret-safety scan.
  Elite is buffered-only: `--stream` is rejected before provider calls. The quality tradeoff is
  explicit—up to `3N + 2` calls for N members versus the ordinary single-fan-out workflow.

### Fixed

- **`ModelHarnessManifest` now rides on *every* mode's result — a true invariant.**
  The auditable manifest was documented as first-class on every `CouncilResult`, but
  `debate`, `adversarial`, and `vote` built their result directly in `modes.py` and
  returned with `manifest = None` (only `ask`/synthesize attached one). The fix moves
  manifest attachment to the single chokepoint every mode funnels through
  (`Council._cached_run` → new `Council._ensure_manifest`): it fills the manifest from
  the resolved membership + collected answers for `debate`/`adversarial`/`vote`
  (including the zero-members early return and cache hits), is a no-op for the
  synthesize/raw path that already builds its own richer manifest, and stamps
  `secret_safety = verified_no_secrets` when the manifest is provably clean. Regression
  tests (`tests/test_manifest_all_modes.py`) pin the invariant per mode so it cannot
  drift again. The clustering **verdict** scope is unchanged: it still runs on
  `synthesize`/`ask` only and is intentionally not layered onto `adversarial` (which
  already emits a judge verdict) — see PDD §4a.

### Documentation

- Added Elite CLI/library examples and reconciled the PDD, system diagram, README, and index
  around its unreleased status, fixed gate, partial-failure behavior, phased receipts, and
  latency/cost tradeoff.
- Reconciled `README.md`, `SYSTEM_CONTEXT_DIAGRAM.md`, and the PDD so the
  manifest-on-every-result claim is now accurate, and documented the constrained-choice
  **`vote` mode** as **shipped** (CAC-09 / #3) rather than "absorbed by `provider_votes`"
  — the two are complementary (a fixed ballot vs. clustered free-form stances), not the
  same feature. Added the `--mode vote --choices` CLI example and a §4a verdict-scope note.

## [1.1.0] - 2026-06-21

The **auditable council**. Every run now produces a structured, agreement-scored,
fully auditable verdict plus a redacted execution manifest, on top of the existing
synthesize/raw/debate/adversarial modes. The verdict is the product wedge: a
multi-model council answer you can act on, with the agreement number computed by
reproducible arithmetic over the model's clustering — never an LLM-emitted figure.

### Added

- **CouncilResult v2.** New top-level fields, all backward-compatible (default
  `None`/empty): `verdict` (`CouncilVerdict | None`), `consensus_score`,
  `consensus_method`, `consensus_label`, `conflicts`, `provider_votes`,
  `minority_reports`, and a first-class `manifest`. The verdict's values are
  mirrored to these top-level fields; member answers remain on `result.answers`
  (each `ModelAnswer` now carries a stable `answer_id`).
- **Auditable `ModelHarnessManifest`** on every result (not behind a debug flag):
  per-provider execution receipts (latency, usage, redacted error, `schema_valid`),
  considered/called/skipped providers, verdict-extraction provenance (which model +
  prompt version produced the disagreement analysis), and a `secret_safety` stamp
  promoted to `verified_no_secrets` only after the serialized manifest is scanned
  clean. `estimated_cost` is deliberately left `None` (no invented pricing).
- **Deterministic consensus `position_cluster_ratio_v1`** (`agreement.py`):
  `consensus_score` = largest cluster / members with a position; arithmetic over the
  model's clustering, never LLM-emitted, never `difflib`. Deterministic
  `consensus_label` buckets: `none` / `unanimous` / `strong` / `majority` / `split`.
- **Native structured output** across OpenAI / Anthropic / Gemini via a new
  `output_contract` threaded through `call_model` → `adapter.build_request`
  (`response_format` json_schema / `responseSchema` / tool `input_schema`),
  enforcing the lowest-common-denominator verdict/member JSON Schemas at decode
  time, with the prompt-level parse-and-validate fallback retained for providers
  without strict support.
- **Verdict default-on**, with `Council(extract_verdict=False)` to opt out (one
  extra synthesizer call per run). Applied identically on the buffered and streaming
  paths.
- **The verdict-optional rule.** A verdict is absent (with synthesis + member
  answers still returned) for one of three reasons, recorded on
  `manifest.verdict_absent_reason`: `"fewer than 2 responding members"`,
  `"open-ended prompt (no decision/review to adjudicate)"`, or
  `"verdict extraction failed schema validation"`.
- **CLI verdict panel.** A green `VERDICT (<type>)` panel (headline, recommendation,
  a `consensus: <label> (<score>) — heuristic: <method>` line, and optional
  conflicts / minority-report blocks), or a dim `No verdict: <reason>` note when
  absent. `conclave ask ... --json` carries the full `verdict` + `manifest`.
- **New public exports:** `CouncilVerdict`, `CouncilConflict`, `CouncilPosition`,
  `ProviderVote`, `MinorityReport`, `ModelHarnessManifest`, `ProviderExecutionReceipt`,
  `ProviderSkip`, `VerdictExtraction`, `extract_verdict`, `VerdictSynthesisResult`,
  `VerdictExtractionModel`, `verdict_json_schema`, `member_answer_json_schema`,
  `verdict_extraction_json_schema`, `VERDICT_SCHEMA_VERSION`,
  `VERDICT_EXTRACTION_PROMPT_VERSION`.

### Note

- **`vote` (council issue #3):** at 1.1.0 the verdict's `provider_votes` +
  `consensus_label`/`consensus_score` were considered to subsume it. **Superseded post-1.1.0**
  — a real constrained-choice `vote` mode later shipped (CAC-09 / #3; see `[Unreleased]`),
  complementary to `provider_votes`: a fixed ballot vs. clustered free-form stances.

## [1.0.0] - 2026-06-14

First stable release. conclave is feature-complete for its 1.0 scope: a
bring-your-own-keys multi-model council that fans a prompt to N foundation
models concurrently and merges their answers. This release integrates three
release-readiness workstreams — distribution/release engineering, key-leak
hardening + threat model, and synthesizer behavior documentation/versioning —
on top of v0.3.0.

### Added

- **Distribution name.** The package is now published to PyPI as `conclave-cli`
  (`pip install conclave-cli`); the import package, CLI command, and repo all
  stay `conclave`. The bare PyPI name `conclave` is an unrelated project.
- **Release engineering.** OIDC Trusted-Publisher release workflow
  (`.github/workflows/release.yml`) with Sigstore keyless signing and PEP 740
  attestations, inert until a GitHub Release fires and the publisher is
  configured; a hash-pinned dev + runtime lockfile (`requirements-dev.lock`)
  for reproducible installs/CI; and a `RELEASING.md` operator runbook.
- **Supply-chain CI.** A fail-closed `pip-audit` job added to the CI workflow.
- **Threat model.** `SECURITY.md` now carries a BYO-keys threat model and the
  key-handling guarantees consumers can rely on; `.gitleaks.toml` plus a
  dedicated `tests/test_keyleak_audit.py` regression suite guard against
  secret leakage.
- **Versioned synthesis prompt.** The synthesis prompt set is versioned via
  `conclave.prompts.SYNTHESIS_PROMPT_VERSION` and stamped onto every
  `CouncilResult.prompt_version`, so a downstream eval can detect a prompt
  change rather than silently absorb it.

### Changed

- **Key-leak: cause-chain fix.** The originating `httpx` exception is no longer
  attached to `TransportError.__cause__`, closing a path where a verbose
  traceback could surface a key-bearing transport exception.
- **Key-leak: transport-logging guard default-on.** `Council.__init__` now
  installs `conclave.transport.guard_transport_logging()` by default, dropping
  the httpx/httpcore `DEBUG` records that emit the auth header. Callers who
  genuinely need that DEBUG band opt out with
  `Council(..., allow_transport_debug_logging=True)`.
- **Synthesizer: observable degradation.** Synthesizer/judge degradation is
  confirmed (never silent) across synthesize, debate, and the adversarial-judge
  paths: an unkeyed or failed synthesizer surfaces on
  `CouncilResult.synthesis_error` (and `AdversarialResult.verdict_error`,
  mirrored to `synthesis_error`), with no path where synthesis is both absent
  and unexplained.
- **Synthesizer behavior documented.** README gains a "Synthesizer behavior"
  section covering selection precedence (`synthesizer=` arg → config →
  default), observable degradation, and the versioned prompt.

### Scope

- Feature-complete for 1.0: 4 council modes (synthesize / raw / debate /
  adversarial), 9 providers, streaming for synthesize/raw, an optional result
  cache, and debate convergence early-stop.

### Roadmap (post-1.0)

- `vote` mode (council issue #3) — a ranked/tallied decision mode — is
  documented as planned, not shipped.
- A stdio MCP server (council issue #8) is documented as planned; the earlier
  HTTP local-server-mode spike was evaluated and shelved.

## [0.3.0] - 2026-06-08

- Provider-highway refactor: LiteLLM removed in favor of an owned `httpx`
  transport + adapter registry across the (then) 5 providers.
- CI foundation: GitHub Actions matrix, ruff lint/format, coverage floor,
  gitleaks, and branch protection.
- Key-leak fix in `redact()` for custom OpenAI-compatible endpoints; CLI
  exit-code contract and httpx client lifecycle hardening; transport/CLI/logging
  test backfill; first public release with community files.

[1.1.0]: https://github.com/ernestprovo23/conclave/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ernestprovo23/conclave/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/ernestprovo23/conclave/releases/tag/v0.3.0
