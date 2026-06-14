# conclave — Documentation Index

Master index for conclave's documentation. conclave follows the **3-Core Documentation
Rule**: every project maintains exactly three core docs — an overview (README), a system
context diagram, and this index linking everything together. The Product Design Document is
the canonical authority spec on top of those.

- **Repo:** `/Users/ernestprovo/dev/conclave/`
- **Version:** 0.3.0 · **License:** MIT
- **Last updated:** 2026-06-09

---

## Core documentation

| # | Doc | Path | Purpose |
|---|-----|------|---------|
| 1 | **README** (project overview) | [`README.md`](README.md) | What conclave is, install, BYO-keys, CLI + library quickstart, config, test. The fast on-ramp. |
| 2 | **System Context Diagram** | [`SYSTEM_CONTEXT_DIAGRAM.md`](SYSTEM_CONTEXT_DIAGRAM.md) | Mermaid system-context view: user/consumer → CLI/library → Council → provider highway (httpx transport + adapter registry) → 5 providers; config + env-var key inputs; custom OpenAI-compatible endpoints; mcp-warden as dev-time consumer. |
| 3 | **Documentation Index** | [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) | This file. Master map of all docs + source layout. |

## Authority spec

| Doc | Path | Purpose |
|-----|------|---------|
| **Product Design Document** | [`docs/PRODUCT_DESIGN_DOCUMENT.md`](docs/PRODUCT_DESIGN_DOCUMENT.md) | **Canonical** product spec and roadmap. Problem & vision, personas, BYO-keys/key-handling security, council modes (synthesize/raw/debate/adversarial=built; vote=roadmap), provider matrix, architecture, scope + non-goals, roadmap, the mcp-warden dev-time boundary, licensing & positioning, open questions. **When docs disagree, the PDD wins.** |

---

## Source layout

Package root: `src/conclave/` (installed as the `conclave` package; console script
`conclave = conclave.cli:app`).

| Module | Path | Responsibility |
|--------|------|----------------|
| Package API | [`src/conclave/__init__.py`](src/conclave/__init__.py) | Public exports: `Council`, `CouncilResult`, `ModelAnswer`, `TokenUsage`, `DebateRound`, `AdversarialResult`, `ConclaveConfig`, `load_config`, `__version__`. |
| Council | [`src/conclave/council.py`](src/conclave/council.py) | Primary importable entry point. Reusable primitives (`fan_out`, `synthesize_blocks`) + the public mode API (`ask`/`debate`/`adversarial`, async + sync) + streaming (`ask_stream`/`stream_sync`, synthesize/raw). |
| Modes | [`src/conclave/modes.py`](src/conclave/modes.py) | `debate` (multi-round, anonymized peers, drop-out) and `adversarial` (propose → refute → verdict) orchestration, built on `Council.fan_out`/`synthesize_blocks`. |
| Streaming | [`src/conclave/streaming.py`](src/conclave/streaming.py) | `stream_ask` — council-level streaming engine behind `Council.ask_stream`: concurrent member interleaving via an `asyncio.Queue`, optional synthesizer streaming, terminal `done` event with the full `CouncilResult` (synthesize/raw only). |
| Prompts | [`src/conclave/prompts.py`](src/conclave/prompts.py) | Role/template strings for debate + adversarial and the anonymized peer-block builder. |
| Providers | [`src/conclave/providers.py`](src/conclave/providers.py) | Async `call_model` (buffered) + `call_model_stream` (SSE) paths: resolve the adapter, read the key by name at call time, call transport, parse; latency/usage/redacted-error capture; never raises (partial text preserved on mid-stream failure). |
| Transport | [`src/conclave/transport.py`](src/conclave/transport.py) | `post_json` + `stream_sse` — the single async httpx network boundary (buffered POST and `client.stream(...)` SSE) for the whole provider highway. |
| Adapter registry | [`src/conclave/adapters/__init__.py`](src/conclave/adapters/__init__.py) | `resolve_adapter(model_id, config)` — provider registry + **extension seam** (one registration per family; config-only for OpenAI-compatible endpoints). |
| Adapter base | [`src/conclave/adapters/base.py`](src/conclave/adapters/base.py) | `ProviderAdapter` protocol (`build_request`/`parse_response` + `stream_request`/`parse_sse_event`), `SSEDelta`, `ProviderError`, and `redact()` (error-string secret scrubber). |
| OpenAI-compat adapter | [`src/conclave/adapters/openai_compat.py`](src/conclave/adapters/openai_compat.py) | `OpenAICompatAdapter` — openai/xai/perplexity + custom OpenAI-compatible endpoints. |
| Anthropic adapter | [`src/conclave/adapters/anthropic.py`](src/conclave/adapters/anthropic.py) | `AnthropicAdapter` — native `/v1/messages`, system-prompt hoist, required `max_tokens`. |
| Gemini adapter | [`src/conclave/adapters/gemini.py`](src/conclave/adapters/gemini.py) | `GeminiAdapter` — native `generateContent`, OpenAI-role mapping, `usageMetadata`. |
| Registry | [`src/conclave/registry.py`](src/conclave/registry.py) | Friendly-name → model-id defaults; provider → env-var mapping; key **presence** logic (never values). |
| Config | [`src/conclave/config.py`](src/conclave/config.py) | Loads/merges `~/.conclave/config.yml` over defaults; resolves model ids and named/CSV councils; parses the `endpoints:` section (custom OpenAI-compatible providers). |
| Models | [`src/conclave/models.py`](src/conclave/models.py) | Pydantic result contract: `TokenUsage`, `ModelAnswer`, `StreamEvent`, `DebateRound`, `AdversarialResult`, `CouncilResult` (`mode`/`rounds`/`adversarial`/`synthesis_error`/`prompt_version`). Stable downstream surface. |
| CLI | [`src/conclave/cli.py`](src/conclave/cli.py) | `conclave ask` (synthesize/raw/debate/adversarial; `--rounds`/`--proposer`/`--stream`) + `conclave providers`; rich panels, live `--stream` output, and `--json`; never prints key values. |
| Logging | [`src/conclave/logging.py`](src/conclave/logging.py) | Logger factory; stderr; verbosity via `CONCLAVE_LOG_LEVEL` (default `WARNING`). |

## Tests

| File | Path | Covers |
|------|------|--------|
| Council tests | [`tests/test_council.py`](tests/test_council.py) | Fan-out, partial failure, synthesis behavior. |
| Synthesizer tests | [`tests/test_synthesizer.py`](tests/test_synthesizer.py) | Pins the synthesizer/judge contract: default + configurable (arg/config/CLI `--synthesizer`) selection; observable degradation (unkeyed/failed → `synthesis_error`/`verdict_error`, never silent) for synthesize, debate, and the adversarial judge; versioned synthesis prompt (`SYNTHESIS_PROMPT_VERSION` + `result.prompt_version`) with prompt-text + version pins. |
| Modes tests | [`tests/test_modes.py`](tests/test_modes.py) | Debate multi-round flow, mid-round drop-out, peer anonymization; adversarial proposer/critic/verdict, proposal/critic failure paths, no-key judge, sync wrappers. |
| Adapter tests | [`tests/test_adapters.py`](tests/test_adapters.py) | Per-adapter `build_request` + `parse_response` for openai-compat/anthropic/gemini: system-hoist, max_tokens, role mapping, usage parsing, empty/malformed/error-status raises. |
| Provider highway tests | [`tests/test_providers.py`](tests/test_providers.py) | `resolve_adapter` (built-in prefixes, per-provider URLs, custom endpoints, unknown-prefix raise), end-to-end `call_model`, and `redact()` (bearer/`sk-`/env-var-value/`x-api-key` scrubbing; pre-redacted provider errors). |
| Registry/config tests | [`tests/test_registry_config.py`](tests/test_registry_config.py) | Name resolution, key-presence logic, config merge. |
| CLI tests | [`tests/test_cli.py`](tests/test_cli.py) | Typer `CliRunner`: exit-code contract (0 success / 1 zero-usable-answers / 2 usage error), `--json` payload + exit code, human renderers per mode, `providers` table never prints secrets, aclose lifecycle. |
| Transport tests | [`tests/test_transport.py`](tests/test_transport.py) | `post_json` via httpx `MockTransport`: success/error-status/non-JSON fallback, timeout & connect/HTTP errors → `TransportError` (key never leaks), client reuse/pooling, aclose idempotency. |
| Streaming tests | [`tests/test_streaming.py`](tests/test_streaming.py) | Per-adapter SSE via `MockTransport` (openai-compat/anthropic/gemini): incremental chunks + assembled answer == concatenation == buffered result; mid-stream malformed-frame/connection-drop/non-2xx → error set with partial text preserved (never raises); key redaction in stream errors; buffered `ask()` never opens a stream; `Council.ask_stream` interleaving + terminal `done` shape; CLI `--stream` smoke + exit-code contract + debate rejection; `--stream` + cache one-shot replay. |
| Logging tests | [`tests/test_logging.py`](tests/test_logging.py) | `CONCLAVE_LOG_LEVEL` resolution (default `WARNING`, case-insensitive, unknown → `WARNING`), factory contract, one-shot configuration. |
| Fixtures | [`tests/conftest.py`](tests/conftest.py) | Shared fixtures; mocks the httpx transport so the suite needs no network and no API keys. |

Run: `pytest` (config in `pyproject.toml`, `asyncio_mode = "auto"`).

## Project files

| File | Path | Purpose |
|------|------|---------|
| Packaging | [`pyproject.toml`](pyproject.toml) | hatchling build, deps (httpx, pydantic, rich, typer, pyyaml — no LLM SDK), dev extras, console script, pytest config. License: MIT. |
| License | [`LICENSE`](LICENSE) | MIT License. Copyright (c) 2026 Ernest Provo. Matches the `pyproject.toml` license field. |
| Security policy | [`SECURITY.md`](SECURITY.md) | BYO-keys vulnerability reporting policy: how to report, scope, and the key-handling guarantees consumers can rely on. |
| Contributing guide | [`CONTRIBUTING.md`](CONTRIBUTING.md) | Dev setup, the BYO-keys contract for contributors, and the PR checklist (tests, ruff lint/format, coverage). |
| Code of conduct | [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Contributor Covenant v2.1. |

---

## Related projects

| Project | Relationship |
|---------|--------------|
| **mcp-warden** (sibling) | Imports conclave as a **DEV-TIME** dependency (design review, taxonomy labeling). **NOT** a runtime dependency — security findings need determinism; a stochastic council is the wrong tool for runtime adjudication. See PDD §10. |

## Version history

| Date | Change |
|------|--------|
| 2026-06-14 | Documented + tested synthesizer behavior (v1.0 readiness must-do #5): README "Synthesizer behavior" section (selection precedence, observable degradation, versioned prompt); synthesis prompt set now versioned via `conclave.prompts.SYNTHESIS_PROMPT_VERSION`, stamped onto every `CouncilResult.prompt_version`; confirmed (not silent) degradation across synthesize/debate/adversarial-judge paths; new `tests/test_synthesizer.py` (21 tests). No non-synthesis behavior changed. |
| 2026-06-09 | Roadmap features shipped: adversarial proposer resilience (#9), optional result cache (#6), debate convergence early-stop (#4), 4 first-class providers groq/deepseek/mistral/together (#5), streaming for synthesize/raw (#7); tests 121→191. #8 local-server-mode spike evaluated (no-go on HTTP). Doc sync: System Context diagram now shows all 9 providers; PDD §12 resolved questions archived to `docs/archive/pdd-resolved-questions-2026-06-09.md` (PDD back under 500 lines); `config.example.yml` stale "LiteLLM" comment fixed. |
| 2026-06-08 | v0.3.0 version bump; CI foundation (Actions matrix, ruff, coverage floor, gitleaks, branch protection); redact() custom-endpoint key-leak fix (#14); status_error consolidation + conditional temperature (#16/#22); provider-metadata single-source + import-time drift guard + config memoization (#19/#15); CLI exit-code contract + httpx client lifecycle (#17/#20); transport/cli/logging test backfill (#18); public release + community files. |
| 2026-06-08 | PDD §11 repositioned vs. new direct peers (`llm-council-core`, `the-llm-council`); §12 Q1/Q3/Q4/Q5 resolved. Index Tests table updated for the PR #2 split (`test_adapters.py`, `test_providers.py`). |
| 2026-06-07 | v0.3 provider-highway refactor (LiteLLM removed → owned httpx transport + adapter registry); 3-core docs + PDD authored. |

---

## Documentation conventions

- Exactly **3 core docs** (README, System Context Diagram, this index); the PDD is the
  authority spec layered on top.
- Each doc stays **under 500 lines**; overflow is archived to dated/topic files and linked
  from here.
- Prefer **editing existing docs** over creating new ones. Check this index before adding
  any new document.
- The **PDD is canonical**; on any conflict, defer to `docs/PRODUCT_DESIGN_DOCUMENT.md`.
