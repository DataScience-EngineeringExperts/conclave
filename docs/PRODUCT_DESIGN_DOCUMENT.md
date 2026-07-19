# conclave — Product Design Document

> **Status:** v1.1 stable (the BYO-keys multi-model council: synthesize/raw/debate/
> adversarial/vote, 9 providers, owned httpx provider highway, key-leak hardening, streaming,
> cache). **v1.1 — the auditable council — SHIPPED:** every run now yields a structured,
> agreement-scored, execution-traceable **verdict** plus a redacted execution **manifest** (see
> §4a). The quality-first **Elite Decision Protocol is implemented but unreleased**.
> This is the **canonical authority document** for conclave's product scope, design,
> and roadmap. When this document and any other doc disagree, this document wins. Code is the
> source of truth for *current behavior*; this document marks anything not yet in code as
> **Roadmap**.

- **Repo:** `/Users/ernestprovo/dev/conclave/`
- **License:** MIT
- **Author:** Data Science & Engineering Experts, Inc. (DSE)
- **Last updated:** 2026-07-17

---

## 1. Problem & Vision

### Problem
Single foundation models are confidently wrong in ways that are hard to detect from inside
a single model. Different models have different training data, failure modes, and blind
spots. Today, if you want a "second opinion" you either:

- paste the same prompt into 3-5 web UIs by hand and eyeball the differences, or
- adopt a heavyweight multi-agent framework (LangGraph, AutoGen) and write orchestration
  graph code, accept a runtime, and learn its abstractions, or
- route everything through a hosted aggregator that takes a margin on your tokens and sees
  your prompts.

None of these is a lightweight, scriptable, **own-your-keys** primitive for "ask N models
the same thing and reconcile the answers."

### Vision
conclave is a small, sharp tool: **a council of foundation models you can call from one
CLI command or one Python import.** Fan a prompt out to several models concurrently — each
through *your own* API keys, no markup, no middleman — and aggregate the answers. The
v0.1 aggregation is a **synthesizer** that merges raw answers into one consolidated
response; v0.2 adds **council modes** — **debate** (multi-round) and **adversarial**
(propose → refute → verdict) — that turn a flat panel of opinions into structured
deliberation.

**The v1.1 wedge — the historically named "auditable council."** A synthesis paragraph is not enough to *act* on.
v1.1 makes the product identity precise: **a multi-model council verdict you can inspect —
structured, scored for agreement, and execution-traceable.** Every run yields a `CouncilVerdict`
exposing agreement, disagreement (`conflicts`), minority views (`minority_reports`), and
per-provider votes (`provider_votes`); a deterministic `consensus_score` (arithmetic over the
model's clustering, *never* an LLM-emitted number); and a redacted `ModelHarnessManifest` (how
the run executed + which model produced the analysis) riding on **every** released-mode result;
source-only Elite results carry it too (one chokepoint, §4a). A constrained-choice **`vote`
mode** also **shipped** (CAC-09 / #3,
`--mode vote --choices "A,B,C"` → plurality winner/split) — distinct from `provider_votes`.

The next product-quality step is **Elite** (implemented, unreleased): independent answers → council-wide answer/claim audits → member revisions → the existing synthesis and execution-traceable verdict. Elite requires three successful responders at every member phase and intentionally spends more calls and time to improve consequential decisions.

conclave's first real use was an **adversarial design review**: a council of Grok, Gemini,
Perplexity, and Claude critiquing a security-tool strategy and catching flaws a single
model missed. That origin is why the adversarial and debate modes are first-class — they
are now built, not a bolt-on. The product stays lightweight: a **library-first primitive
with structured, execution-traceable results**, not an agent framework and not a general AI SDK — it
builds only what deepens the council wedge (the boundary vs. LiteLLM/Vercel/LangChain/
Helicone is in §11).

---

## 2. Target Users & Personas

| Persona | Who | What they want from conclave |
|---------|-----|------------------------------|
| **The skeptical engineer** | Senior dev / architect making a consequential technical call | A fast second/third opinion across models, with raw per-model answers visible so they can judge disagreement themselves. Uses the CLI ad hoc. |
| **The library integrator** | Developer building a tool that needs multi-model input at *design/eval time* | `from conclave import Council`, structured `CouncilResult` (latency, token usage, per-model errors), partial-failure resilience. The primary downstream example is **mcp-warden** (see §10). |
| **The researcher / evaluator** | Someone comparing model behavior on a prompt set | Deterministic structure around answers, JSON output (`--json`) for downstream analysis, per-model latency and token accounting. |
| **The cost-conscious power user** | Heavy LLM user who already pays each provider directly | BYO-keys with **no markup** and **no third party seeing the prompt**. conclave is a thin local orchestrator over the user's own accounts. |

Non-personas (*not* who we build for): teams wanting a hosted multi-agent SaaS, or anyone
needing a deterministic runtime adjudicator (Non-Goals §8, mcp-warden boundary §10).

---

## 3. BYO-keys Model & Key-Handling Security

conclave is **bring-your-own-keys** by design. This is both a positioning choice (no
markup, no middleman) and a security property.

**Key-handling invariants (enforced in code today):**

1. **Keys are referenced by env-var NAME only, never by value.** The provider registry
   (`registry.py`) maps each provider prefix to the env var(s) that satisfy it
   (e.g. `xai → ["XAI_API_KEY"]`, `gemini → ["GEMINI_API_KEY", "GOOGLE_API_KEY"]`). The
   functions `key_present()` and `key_source()` answer *"is a key set?"* and *"which
   variable name holds it?"* — they **never read, return, or log the value.**
2. **conclave never stores keys.** Config (`~/.conclave/config.yml`) references providers
   by friendly name and model id only. There is no field in `ConclaveConfig` that can hold
   a secret. The example config in `config.py` is keys-free by construction.
3. **The key value is read by name, at call time, and is transient in-process**
   (`providers.py`). `call_model` reads the env var *by name*, hands the value to the adapter
   to build the auth header, and the transport sends it. The value is **never stored on any
   object** (config, registry, `ModelAnswer`, `CouncilResult`, or `ModelHarnessManifest`),
   **never logged, never serialized, and scrubbed from error strings** via `redact()`
   (`adapters/base.py`). Honest framing: it *is* read in-process to authenticate, but its
   lifetime is a single request and it leaves no trace on any persisted/returned object.
4. **Secrets never reach serialized output.** `CouncilResult.model_dump()` (`--json`) carries
   prompts, answers, model ids, latency, usage, errors, the verdict, and the manifest — no
   key material. The v1.1 manifest goes further: `secret_safety` is promoted to
   `verified_no_secrets` only after `scan_for_secret_material()` proves the serialized
   manifest clean (§4a). The `providers` CLI shows a check/cross and the env-var *name* only.
5. **Missing keys degrade gracefully, they don't crash.** A requested member whose key is
   absent is skipped with a warning and recorded in `CouncilResult.skipped`. Unknown
   providers (no static env-var mapping) are *not* pre-emptively skipped — the live call is
   attempted and any auth error is captured as a `ModelAnswer.error`.

**Residual considerations:** a provider error could in principle echo a key fragment. Since
v0.3 every provider/transport error is passed through `redact()` before it reaches
`ModelAnswer.error`; the residual risk is limited to a secret in a shape `redact()` does not
recognize. (Was §9 hardening item 7 — landed.)

---

## 4. Council Modes & Consensus Algorithms

A **council mode** is the algorithm that turns N independent model calls into one useful
output. The v1.1 verdict layer (§4a) sits on top of whichever mode produced the answers.

| Mode | Status | What it does |
|------|--------|--------------|
| **synthesize** | **BUILT (v0.1)** | Fan out concurrently → collect each raw answer → a **synthesizer model** merges them into one consolidated answer, reconciling agreement, adjudicating disagreement, and flagging clearly-wrong answers. The synthesizer is instructed to rely only on the provided answers and not invent a model's position. |
| **raw** | **BUILT (v0.1)** | Fan out and return every member's raw answer with no synthesis. Not a deliberation mode — it is "synthesize off." Exposed as `--mode raw` / `ask(..., synthesize=False)`. |
| **debate** | **BUILT (v0.2)** | N rounds (`--rounds`, default 2). Round 1 is an independent fan-out; rounds 2..N show each member its peers' **anonymized** prior-round answers (`Model A/B/C`) and ask it to revise or defend. A member that errors in a round drops out of later rounds; the debate continues with survivors. The synthesizer consolidates the final round. Exposed as `--mode debate` / `Council.debate()` / `debate_sync()`. |
| **adversarial** | **BUILT (v0.2)** | Structured propose → refute → verdict. A `--proposer` (default: first member) answers; the remaining members are CRITICS explicitly prompted to refute it; the synthesizer acts as JUDGE, weighing proposal vs. critiques and issuing a verdict + strengthened answer. This is the mode conclave's origin story (the security design review) exercised by hand. Exposed as `--mode adversarial` / `Council.adversarial()` / `adversarial_sync()`. |
| **vote** | **BUILT (v1.1, CAC-09 / #3)** | Constrained-choice ballot: each member sees a fixed labelled option set (`A, B, C, …`) and answers with one letter; responses are tallied to a plurality `winner` (or `split` on a tie) on `result.vote` (`VoteResult`). Exposed as `--mode vote --choices "A,B,C"` / `Council.vote()` / `vote_sync()`. Distinct from the verdict's `provider_votes`, which cluster free-form stances with evidence (§4a); `vote` tallies a fixed ballot. |
| **elite** | **IMPLEMENTED, UNRELEASED** | Quality-first decision protocol: independent fan-out → concurrent council-wide claim audits → concurrent member revisions → existing synthesis and canonical verdict. Every member phase has a fixed three-success gate. Exposed in source as `--mode elite` / `Council.elite()` / `elite_sync()`; buffered only, no streaming. |

**Mode algorithms (as built).** The step-by-step "as built" prose for synthesize / raw /
debate / adversarial / vote (fan-out + partial-results, peer anonymization + drop-out, proposer →
critic → judge, ballot tally) is landed history and lives in
[`docs/archive/pdd-v0.x-modes-detail.md`](archive/pdd-v0.x-modes-detail.md). In brief: every mode
fans out concurrently, captures each call as a `ModelAnswer` (answer **or** redacted error —
`call_model` never raises), and survives partial failure; the deliberation modes extend
`CouncilResult` (`mode`, `rounds`, `adversarial`, `vote`, `elite`) backward-compatibly so v0.1
`answers`/`synthesis` consumers keep working. The mode *text* output is a generative, inherently
stochastic reconciliation (load-bearing for the mcp-warden boundary, §10); the v1.1 verdict (§4a)
adds a *deterministic* agreement number on top, by arithmetic over a clustering.

**Elite gate and partial-failure contract.** `required_responders` is fixed at 3. Councils may start larger, but only members successful in a phase advance. One failure in a four-member council is tolerated; fewer than three successes in `initial`, `critique`, or `revision` stops the protocol immediately.
The result is incomplete with a phase-specific `failure_reason`, no later calls, synthesis, or verdict; attempted phase artifacts and redacted failures remain serialized and the CLI exits 1. A completed run mirrors successful revisions to `answers`, but completion only means the three member phases passed.
Decision readiness is separate: `ready`, `not_ready`, or `indeterminate`, plus machine-readable `readiness_reasons`; synthesis or required-adjudication failure is not ready, and disabled or inapplicable adjudication is indeterminate. The CLI exits 1 unless readiness is `ready`. Elite normally uses up to `3N + 2` calls, `3N + 3` if verdict repair runs, or `3N + 1` with extraction disabled.
Cache identity is version-aware and secret-free: protocol/prompt/schema/cache versions, resolved model roster, generation and mode settings, extraction behavior, sanitized endpoint routing, and optional source-bundle digest all invalidate incompatible entries; old envelopes miss safely.

---

## 4a. The Execution-Traceable Verdict (v1.1)

The verdict layer turns a council run's answers into a structured, agreement-scored,
execution-traceable adjudication — on top of any mode, default-on, never breaking the v0.1 surface
(every new field defaults to `None`/empty).

### CouncilResult v2 surface
`CouncilResult` gains these top-level fields, all backward-compatible:

| Field | Type | Meaning |
|-------|------|---------|
| `verdict` | `CouncilVerdict \| None` | The canonical adjudication object (`None` when no verdict applies). The fields below are convenience **mirrors** of the verdict; the verdict object is canonical. |
| `consensus_score` | `float \| None` | Position-cluster ratio in `[0.0, 1.0]`. |
| `consensus_method` | `str \| None` | The method literal `"position_cluster_ratio_v1"`. |
| `consensus_label` | `str \| None` | One of `unanimous \| strong \| majority \| split \| none`. |
| `conflicts` | `list[CouncilConflict]` | Disagreements, each with a per-conflict ratio. |
| `provider_votes` | `list[ProviderVote]` | Who took which position (absorbs GH #3 "who voted for what"). |
| `minority_reports` | `list[MinorityReport]` | Dissenting views worth surfacing (for adversarial = unrefuted critic points). |
| `manifest` | `ModelHarnessManifest \| None` | First-class execution + provenance receipt on every real run. |

Member answers stay exposed as `result.answers`; each `ModelAnswer` carries a stable
`answer_id`. The verdict types (public-exported Pydantic v2 models in `verdict.py`):
`CouncilVerdict{verdict_type ∈ decision|review|synthesis, headline, recommendation,
consensus_score/method/label, positions, conflicts, provider_votes, minority_reports,
caveats, dissent_summary, schema_version}`; `CouncilPosition{label, summary, providers,
evidence_answer_ids}`; `CouncilConflict{topic, position_labels, summary, consensus_score}`;
`ProviderVote{provider, position_label, confidence}`; `MinorityReport{providers, claim,
evidence_answer_ids, why_it_matters}`.

**Answer provenance is the current product, not external evidence.** Every clustered stance
cites `evidence_answer_ids` (the member `answer_id`s backing it) and every conflict names the positions in tension — a
conflict that just says "models disagreed about cost" without pointing at answers is a
*failure*. `ProviderVote.confidence` is recorded but **never used in arithmetic**.

### Deterministic consensus — the traceability fix
The consensus number is **arithmetic over the model's clustering, never LLM-emitted**
(`agreement.py`, method `position_cluster_ratio_v1`).

- `consensus_score(positions)` = `|largest cluster| / |members with a non-null position|`.
  Returns `None` when fewer than 2 members expressed a position (N<2 → agreement undefined).
  A `None` position is excluded from numerator *and* denominator; `"conditional"`/`"it
  depends"` is a valid cluster and counts.
- `consensus_label(score)` is a deterministic bucket:

| Label | Range |
|-------|-------|
| `none` | score is `None` |
| `unanimous` | score == 1.0 (N ≥ 2) |
| `strong` | 0.75 ≤ score < 1.0 |
| `majority` | 0.5 < score < 0.75 |
| `split` | score ≤ 0.5 (no majority); a 1-of-2 tie is `0.5` = `split`, never "50% consensus" |

**Why it is execution-traceable, not ground truth.** The extraction schema carries *no* consensus field —
`verdict_extraction_json_schema()` strips it and `VerdictExtractionModel` ignores extra keys,
so a model that smuggles a number in is dropped by the validator. The module deliberately
does **not** import `difflib`: text-similarity is the debate `convergence_score` (a
*forbidden* consensus measure), never conflated with agreement. The single LLM-assisted step
is the **semantic clustering** of stances; the number is reproducible arithmetic over it,
each cluster cites its `evidence_answer_ids`, and the manifest records which model + prompt
version did the clustering — so the score is reproducible and traceable. Answer IDs point to
council outputs, not external sources; source grounding is Roadmap (§9).

### Verdict extraction + native structured output
`extract_verdict(prompt, member_answers, *, synthesizer_name, synthesizer_model_id,
config=None, temperature=0.7, timeout=120.0, protocol_version=None) ->
VerdictSynthesisResult(verdict, extraction, verdict_absent_reason, attempt_receipts)`
(`verdict_synthesis.py`) makes one initial extraction call asking the synthesizer model to
*cluster* stances (not to re-answer, not to emit a number), validates, and makes at most one
repair call before falling back gracefully — never raises.

It builds an `OutputContract(schema=verdict_extraction_json_schema(),
schema_name="VerdictExtraction", strict=True)` (CAC-06-PLUMB threaded `output_contract`
through `call_model` → `adapter.build_request`), passed to both the initial call and the
repair retry. Capable providers **enforce the schema at decode time** — OpenAI
`response_format` json_schema, Gemini `responseSchema`, Anthropic tool `input_schema`. The
three public schemas (`verdict_json_schema`/`member_answer_json_schema`/
`verdict_extraction_json_schema`) are a deliberate **lowest-common-denominator** shape
(shallow nesting ≤3, enums not `oneOf`, no `$ref`, `additionalProperties:false`, optionality
by omission) so one schema spans all three native surfaces. A **prompt-level fallback**
(schema in messages → JSON parsed → Pydantic-validated → repair-once) is retained for
providers without strict support; the native contract is *additive*, failure behavior
unchanged.

### The verdict-optional rule
A verdict is not always meaningful. In three cases `result.verdict is None` while `synthesis`
+ member answers stay populated, `consensus_score = None`, and the exact reason is recorded on
`result.manifest.verdict_absent_reason` (provenance — extractor model id + prompt version — is
recorded on **every** return path, including these three):

- `"fewer than 2 responding members"` (N<2 → no LLM call at all).
- `"open-ended prompt (no decision/review to adjudicate)"` (creative/open-ended generation).
- `"verdict extraction failed schema validation"` (extraction failed after one repair).

### Default-on, with an opt-out
Verdict extraction is **default-on** (`Council(..., extract_verdict=True)`) — it is the
council's product. Opt out with `Council(extract_verdict=False)` (then `result.verdict` stays
`None` and the manifest's verdict-provenance slots stay `None`). It is a constructor flag
(`self.extract_verdict_enabled`), no per-call override. Buffered (`ask`) and streaming
(`stream_ask`) both run the same `_apply_verdict` helper *after* the manifest exists, so the
verdict appears identically in the buffered result and the streaming `done` event and the
`secret_safety` stamp is re-run over the final content. **Cost:** default-on adds one initial
synthesizer call and at most one schema-repair retry; the opt-out exists for cost-sensitive callers.

**Verdict scope across modes (deliberate).** Posture: *manifest everywhere, clustering verdict only where unambiguously additive.* The manifest rides on all five released modes and on source-only Elite. The clustering verdict runs on `synthesize`/`ask` and on **completed Elite runs after synthesis of the revised answers**; it is deferred on `debate`/`vote` and intentionally not layered onto `adversarial`, which already emits a judge verdict. An incomplete Elite run has no synthesis or verdict.

### ModelHarnessManifest — first-class, secret-free
The `ModelHarnessManifest` (`manifest.py`) rides on every `CouncilResult` — *not* behind a debug
flag, and now a **true invariant** enforced at one chokepoint: all five released modes plus source-only Elite funnel through
`Council._cached_run` → `_ensure_manifest`, which attaches the manifest on every returned result
— including `debate`/`adversarial`/`vote` (built in `modes.py`), the zero-members early return,
and cache hits (synthesize/raw builds its own richer one earlier). Pinned by
`tests/test_manifest_all_modes.py`. It records `request_id`, `conclave_version`, `mode`,
`providers_considered/called/skipped`
(each skip a `ProviderSkip{name, reason}`), `model_ids`, `generation_settings`, `receipts` (each
a `ProviderExecutionReceipt{phase, attempt, outcome, name, provider, model_id,
generation_settings, latency_ms, usage, error_category, schema_valid, versions}`),
`total_latency_ms`, `total_usage`, `schema_valid`,
`redacted_errors`, and verdict-provenance slots (`verdict_extraction: VerdictExtraction{model_id,
prompt_version}` — the execution-trace hook — plus `verdict_type`, `consensus_method`,
`verdict_absent_reason`). Two deliberate honesty choices:

For buffered Elite, every attempted call becomes a receipt: `initial`, `critique`, `revision`, `synthesis`, `verdict_extraction`, and `verdict_repair` when repair is attempted. Receipts carry phase, attempt/outcome, provider/model identity, latency, available usage/cost, bounded error category, and prompt/schema/protocol versions; totals are recomputed from this complete ledger. Incomplete runs retain only calls actually attempted.

- **No invented pricing.** Unknown per-call or aggregate `estimated_cost` stays `None`; a total is
  computed only when every actual call has trustworthy priced data. Usage is recorded when reported.
- **Proven secret-safety.** `secret_safety` defaults to `unverified`, promoted to
  `verified_no_secrets` **only** after `scan_for_secret_material()` proves the serialized manifest
  free of forbidden substrings (`sk-`, `bearer`, `authorization`, `api_key`, `x-api-key`). Key
  *values* never appear; errors are redacted upstream and re-redacted on construction.

---

## 5. Provider Support Matrix

Friendly names, default model ids, and the env var(s) that satisfy each. Defaults
live in `registry.DEFAULT_MODELS` / `registry.PROVIDER_ENV_VARS` and are overridable via
`~/.conclave/config.yml`.

| Provider | Friendly name | Default model id | Env var(s) (first present wins) | Status |
|----------|---------------|------------------|---------------------------------|--------|
| xAI | `grok` | `xai/grok-4.3` | `XAI_API_KEY` | BUILT |
| Google | `gemini` | `gemini/gemini-2.5-pro` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | BUILT |
| Anthropic | `claude` | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | BUILT |
| Perplexity | `perplexity` | `perplexity/sonar-pro` | `PERPLEXITY_API_KEY` | BUILT |
| OpenAI | `openai` | `openai/gpt-4.1` | `OPENAI_API_KEY` | BUILT |
| Groq | `groq` | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` | BUILT |
| DeepSeek | `deepseek` | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` | BUILT |
| Mistral | `mistral` | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` | BUILT |
| Together | `together` | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` | BUILT |
| *(any provider known to an adapter)* | *raw id as name* | *passed through verbatim* | *adapter's provider env var* | SUPPORTED (untyped) |
| *(any OpenAI-compatible endpoint)* | *config `endpoints:` entry* | *your model id* | *the endpoint's `api_key_env`* | SUPPORTED (config-only) |

All nine first-class providers are **direct vendor key → direct vendor endpoint** (no
aggregator/router, per §11). Groq/DeepSeek/Mistral/Together (issue #5) are OpenAI-compatible,
served by `OpenAICompatAdapter`; aggregators/routers (e.g. OpenRouter) are deliberately *not*
promoted — they stay config-only via `endpoints:`, keeping no-middleman intact. **Default
synthesizer:** `claude`. **Default council:** all nine. Unknown names pass through verbatim
(adapter-recognized prefix, "attempt and catch"); a wholly new OpenAI-compatible vendor needs
no code — a `config.yml` `endpoints:` entry (base URL + `api_key_env`) makes it a first-class
member (§6). Further first-class defaults remain Roadmap, §9.

---

## 6. Architecture

conclave is a thin, layered orchestrator over its **own provider highway** — an httpx
transport behind a per-provider adapter registry, with **no LLM-SDK dependency**. Each
module has one job; the data models are the stable contract between layers and downstream
consumers. The end-to-end flow — `CLI/Library → Council → call_model → adapters → transport
→ providers`, plus `_apply_verdict → extract_verdict → agreement → CouncilVerdict`, with the
`ModelHarnessManifest` riding on the result — is drawn in `SYSTEM_CONTEXT_DIAGRAM.md`.

**Module responsibilities (ground truth):**

| Module | Responsibility |
|--------|----------------|
| `council.py` | `Council` — primary importable entry point. Resolves names, partitions members, and exposes two reusable primitives: `fan_out` (the single concurrent + partial-failure call loop) and `synthesize_blocks` (the single synthesizer/judge call path). Hosts the async/sync APIs for five released modes plus unreleased `elite`/`elite_sync`. Every source mode funnels through `_cached_run`; completed Elite runs synthesize revisions and then use `_apply_verdict`. |
| `verdict.py` | Public verdict/member Pydantic types (`CouncilVerdict`, `CouncilPosition`, `CouncilConflict`, `ProviderVote`, `MinorityReport`) + the LCD JSON Schemas (`verdict_json_schema`/`member_answer_json_schema`/`verdict_extraction_json_schema`) usable across all three native structured-output surfaces; `VERDICT_SCHEMA_VERSION`. |
| `agreement.py` | Deterministic consensus: `consensus_score` (`position_cluster_ratio_v1` — largest cluster / positioned members; `None` for N<2) + `consensus_label` buckets. Pure arithmetic, no `difflib`, never LLM-emitted. |
| `verdict_synthesis.py` | `extract_verdict` engine: one initial extraction call and at most one repair (clusters stances, never emits a number), native `output_contract` enforcement + prompt-level fallback, validate → repair-once → graceful `verdict=None`; the three verdict-absent reasons; provenance on every return path. |
| `manifest.py` | `ModelHarnessManifest` (first-class on every result), phased `ProviderExecutionReceipt`/`ProviderSkip`/`VerdictExtraction`, and `scan_for_secret_material()` → `secret_safety` stamp. No key values; unknown `estimated_cost` remains `None`. |
| `modes.py` | Deliberation orchestration: `run_debate`, `run_adversarial`, `run_vote`, and unreleased `run_elite` (three gated member phases). Built on `Council.fan_out` + `synthesize_blocks`. |
| `prompts.py` | Role/template strings for debate, adversarial, vote, and Elite claim-audit/revision prompts. Elite panel text uses stable Model A/B/C aliases and answer ids, never provider identities. |
| `providers.py` | `call_model` (+ `call_model_stream`) — the single async call path: resolve adapter, read key *by name at call time*, call adapter+transport (with an optional `output_contract`), parse, capture latency/usage/redacted-error into a `ModelAnswer`; never raises for provider-side failures. |
| `transport.py` | The single async network boundary: `post_json` (buffered) + `stream_sse` (issue #7) — the only two httpx call sites in the highway. |
| `streaming.py` | Streaming engine (issue #7): `stream_ask` interleaves members via an `asyncio.Queue`, optionally streams the synthesizer, ends with a `done` result. Synthesize/raw only; Elite explicitly rejects streaming. |
| `adapters/__init__.py` | `resolve_adapter(model_id, config)` — the provider registry + **extension seam**: one registration per family; config-only for OpenAI-compatible endpoints. |
| `adapters/base.py` | `ProviderAdapter` protocol, `OutputContract` (native-structured-output request), `ProviderError`, and `redact()` (error-string secret scrubber). |
| `adapters/openai_compat.py` | `OpenAICompatAdapter` — openai/xai/perplexity/groq/deepseek/mistral/together + custom endpoints; per-provider completions URL (Perplexity no `/v1`; Groq under `/openai/v1`); `response_format` json_schema when an `output_contract` is set. |
| `adapters/anthropic.py` | `AnthropicAdapter` — native `/v1/messages` (system-hoist, `max_tokens` required); `input_schema` tool for an `output_contract`. |
| `adapters/gemini.py` | `GeminiAdapter` — native `generateContent` (role-map, `systemInstruction` hoist, `usageMetadata`); `responseSchema` for an `output_contract`. |
| `registry.py` | Single source of truth for name→model-id defaults + provider→env-var mapping. Key *presence* only — never values. |
| `config.py` | Loads/merges `~/.conclave/config.yml` over defaults; resolves model ids + named/CSV councils; parses `endpoints:`. Keys-free by construction. |
| `models.py` | Pydantic contract: `TokenUsage`, `ModelAnswer`, `EliteResult` (protocol state + three phase-artifact lists), and backward-compatible `CouncilResult` v2 with verdict/consensus/manifest fields. |
| `cli.py` | `conclave ask` and `conclave providers`. Rich panels for humans (incl. the green `VERDICT (<type>)` panel + consensus/conflicts/minority blocks, or a dim `No verdict: <reason>` note when absent), `--json` for machines (carries verdict + manifest). Never prints key values. |
| `logging.py` | One logger factory, stderr, verbosity via `CONCLAVE_LOG_LEVEL` (default `WARNING`). |

**Key design properties:** library-first (the CLI is a thin shell over the same `Council`);
partial-failure resilience is structural (failures become `ModelAnswer.error` data, never
run-aborting exceptions); structured + stable results (`models.py` field names are a
deliberate downstream contract, e.g. for mcp-warden). **Extension is cheap:** a new provider
family is one registration in `adapters/__init__.py`; a new OpenAI-compatible endpoint is
config-only (`endpoints:` entry — base URL + key-env-var *name*), served by
`OpenAICompatAdapter` with no code change. The key value is read by name at call time and
never stored, logged, or serialized (§3).

**Stack:** Python 3.11+, `httpx` (the only network dependency), `asyncio`, Pydantic v2,
Typer + Rich, PyYAML. **No LLM-SDK dependency.** hatchling build; console script
`conclave = conclave.cli:app`.

---

## 7. Scope

Condensed history (v0.x mode-detail archived per §4, per-release changelog in `CHANGELOG.md`, verdict layer in §4a):

- **v0.1:** `synthesize` + `raw` modes; first-class providers (5, now 9) + adapter
  pass-through; BYO-keys by env-var name with graceful skip; concurrent fan-out; structured
  `CouncilResult`; CLI (`ask`/`providers`, `--json`); config; sync + async API; mocked suite.
- **v0.2:** `debate` (multi-round, anonymized peers, drop-out, `.rounds`) and `adversarial`
  (proposer → critics → judge, `.adversarial`); backward-compatible `CouncilResult` extension.
- **v0.3:** **LiteLLM removed** → owned `httpx` provider highway + adapter registry (§6, the
  only network dependency; 3 adapters cover 9 providers); custom OpenAI-compatible `endpoints:`;
  key-leak hardening via `redact()`; `call_model` signature + never-raises contract unchanged.
- **v1.0 (stable):** dist name `conclave-cli`; OIDC Trusted-Publishing + Sigstore + PEP 740;
  key-leak threat model (`SECURITY.md`, transport-logging guard default-on); versioned synthesis
  prompt (`SYNTHESIS_PROMPT_VERSION`); streaming (synthesize/raw) + result cache + debate early-stop.
- **v1.1 (the auditable council):** `CouncilResult` v2 — `verdict` + `consensus_*` +
  `conflicts`/`provider_votes`/`minority_reports` + first-class `manifest`; deterministic
  `position_cluster_ratio_v1` consensus; native + fallback structured output across
  OpenAI/Anthropic/Gemini; the verdict-optional rule; verdict default-on with
  `Council(extract_verdict=False)` opt-out; the auditable `manifest` made a true
  every-mode invariant (one chokepoint). A constrained-choice **`vote` mode** shipped
  (CAC-09 / #3) — `--mode vote --choices` — distinct from the verdict's `provider_votes`.
  Full detail in §4a.

---

## 8. Non-Goals (v0.1, and some permanent)

- **Not a runtime adjudicator.** conclave is stochastic; it must not be a deterministic
  decision gate (§10). **Permanent** for synthesize/debate/adversarial — *and* for the v1.1
  verdict: the verdict's *clustering* is LLM-assisted (stochastic), so even the deterministic
  `consensus_score` is not a reproducible security gate. The calculation is traceable, not authoritative.
- **Not an agent framework.** No tool-calling graphs, stateful agents, or orchestration DSL — we compete by being *small*. (Permanent.)
- **Not a key manager / secrets vault.** conclave reads env vars; it does not provision, rotate, store, or proxy keys. (Permanent.)
- **No hosted/proxied token path.** No conclave-operated endpoint that sees prompts or takes a margin — BYO-keys, direct-to-provider, always. (Permanent.)
- **No streaming for debate/adversarial/vote/elite** (synthesize/raw streaming landed in v0.3, #7).
- **No server mode** (possible Roadmap, §9; local HTTP spike #8 closed no-go — §9 item 6).

---

## 9. Roadmap

`adversarial`/`debate` shipped in v0.2; streaming/cache/convergence in v1.0; the **auditable
council shipped in v1.1** (§4a — the wedge).

The revised thesis is a **source-grounded, execution-traceable decision record with
empirically proven quality**. Current answer IDs identify model outputs, not external evidence;
source-auditable language is therefore too broad until source grounding ships. Elite remains
implemented but unreleased in source.

H1 also includes an opt-in live runner that is **paid exploratory only**. Dry-run is the default;
paid execution requires `--execute` and exact `--approve-spend-usd 10.00`. One provider call is
in flight, its reservation is persisted before each call, and resume never repeats an interrupted
cell. The 24-task fixture remains offline/open-book and is not the paid smoke corpus.
The smoke establishes correctness only, not efficiency or decision quality; its artifacts remain not decision eligible and cannot change a confirmatory gate.

The canonical roadmap is
[`docs/plans/2026-07-17-decision-quality-roadmap.md`](plans/2026-07-17-decision-quality-roadmap.md):
**H0** closes Elite correctness and wording gaps before merge; **H1** runs budget-matched,
randomized ablations with go/kill gates; **H2** adds a minimal Markdown evidence bundle,
claim ledger, deterministic citation-integrity checks, readiness tri-state, and Markdown/JSON
Decision Brief; **H3** proves onboarding, repeat use, and paid buyer pull; **H4** learns
escalation and roster weighting from outcomes. The plan also records retain/promote/retire
decisions for the old v1.2 backlog, commodity-versus-moat boundaries, demand-gated items, and
portfolio kill criteria. No horizon is a release commitment.

---

## 10. Downstream Boundary: conclave ↔ mcp-warden

**mcp-warden** imports conclave for dev-time design review and taxonomy work only, never as a runtime dependency. Security findings require determinism; conclave remains stochastic even though its consensus arithmetic is deterministic over LLM-assisted clustering. This boundary is load-bearing:

| | conclave (this project) | mcp-warden runtime |
|---|---|---|
| Nature | Stochastic, generative, multi-model | Deterministic, reproducible |
| Right use | Design review, eval, taxonomy labeling (dev time) | Runtime security adjudication |
| Dependency direction | — | imports conclave **at dev time only** |

If you find yourself wanting conclave inside mcp-warden's runtime decision path, that is a
design smell — re-read this section.

---

## 11. Licensing & Positioning

**License:** MIT (`pyproject.toml`) — permissive on purpose: a small primitive others embed.

**Market reality.** The "ask N models and reconcile" category is crowded — `llm-council-core`
(closest peer: library-first, direct-provider mode, anonymized ranking, structured verdicts,
`doctor`) and `the-llm-council` (library + CLI, adversarial critique, JSON-schema-validated
output) occupy the original niche directly. So **library-first + structured-result +
partial-failure-resilient + Model A/B/C anonymization are table-stakes, not a moat**, and we
no longer market them as distinctive. conclave is also **not** a general AI SDK — it does not
chase LiteLLM (routing/budgets), Vercel AI SDK (provider abstraction), LangChain/promptfoo
(evals), or Helicone (observability); it builds only what deepens the council wedge.

**Where we are *now* distinct** (re-anchored on what competitors have not replicated):

1. **The execution-traceable verdict (the v1.1 wedge).** A council answer you can inspect: structured
   positions, `conflicts` and `minority_reports` that cite `evidence_answer_ids`,
   `provider_votes`, a **deterministic** `consensus_score` (arithmetic over the model's
   clustering, never an LLM-emitted number), and a redacted `ModelHarnessManifest` recording
   which model + prompt version produced the disagreement analysis. Peers ship "structured
   verdicts" as synthesizer *content*; conclave's current verdict is a reproducible,
   answer-linked, provenance-stamped object. External source grounding remains Roadmap (§9).
2. **Owned, zero-LLM-SDK provider highway** — a single hand-owned httpx transport + adapter
   registry, no provider SDKs, no OpenRouter. Competitors lean on aggregators or vendor SDKs.
3. **Direct-keys / no-middleman + name-only key rigor** — never an aggregator, never a token
   proxy; the value never transits a data structure, is never serialized, is `redact()`-
   scrubbed from errors, and the manifest is proven secret-free (minimal-surface vs. BYOK).
4. **A telemetry-grade `CouncilResult` contract** — per-model latency + token usage + typed
   error capture as a *stable downstream contract* (the mcp-warden dev-time story).

We are the small, embeddable, **execution-traceable** council primitive — not a LangGraph/AutoGen
rival. Against the direct peers (`llm-council-core`, `the-llm-council`) we differentiate on
the execution-traceable verdict, the owned provider layer, the no-aggregator posture, and key rigor.

---

## 12. Open Product Questions

**Open:** none currently.

2. ~~**`vote` answer schema.**~~ **RESOLVED in v1.1** by *two* complementary deliveries: (a) a
   real constrained-choice **`vote` mode** (CAC-09 / #3) — fixed labelled ballot, plurality
   winner/split on `result.vote`; and (b) the verdict's structured `provider_votes`, which
   cluster free-form stances with evidence (LCD JSON Schemas + native structured output, §4a).
   The constrained-answer-format question is answered by the ballot; the tally question by
   `provider_votes`.

**Resolved (2026-06-08):** questions 1 (synthesizer-in-council), 3 (per-member overrides),
4 (server-mode scope, plus the 2026-06-09 #8 spike outcome), and 5 (first-class provider
criteria) are decided and archived for traceability in
[`docs/archive/pdd-resolved-questions-2026-06-09.md`](archive/pdd-resolved-questions-2026-06-09.md).
The numbering is preserved so the resolved Q2 keeps its identity.
