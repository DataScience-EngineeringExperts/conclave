# conclave — Product Design Document

> **Status:** v1.0 stable (the BYO-keys multi-model council: synthesize/raw/debate/
> adversarial, 9 providers, owned httpx provider highway, key-leak hardening, streaming,
> cache). **v1.1 — the auditable council — SHIPPED:** every run now yields a structured,
> agreement-scored, fully auditable **verdict** plus a redacted execution **manifest** (see
> §4a). This is the **canonical authority document** for conclave's product scope, design,
> and roadmap. When this document and any other doc disagree, this document wins. Code is the
> source of truth for *current behavior*; this document marks anything not yet in code as
> **Roadmap**.

- **Repo:** `/Users/ernestprovo/dev/conclave/`
- **License:** MIT
- **Author:** Data Science & Engineering Experts, Inc. (DSE)
- **Last updated:** 2026-06-21

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

**The v1.1 wedge — the auditable council.** A synthesis paragraph is not enough to *act* on.
v1.1 makes the product identity precise: **a multi-model council verdict you can act on —
structured, scored for agreement, fully auditable.** Every run yields a `CouncilVerdict`
exposing agreement, disagreement (`conflicts`), minority views (`minority_reports`), and
per-provider votes (`provider_votes`); a deterministic `consensus_score` (arithmetic over the
model's clustering, *never* an LLM-emitted number); and a redacted `ModelHarnessManifest`
recording how the run executed and which model produced the disagreement analysis (§4a). The
roadmapped `vote` mode is therefore **absorbed and superseded** — "show me who voted for what"
is now `provider_votes`, with evidence, not a separate mode.

conclave's first real use was an **adversarial design review**: a council of Grok, Gemini,
Perplexity, and Claude critiquing a security-tool strategy and catching flaws a single
model missed. That origin is why the adversarial and debate modes are first-class — they
are now built, not a bolt-on. The product stays lightweight: a **library-first primitive
with structured, auditable results**, not an agent framework and not a general AI SDK — it
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
| ~~**vote**~~ | **ABSORBED (v1.1)** | ~~Structured majority with reported split.~~ Superseded by the v1.1 verdict: `provider_votes` records which provider took which position (with evidence) and `consensus_label`/`consensus_score` report the split deterministically — no separate mode needed. See §4a. |

**Mode algorithms (as built).** The step-by-step "as built" prose for synthesize / raw /
debate / adversarial (fan-out + partial-results, peer anonymization + drop-out, proposer →
critic → judge) is landed history and lives in
[`docs/archive/pdd-v0.x-modes-detail.md`](archive/pdd-v0.x-modes-detail.md). In brief: every
mode fans out concurrently, captures each call as a `ModelAnswer` (answer **or** redacted
error — `call_model` never raises), and survives partial failure; the deliberation modes
extend `CouncilResult` (`mode`, `rounds`, `adversarial`) backward-compatibly so v0.1
`answers`/`synthesis` consumers keep working. The mode *text* output is a generative
reconciliation, inherently stochastic (load-bearing for the mcp-warden boundary, §10); the
v1.1 verdict (§4a) adds a *deterministic* agreement number on top, by arithmetic over a
clustering, never lifted from that free text.

---

## 4a. The Auditable Verdict (v1.1)

The verdict layer turns a council run's answers into a structured, agreement-scored,
auditable adjudication — on top of any mode, default-on, never breaking the v0.1 surface
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

**Evidence is the product, not a nicety.** Every clustered stance cites `evidence_answer_ids`
(the member `answer_id`s backing it) and every conflict names the positions in tension — a
conflict that just says "models disagreed about cost" without pointing at answers is a
*failure*. `ProviderVote.confidence` is recorded but **never used in arithmetic**.

### Deterministic consensus — the auditability fix
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

**Why it is auditable, not theater.** The extraction schema carries *no* consensus field —
`verdict_extraction_json_schema()` strips it and `VerdictExtractionModel` ignores extra keys,
so a model that smuggles a number in is dropped by the validator. The module deliberately
does **not** import `difflib`: text-similarity is the debate `convergence_score` (a
*forbidden* consensus measure), never conflated with agreement. The single LLM-assisted step
is the **semantic clustering** of stances; the number is reproducible arithmetic over it,
each cluster cites its `evidence_answer_ids`, and the manifest records which model + prompt
version did the clustering — so the score is traceable.

### Verdict extraction + native structured output
`extract_verdict(prompt, member_answers, *, synthesizer_name, synthesizer_model_id,
config=None) -> VerdictSynthesisResult(verdict, extraction, verdict_absent_reason)`
(`verdict_synthesis.py`) makes **one** extraction call asking the synthesizer model to
*cluster* stances (not to re-answer, not to emit a number), validates, repairs once, falls
back gracefully — never raises.

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
`secret_safety` stamp is re-run over the final content. **Cost:** default-on adds exactly
**one** extra synthesizer call per run; the opt-out exists for cost-sensitive callers.

### ModelHarnessManifest — first-class, secret-free
The `ModelHarnessManifest` (`manifest.py`) rides on every `CouncilResult` — *not* behind a
debug flag. It records `request_id`, `conclave_version`, `mode`, `providers_considered/
called/skipped` (each skip a `ProviderSkip{name, reason}`), `model_ids`,
`generation_settings`, `receipts` (each a `ProviderExecutionReceipt{name, provider,
model_id, generation_settings, latency_ms, usage, error(redacted), schema_valid}`),
`total_latency_ms`, `total_usage`, `schema_valid`, `redacted_errors`. Verdict-provenance
slots: `verdict_extraction: VerdictExtraction{model_id, prompt_version}` (which model + prompt
version produced the disagreement analysis — *the* auditability hook), `verdict_type`,
`consensus_method`, `verdict_absent_reason`. Two deliberate honesty choices:

- **No invented pricing.** `estimated_cost` is `None` (a wrong number in an audit receipt is
  worse than none); `pricing_snapshot_date` is the dated-estimate slot, `None` until a real
  pricing table exists. Usage (tokens) is recorded; cost is not guessed.
- **Proven secret-safety.** `secret_safety` defaults to `unverified`, promoted to
  `verified_no_secrets` **only** after `scan_for_secret_material()` proves the serialized
  manifest free of forbidden substrings (`sk-`, `bearer`, `authorization`, `api_key`,
  `x-api-key`). Key *values* never appear; errors are redacted upstream and re-redacted on
  construction.

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
| `council.py` | `Council` — primary importable entry point. Resolves names, partitions members, and exposes two reusable primitives: `fan_out` (the single concurrent + partial-failure call loop) and `synthesize_blocks` (the single synthesizer/judge call path). Hosts the public mode API: `ask`/`ask_sync` (synthesize/raw), `debate`/`debate_sync`, `adversarial`/`adversarial_sync`. Sync wrappers guard against a running event loop. Runs `_apply_verdict` (default-on; `extract_verdict=False` opts out) on both buffered + streaming paths after the manifest exists. |
| `verdict.py` | Public verdict/member Pydantic types (`CouncilVerdict`, `CouncilPosition`, `CouncilConflict`, `ProviderVote`, `MinorityReport`) + the LCD JSON Schemas (`verdict_json_schema`/`member_answer_json_schema`/`verdict_extraction_json_schema`) usable across all three native structured-output surfaces; `VERDICT_SCHEMA_VERSION`. |
| `agreement.py` | Deterministic consensus: `consensus_score` (`position_cluster_ratio_v1` — largest cluster / positioned members; `None` for N<2) + `consensus_label` buckets. Pure arithmetic, no `difflib`, never LLM-emitted. |
| `verdict_synthesis.py` | `extract_verdict` engine: one extraction call (clusters stances, never emits a number), native `output_contract` enforcement + prompt-level fallback, validate → repair-once → graceful `verdict=None`; the three verdict-absent reasons; provenance on every return path. |
| `manifest.py` | `ModelHarnessManifest` (first-class on every result), `ProviderExecutionReceipt`/`ProviderSkip`/`VerdictExtraction`, and `scan_for_secret_material()` → `secret_safety` stamp. No key values; `estimated_cost` left `None`. |
| `modes.py` | Deliberation orchestration: `run_debate` (multi-round, anonymized peers, drop-out) and `run_adversarial` (propose → refute → verdict). Built entirely on `Council.fan_out` + `synthesize_blocks` — no duplicated concurrency or synthesizer code. |
| `prompts.py` | Role/template strings for debate and adversarial (member, critic, judge, debate-final system prompts) and the anonymized peer-block builder. Separates *what each role is told* from *when to call whom*. |
| `providers.py` | `call_model` (+ `call_model_stream`) — the single async call path: resolve adapter, read key *by name at call time*, call adapter+transport (with an optional `output_contract`), parse, capture latency/usage/redacted-error into a `ModelAnswer`; never raises for provider-side failures. |
| `transport.py` | The single async network boundary: `post_json` (buffered) + `stream_sse` (issue #7) — the only two httpx call sites in the highway. |
| `streaming.py` | Streaming engine (issue #7): `stream_ask` interleaves members via an `asyncio.Queue`, optionally streams the synthesizer, ends with a `done` event whose `CouncilResult` (incl. verdict) matches the buffered shape. synthesize/raw only. |
| `adapters/__init__.py` | `resolve_adapter(model_id, config)` — the provider registry + **extension seam**: one registration per family; config-only for OpenAI-compatible endpoints. |
| `adapters/base.py` | `ProviderAdapter` protocol, `OutputContract` (native-structured-output request), `ProviderError`, and `redact()` (error-string secret scrubber). |
| `adapters/openai_compat.py` | `OpenAICompatAdapter` — openai/xai/perplexity/groq/deepseek/mistral/together + custom endpoints; per-provider completions URL (Perplexity no `/v1`; Groq under `/openai/v1`); `response_format` json_schema when an `output_contract` is set. |
| `adapters/anthropic.py` | `AnthropicAdapter` — native `/v1/messages` (system-hoist, `max_tokens` required); `input_schema` tool for an `output_contract`. |
| `adapters/gemini.py` | `GeminiAdapter` — native `generateContent` (role-map, `systemInstruction` hoist, `usageMetadata`); `responseSchema` for an `output_contract`. |
| `registry.py` | Single source of truth for name→model-id defaults + provider→env-var mapping. Key *presence* only — never values. |
| `config.py` | Loads/merges `~/.conclave/config.yml` over defaults; resolves model ids + named/CSV councils; parses `endpoints:`. Keys-free by construction. |
| `models.py` | Pydantic contract: `TokenUsage`, `ModelAnswer` (stable `answer_id`), `CouncilResult` v2 — adds top-level `verdict`/`consensus_score`/`consensus_method`/`consensus_label`/`conflicts`/`provider_votes`/`minority_reports`/`manifest` (all backward-compatible). The stable importable surface for downstream consumers. |
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
  pass-through; BYO-keys by env-var name with graceful skip; concurrent fan-out;
  structured `CouncilResult` (latency, usage, per-model error); CLI (`ask`/`providers`,
  `--json`); config (named models/councils, default synthesizer); sync + async library
  API; transport-mocked test suite (no network, no keys).
- **v0.2:** `debate` (multi-round, anonymized peers, drop-out, `CouncilResult.rounds`) and
  `adversarial` (proposer → critics → judge, `CouncilResult.adversarial`); backward-
  compatible `CouncilResult` extension; both on the library API + CLI with rich rendering.
- **v0.3:** **LiteLLM removed** → owned `httpx` provider highway + adapter registry (§6),
  the only network dependency; three adapters cover all nine providers; custom
  OpenAI-compatible `endpoints:` (config-only); key-leak hardening via `redact()` (was §9
  item 7); `call_model` signature + never-raises contract unchanged.
- **v1.0 (stable):** distribution name `conclave-cli`; OIDC Trusted-Publishing release
  workflow + Sigstore + PEP 740; key-leak threat model (`SECURITY.md`, cause-chain fix,
  transport-logging guard default-on); versioned synthesis prompt
  (`SYNTHESIS_PROMPT_VERSION` → `result.prompt_version`); streaming (synthesize/raw) +
  optional result cache + debate convergence early-stop.
- **v1.1 (the auditable council):** `CouncilResult` v2 — `verdict` + `consensus_*` +
  `conflicts`/`provider_votes`/`minority_reports` + first-class `manifest`; deterministic
  `position_cluster_ratio_v1` consensus; native + fallback structured output across
  OpenAI/Anthropic/Gemini; the verdict-optional rule; verdict default-on with
  `Council(extract_verdict=False)` opt-out. The `vote` mode is **absorbed** by
  `provider_votes`. Full detail in §4a.

---

## 8. Non-Goals (v0.1, and some permanent)

- **Not a runtime adjudicator.** conclave is stochastic; it must not be a deterministic
  decision gate (§10). **Permanent** for synthesize/debate/adversarial — *and* for the v1.1
  verdict: the verdict's *clustering* is LLM-assisted (stochastic), so even the deterministic
  `consensus_score` is not a reproducible security gate. The number is auditable, not authoritative.
- **Not an agent framework.** No tool-calling graphs, stateful agents, or orchestration DSL — we compete by being *small*. (Permanent.)
- **Not a key manager / secrets vault.** conclave reads env vars; it does not provision, rotate, store, or proxy keys. (Permanent.)
- **No hosted/proxied token path.** No conclave-operated endpoint that sees prompts or takes a margin — BYO-keys, direct-to-provider, always. (Permanent.)
- **No streaming for debate/adversarial** (synthesize/raw streaming landed in v0.3, #7).
- **No server mode** (possible Roadmap, §9).
- ~~**No `vote` mode** yet.~~ **Absorbed in v1.1** — `provider_votes` + `consensus_label`/`consensus_score` deliver "who voted for what, and the split" with evidence (§4a).

---

## 9. Roadmap

`adversarial`/`debate` shipped in v0.2; streaming/cache/convergence in v1.0; the **auditable
council shipped in v1.1** (§4a — the wedge).

### v1.2 — the "Operable Council" (DEMAND-GATED, not scheduled)
v1.2 is held behind a **prove-it gate**: put the auditable verdict in front of real users
first; observed pull authorizes the build. This substrate is demoted from the old
"engagement-modes-first" plan and only builds if demand appears — **engagement modes**
(regular/smart) on a generation-settings substrate; **thin task profiles**
(cheap/balanced/frontier/critic); **profile compilation + routing**
(`parallel_synthesize`, `sequential_fallback`, `cheap_then_smart`); a **capability cache**
with explicit refresh/discovery; `conclave doctor` + `providers` subcommands; a minimal
mock/replay transport; and a narrow eval harness. The `cheap_then_smart` soft-triggers (low
confidence / high disagreement) are gated behind **proven scoring** — flag-only until the
score is validated against real runs.

### Landed history (kept struck-through for traceability)

1. ~~**`vote` mode**~~ **ABSORBED in v1.1** — `provider_votes` + `consensus_label`/`consensus_score` deliver "who voted for what + the split" with evidence; the structured-answer-schema prerequisite is satisfied by the LCD schemas + native structured output (§4a).
2. ~~**Debate convergence/stop criteria**~~ **LANDED (#4)** — opt-in `converge_threshold` early-stop on round-over-round text stability (`difflib`); off by default; recorded on `CouncilResult.converged`/`convergence_score`. (NB: text-stability ≠ the verdict `consensus_score`, §4a.)
3. ~~**More first-class providers**~~ **LANDED (#5)** — `groq`/`deepseek`/`mistral`/`together` promoted to typed defaults (`registry.OPENAI_COMPAT_PROVIDERS`); no native adapter; aggregators excluded (§11).
4. ~~**Caching**~~ **LANDED (#6)** — opt-in result cache (`config.cache` / `--cache`), off by default, on-disk, secret-free SHA-256 key, corrupt = silent miss; hit on `CouncilResult.cached`.
5. ~~**Streaming**~~ **LANDED (#7)** — `synthesize`/`raw` streaming via `Council.ask_stream` + CLI `--stream` over `transport.stream_sse`; never-raises + partial text preserved; non-streaming default byte-for-byte unchanged.
6. **Local HTTP/server mode (open)** — a *local* server for convenience only; must not become a hosted token path. **Spike #8 (2026-06-09): no-go on HTTP** (`127.0.0.1` still carries DNS-rebinding/CSRF surface; the library already serves in-process); if cross-process access is wanted, prefer a thin **stdio MCP server**. Final disposition is the maintainer's.
7. ~~**Key-leak hardening**~~ **LANDED in v0.3** via `redact()` on every provider/transport error before `ModelAnswer.error` (§3).

**Roadmap discipline:** items are reprioritized freely but not *removed* on a single data
point; completed items are **marked done in place** (struck through with a "LANDED" note).

---

## 10. Downstream Boundary: conclave ↔ mcp-warden

**mcp-warden** (sibling MCP-server security gateway) **imports conclave as a DEV-TIME
dependency only** — adversarial design review of warden's strategy, taxonomy brainstorming.

**mcp-warden will NOT use conclave as a RUNTIME dependency.** Security findings require
determinism and reproducibility; a stochastic council is the wrong tool for runtime
adjudication. The v1.1 verdict does **not** change this: its `consensus_score` is
deterministic *arithmetic*, but the *clustering* it scores is LLM-assisted, so the verdict
is auditable, not a reproducible gate (§8). This boundary is deliberate and load-bearing:

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

1. **The auditable verdict (the v1.1 wedge).** A council answer you can *act on*: structured
   positions, `conflicts` and `minority_reports` that cite `evidence_answer_ids`,
   `provider_votes`, a **deterministic** `consensus_score` (arithmetic over the model's
   clustering, never an LLM-emitted number), and a redacted `ModelHarnessManifest` recording
   which model + prompt version produced the disagreement analysis. Peers ship "structured
   verdicts" as synthesizer *content*; conclave's verdict is a reproducible, evidence-cited,
   provenance-stamped object. **This is now the most defensible claim.**
2. **Owned, zero-LLM-SDK provider highway** — a single hand-owned httpx transport + adapter
   registry, no provider SDKs, no OpenRouter. Competitors lean on aggregators or vendor SDKs.
3. **Direct-keys / no-middleman + name-only key rigor** — never an aggregator, never a token
   proxy; the value never transits a data structure, is never serialized, is `redact()`-
   scrubbed from errors, and the manifest is proven secret-free (minimal-surface vs. BYOK).
4. **A telemetry-grade `CouncilResult` contract** — per-model latency + token usage + typed
   error capture as a *stable downstream contract* (the mcp-warden dev-time story).

We are the small, embeddable, **auditable** council primitive — not a LangGraph/AutoGen
rival. Against the direct peers (`llm-council-core`, `the-llm-council`) we differentiate on
the auditable verdict, the owned provider layer, the no-aggregator posture, and key rigor.

---

## 12. Open Product Questions

**Open:** none currently.

2. ~~**`vote` answer schema.**~~ **RESOLVED in v1.1** — the question (constrained answer
   format vs. post-hoc tally, and its structured-output prerequisite) is moot: the verdict
   ships the LCD verdict/member JSON Schemas enforced via native structured output across
   all three adapters, and `provider_votes` records the per-provider positions. The
   structured-output prerequisite this question flagged is now landed (§4a).

**Resolved (2026-06-08):** questions 1 (synthesizer-in-council), 3 (per-member overrides),
4 (server-mode scope, plus the 2026-06-09 #8 spike outcome), and 5 (first-class provider
criteria) are decided and archived for traceability in
[`docs/archive/pdd-resolved-questions-2026-06-09.md`](archive/pdd-resolved-questions-2026-06-09.md).
The numbering is preserved so the resolved Q2 keeps its identity.
