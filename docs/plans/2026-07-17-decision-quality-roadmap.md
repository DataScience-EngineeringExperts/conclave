# Conclave Decision Quality Roadmap

**Status:** Active; Horizon 0 released in v1.2.0, Horizon 1 exploratory pilot preparation underway
**Date:** 2026-07-17
**Current product:** v1.2.0 stable on PyPI and GitHub; Elite is released, while
decision-quality and efficiency claims remain unvalidated

## Thesis

Conclave should become the smallest tool that produces a **source-grounded,
execution-traceable decision record whose quality is empirically proven**. The product is
not "more agents." It is a disciplined path from source material to competing claims,
critique, decision readiness, and an exportable brief, with receipts that let a reviewer
reconstruct what happened.

That wording is intentionally narrower than source-auditable product language. Today,
Conclave traces model execution and the model-assisted clustering behind a verdict. Its
`evidence_answer_ids` identify council answers, **not external evidence**. Until source
grounding and deterministic citation validation ship, the honest claim is
**execution-traceable**, not source-auditable.

## Product doctrine

- Quality claims require comparative evidence, not architectural intuition.
- A completed protocol is not necessarily a decision ready for use.
- Atomic, source-linked claims are the unit of quality; prose fluency is not.
- Preserve dissent, uncertainty, and failure states rather than forcing a verdict.
- Keep BYO keys, direct provider access, the owned transport, and a library-first API.
- Build the narrowest workflow that improves consequential decisions; integrate commodity
  infrastructure instead of recreating it.

## Horizon 0 — Elite correctness and release gate (complete)

Elite shipped in v1.2.0 after the release gate verified these baseline contracts:

1. **Persistent identities.** Each initial answer has a stable identifier that survives claim
   audit, revision, synthesis, serialization, and cache replay. Do not present those IDs as
   external citations.
2. **Two states, not one.** Separate `completed` (all required member phases succeeded)
   from `decision_readiness`. Readiness is `ready`, `not_ready`, or `indeterminate`, with
   machine-readable reasons. A completed run may still be `not_ready`.
3. **Full-call receipts.** Buffered Elite manifest totals include member, claim-audit, revision, synthesis,
   repair, and verdict-extraction calls, with phase, latency, usage, errors, and prompt/schema
   versions. Never imply completeness when final calls are absent from totals.
4. **Version-aware cache.** Cache identity must cover protocol, prompt, schema, model,
   generation settings, source-bundle digest, and relevant config versions so incompatible
   artifacts cannot replay as current.
5. **Custom config threading.** Every Elite phase, including synthesis and verdict
   extraction, must receive the same resolved custom-endpoint/provider configuration.
6. **Correct language.** Use "answer/claim audit" until external sources exist. Product copy
   must say execution-traceable. Answer IDs prove
   provenance within a run; they do not prove truth.

**Gate outcome:** all six items have regression coverage; the complete suite, static checks,
secret scan, package build/install, release review, and documentation checks passed. A capped
12-cell paid smoke completed every planned cell, and a separate three-provider connectivity
smoke passed from an isolated runner using the published package. These results validate
execution correctness only, not comparative quality or efficiency.

## Horizon 1 — prove that the protocol improves decisions

Run a budget-matched randomized ablation before claiming Elite is better. Compare at least
single-frontier-model, ordinary synthesis, adversarial, and Elite conditions using the same
task set and matched token or dollar budgets. Randomize condition order and blind human
graders to mode and provider identity.

### Current H1 position (2026-07-19)

The offline harness, blinding/scoring workflow, USD 10 execution ceiling, authenticated
checkpoints, and paid correctness smoke are complete. The 24-task open-book QA pack remains
synthetic and offline-only evidence. The next valid step is a separately frozen 20-30 task
private exploratory pilot with protected grader keys, six matched conditions, two independent
human graders, and adjudication. Grader calibration is staged before the full pilot. DSE-690
and DSE-708 remain open until pilot evidence supports a go, redesign, or kill decision.

### Program stages

1. **Pilot:** 20-30 diverse, answerable decision tasks to debug rubrics, establish grader
   agreement, estimate variance, and set confirmatory sample size. Results are exploratory.
2. **Confirmatory:** preregister hypotheses, primary metric, exclusions, analysis, minimum
   effect, and sample size; freeze prompts and scoring before running the held-out set.
3. **Shadow:** run the candidate protocol alongside real decisions without influencing them;
   compare usefulness, unsupported claims, reversals, cost, latency, and reviewer effort.

### Atomic metrics

- supported-claim precision and material-claim source coverage;
- factual error and unsupported-claim rates, severity-weighted;
- recommendation correctness or expert preference on tasks with a defensible reference;
- calibrated readiness: error/reversal rate by `ready`/`not_ready`/`indeterminate`;
- conflict and minority-view recall against expert annotation;
- decision completeness and actionability on a fixed rubric;
- inter-rater reliability, run-to-run stability, latency, tokens, cost, and reviewer minutes.

Report distributions and confidence intervals, not one composite "quality score." Cost and
latency are co-primary constraints, not footnotes.

### Anti-gaming controls

- held-out tasks and sources; no tuning on confirmatory cases;
- atomic claim scoring before holistic preference scoring;
- blinded, randomized outputs with length-normalized views;
- identical evidence access and budget ceilings across conditions;
- duplicate/leakage checks, adjudicated grader disagreements, and preserved raw scores;
- failure-inclusive analysis: timeouts, abstentions, invalid citations, and incomplete runs
  remain in the denominator;
- publish prompt, roster, version, exclusions, and analysis artifacts with each result.

### Go/kill gates

Advance to productization only if Elite shows a statistically credible improvement on the
preregistered primary quality metric and no material regression in severe-error rate,
calibration, or reviewer effort, within the agreed cost/latency ceiling. Repeat on at least
two task families. Kill or redesign Elite if it does not beat the best budget-matched
baseline, gains disappear on held-out tasks, graders cannot agree on the rubric, or the same
quality is achieved by a simpler single-model critique/revision loop.

## Horizon 2 — source-grounded decision records

Build the minimum evidence product, in this order:

1. **Markdown evidence bundle.** Accept local Markdown containing source blocks with stable
   source IDs, title/origin metadata, and quoted or summarized content. No crawler, vector
   database, or document platform in the first version.
2. **Integrity.** Canonicalize each source block and store a content digest. Record the bundle
   digest in cache identity and the run manifest.
3. **Structured claim audit.** Emit atomic claims with disposition, materiality, supporting
   and contradicting source IDs, exact citation spans, and audit rationale.
4. **Deterministic validation.** Verify that every cited source and span exists and that the
   quoted text matches the digested bundle. This validates citation integrity, not semantic
   truth; semantic entailment remains scored and explicitly probabilistic.
5. **Claim ledger.** Preserve claim lineage from initial answer through critique, revision,
   synthesis, and final disposition, including unresolved conflicts and minority positions.
6. **Readiness tri-state.** Compute `ready`, `not_ready`, or `indeterminate` from explicit
   policy and validated artifacts; never infer readiness from protocol completion alone.
7. **Decision Brief.** Export equivalent Markdown and JSON containing recommendation,
   alternatives, assumptions, claim ledger, conflicts, minority report, source citations,
   readiness, execution receipts, and limitations.

**Go gate:** deterministic citation checks catch seeded broken citations, humans can trace
every material brief claim to a source or explicit unsupported status, and JSON/Markdown
round-trip without semantic drift. **Kill criterion:** if the minimal bundle adds process
cost without improving supported-claim precision or reviewer time in the confirmatory study,
do not expand ingestion.

## Horizon 3 — prove buyer pull and repeat use

Target skeptical engineering and security leaders making architecture, vendor, build/buy,
and risk decisions. Run small experiments rather than building a sales platform:

- **Onboarding:** five fresh users install, configure three providers, run a supplied evidence
  bundle, and export a brief; measure completion, time-to-first-brief, and support needed.
- **Buyer pilots:** 3-5 teams use Conclave on live but reversible decisions for 4-6 weeks;
  compare their current review process, reviewer minutes, defects caught, and decision reuse.
- **Repeat use:** measure a second self-initiated decision within 30 days and whether briefs
  are shared in an existing review workflow.
- **Paid design partners:** ask qualified pilots to pay for a defined support/evaluation
  package before building collaboration or hosting features.

**Go gate:** at least three teams complete pilots, two use it again without prompting, and
two accept a paid design-partner offer or equivalent written budget commitment. **Kill or
pivot:** fewer than two teams repeat after onboarding fixes, no buyer treats the brief as a
review artifact, or willingness to pay depends on hosted collaboration rather than decision
quality. Revisit the persona and workflow before adding features.

## Horizon 4 — learn from outcomes

Only after repeat use exists, add an opt-in outcome journal linking a decision record to later
results, reversals, and reviewer feedback. Use that evidence to learn, by task family:

- when ordinary synthesis is enough and when escalation to Elite pays;
- which roster compositions improve quality per dollar/minute;
- whether provider weights help out of sample;
- which readiness signals predict reversals or severe errors.

Escalation and roster weighting must start as offline recommendations, be evaluated on
held-out outcomes, retain an explicit manual override, and never hide minority views. Kill
learned weighting if lift does not replicate, creates provider monoculture, worsens
calibration, or cannot be explained from recorded evidence.

## Previous v1.2 backlog disposition

| Previous item | Disposition | Reason / gate |
|---|---|---|
| Narrow eval harness and mock/replay transport | **Promote to H1** | Required for reproducible ablations; keep narrow and artifact-based. |
| `conclave doctor` and provider diagnostics | **Retain in H3** | Valuable onboarding aid; implement only against observed setup failures. |
| Generation-settings substrate | **Retain** | Build the minimum needed for budget-matched experiments and receipts. |
| Thin profiles (`cheap`, `balanced`, `frontier`, `critic`) | **Retire as product presets for now** | Roster/profile names imply quality before evidence; experiments may use internal fixed conditions. |
| Regular/smart engagement modes | **Retire** | Product framing is decision quality, not engagement. |
| Profile compilation and generic routing | **Demand-gated** | Commodity gateway territory; add only for a proven decision workflow. |
| `cheap_then_smart` adaptive routing | **Promote concept to H4** | Learn escalation from outcomes; do not trigger on unvalidated confidence. |
| Capability cache/discovery | **Retain, demand-gated** | Add only when provider drift blocks pilots; require explicit refresh and provenance. |
| Local HTTP server | **Retire / no-go** | Prior spike found no need; keep library and CLI surfaces. |
| Thin stdio MCP | **Demand-gated** | Build only after repeated integration requests from pilots. |

## Commodity versus moat

**Commodity/integrate:** provider routing, generic model catalogs, hosted observability,
prompt management, vector search, crawlers, document conversion, workflow engines, dashboards,
and general eval platforms. Conclave should emit clean artifacts and interoperate with them.

**Potential moat, still to be proven:** the evaluation corpus and atomic scoring method;
source/claim lineage across multi-model critique and revision; calibrated decision-readiness
signals; evidence about which deliberation protocol and roster works for which decision;
and trusted, portable Decision Briefs with deterministic integrity checks. The owned provider
highway and BYO-key rigor are valuable control and trust properties, but not sufficient moat.

## Do not build without demonstrated demand

- hosted multi-tenant SaaS, accounts, billing, collaboration, or a cloud prompt proxy;
- a general agent framework, gateway/router, provider SDK, or workflow engine;
- broad RAG, crawling, vector storage, or document-management infrastructure;
- a local review UI, IDE extension, stdio MCP, or OpenTelemetry exporter;
- generic dashboards, marketplace profiles, autonomous action execution, or security gates;
- streaming Elite, dynamic swarms, automatic model procurement, or opaque self-improvement.

Any item above needs repeated pilot evidence, a named buyer/workflow, a measurable quality or
adoption hypothesis, a simpler-alternative analysis, and an explicit kill date.

## Portfolio kill criteria

Stop or materially reframe the Decision Quality strategy if, after H1 and corrected H3
onboarding, any of these hold:

- no protocol beats the strongest budget-matched baseline on held-out tasks;
- source grounding does not reduce severe unsupported claims or reviewer effort;
- quality gains require cost/latency buyers reject and adaptive escalation cannot recover it;
- users value model comparison as entertainment but do not use briefs in real decisions;
- pilots do not repeat or produce credible willingness to pay;
- outcome data is too sparse, biased, or sensitive to support safe learning;
- a simpler commodity tool matches the full decision record and measured quality.

If killed, preserve Conclave as a small open-source council/evaluation primitive; do not chase
usage with unrelated platform features.
