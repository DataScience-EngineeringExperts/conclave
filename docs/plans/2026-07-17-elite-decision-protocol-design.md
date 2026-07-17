# Elite Decision Protocol Design

## Outcome

Add a first-class `elite` council mode optimized for consequential decisions. Elite runs
favor decision quality over latency and cost while preserving conclave's existing BYO-key,
partial-failure, auditability, and library-first guarantees.

The protocol requires at least three successful responders. Three is the minimum that can
express a majority and a minority; four or five remain supported for higher-stakes use.
The threshold is an invariant, not a user setting, so `elite` cannot silently degrade into
an ordinary one- or two-model answer.

## Protocol

1. **Independent answer:** fan the original prompt to every member concurrently. Members do
   not see one another's answers, preventing early anchoring and groupthink.
2. **Evidence audit:** each surviving member receives an anonymized Model A/B/C panel and
   produces one council-wide critique. The prompt asks it to identify supported claims,
   conflicts, hidden assumptions, and externally unverified claims. It must cite stable
   answer IDs and must not invent sources.
3. **Revision:** each survivor receives its original answer, the anonymized panel, and all
   anonymized critiques, then writes one complete revised answer to the original prompt.
4. **Decision:** the existing synthesis and auditable-verdict pipeline processes the revised
   answers. The current `CouncilVerdict` remains canonical; elite does not create a competing
   verdict schema.

Each member phase is concurrent, making the protocol O(N) calls per phase rather than an
O(N-squared) pairwise debate. If fewer than three responses survive any member phase, the
protocol stops further calls and returns partial artifacts with an explicit failure reason.
It never raises for provider-side failure and never emits a final synthesis or verdict for an
incomplete elite run.

## Public API

- CLI: `conclave ask "..." --mode elite -c grok,gemini,claude`
- Async: `await Council(...).elite(prompt)`
- Sync: `Council(...).elite_sync(prompt)`
- Result: backward-compatible optional `CouncilResult.elite: EliteResult | None`

`EliteResult` records the protocol version, required responders, completion status, failure
reason, initial answers, critiques, and revisions. On success, `CouncilResult.answers` holds
the successful revisions. On incomplete runs it holds the latest complete answer-to-original
prompt stage, while critiques remain under `result.elite`.

Streaming is not supported initially because quality depends on completing and validating
each stage before the next begins. The CLI rejects `--mode elite --stream` explicitly.

## Auditability and safety

The manifest must cover initial, critique, and revision calls. Provider receipts gain a
backward-compatible optional `phase` field, while `providers_called` remains a unique member
list. Total latency and usage aggregate the recorded member calls across all three phases.
The manifest is rescanned before receiving `verified_no_secrets`.

Model identities remain anonymized inside cross-member prompts. Answer IDs provide evidence
links without disclosing provider brands. The audit prompt distinguishes externally
unverified claims from disproven claims because conclave has no retrieval or tool-use layer.

## Acceptance criteria

- Three successful members complete all stages and produce the existing auditable verdict.
- Four or five configured members may still complete if failures leave at least three.
- Fewer than three successes at any phase returns an incomplete result without a verdict.
- Critiques and revisions retain attributable answer IDs without provider-name leakage.
- Manifest receipts expose every member phase and remain secret-free.
- Existing modes, cache behavior, output schemas, and public APIs remain compatible.
- Unit, CLI, manifest, cache, secret-safety, lint, formatting, and full test suites pass.

