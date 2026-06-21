# Archived: v0.x council-mode algorithm detail

> **Archived 2026-06-21** from `docs/PRODUCT_DESIGN_DOCUMENT.md` §4 during the v1.1
> "auditable council" pivot (CAC-08). The PDD now summarizes the modes and points here for
> the blow-by-blow "as built" algorithm prose. This is landed history, preserved verbatim
> for traceability — it is **not** a competing authority doc. When this file and the code
> disagree, the code wins; when this file and the PDD disagree, the PDD wins.

These are the step-by-step "as built" descriptions of the v0.1/v0.2 council modes
(`synthesize`, `raw`, `debate`, `adversarial`). They were moved out of the PDD to make
room for the v1.1 verdict content while keeping the PDD under its 500-line budget. The
v1.1 verdict layer (PDD §4a) sits *on top of* these modes — it adjudicates whatever
`answers` the mode produces and is orthogonal to the per-mode mechanics below.

---

## Synthesize algorithm (as built — v0.1)

1. Resolve requested friendly names to model ids via config.
2. Partition members into *available* (key present) and *skipped* (no key).
3. Fan out the prompt to all available members concurrently (`asyncio.gather`,
   `return_exceptions=True` as a belt-and-suspenders guard).
4. Each call returns a structured `ModelAnswer` (answer **or** error — `call_model` never
   raises for provider failures), so partial results always survive.
5. If `synthesize=True` and at least one member succeeded, build a synthesis prompt that
   embeds the original prompt and each successful answer, and call the synthesizer model.
6. If the synthesizer has no key, or no member succeeded, set `synthesis_error` and return
   raw answers only. A run with zero available members returns an empty-answer result
   rather than raising.

**Consensus note:** synthesize is a *generative* reconciliation, not a deterministic vote.
It is inherently stochastic. This matters for the mcp-warden boundary (PDD §10). The v1.1
verdict layer adds a *deterministic* consensus number on top (PDD §4a) — but that number
is arithmetic over a clustering, never the synthesizer's free text.

## Debate algorithm (as built — v0.2)

1. Resolve + partition members (same as synthesize). Assign each member a stable
   position-based letter label (`Model A`, `Model B`, …).
2. **Round 1:** independent fan-out of the bare prompt (reuses `Council.fan_out`).
3. **Rounds 2..N:** each surviving member is shown its own prior answer (told which
   letter is "you") plus its peers' prior answers, **anonymized by letter, not brand**,
   and asked to revise or defend. Anonymization reduces brand-bias (a model deferring to
   or attacking another *by name*) while preserving the cross-pollination that makes
   debate useful. The answer *body* is passed verbatim; only the attribution is relabeled.
4. **Drop-out:** a member that errors in a round is removed from subsequent rounds; the
   debate continues with survivors. If every member fails a round, the debate ends there.
5. The synthesizer consolidates the final round's surviving answers (same call path as
   synthesize, with a debate-specific system prompt).

**Convergence early-stop (issue #4, opt-in):** `converge_threshold` (config field,
`Council.debate` param, `--converge-threshold` / `--converge`/`--no-converge`) stops a
debate before `--rounds` once round-over-round answer stability
(`difflib.SequenceMatcher`, stdlib-only) crosses the threshold; off by default, recorded on
`CouncilResult.converged`/`convergence_score`, part of the debate cache key. This text-
stability signal is the debate `convergence_score` and is **deliberately distinct** from the
v1.1 verdict `consensus_score` — text-similarity between rounds is never used as an
agreement measure (PDD §4a).

## Adversarial algorithm (as built — v0.2)

1. Resolve + partition members. Pick the proposer: `--proposer` if given, else the first
   requested member; if that member has no key, fall back to the first available member.
2. **Propose:** the proposer answers the prompt (single-member `fan_out`).
3. **Refute:** every other available member is a CRITIC, explicitly prompted to find the
   strongest flaws in the proposal (not to agree). One critic failing never aborts the run.
   If the proposal itself failed, critics are skipped.
4. **Verdict:** the synthesizer acts as JUDGE — given the prompt, proposal, and critiques —
   accepting correct critiques, rejecting overstated ones, and issuing a verdict plus the
   strengthened final answer.

For an adversarial run, the v1.1 verdict layer surfaces unrefuted critic points as
`minority_reports` (PDD §4a).

## Result-model extension (backward-compatible — v0.2)

The deliberation modes extend `CouncilResult` **without breaking** synthesize/raw consumers:

- New `mode` field (`"synthesize" | "raw" | "debate" | "adversarial"`).
- New `rounds: list[DebateRound]` (debate) and `adversarial: AdversarialResult | None`.
- For **debate**, the final round is mirrored into the existing `answers`, and the
  consolidated answer into the existing `synthesis`. For **adversarial**, the proposal +
  critiques populate `answers` and the verdict mirrors into `synthesis`. Any existing
  consumer that reads `answers`/`synthesis`/`successful_answers` keeps working unchanged;
  new consumers read `rounds`/`adversarial` for the full structure. All fields are
  keys-free and serialize cleanly via `model_dump()` (`--json`).

The v1.1 `CouncilResult` v2 surface (`verdict`, `consensus_*`, `conflicts`,
`provider_votes`, `minority_reports`, `manifest`) extends this same contract again, also
backward-compatibly — every new field defaults to `None`/empty (PDD §4a).
