# Elite Synthetic QA v1

This directory is an **open-book synthetic harness QA** pack for debugging the Conclave H1
study method. It is not a paid pilot, production benchmark, confirmatory study, or evidence
that Elite improves decisions. Its results **must not support product-quality claims**.

## Contents

- `public_tasks.json` contains the 24 synthetic tasks visible to every experimental condition.
- `grader_keys.json` contains expected decisions, critical errors, and rubric anchors. Because
  it is committed beside the tasks, it is an open-book test fixture, not access-controlled
  grader material.
- `qa_protocol.md` freezes the synthetic QA execution and grading rules.
- `confirmatory_preregistration.md` is a gated template for a later, independently authored
  held-out study. It is not an active preregistration.

The pack has two prespecified macro-families with 12 tasks each, six subfamilies with four
tasks each, and balanced tiers and expected readiness outcomes:

| Axis | Distribution |
| --- | --- |
| Macro-family | 12 `operational_execution`; 12 `organizational_stewardship` |
| Subfamily | 4 each: procurement, release operations, product experiments, capital allocation, governance, staffing |
| Difficulty | 8 each: `tier_1`, `tier_2`, `tier_3` |
| Keyed readiness | 8 each: `ready`, `not_ready`, `indeterminate` |

All facts, organizations, people, products, and prices are fictional. The tasks require no
current knowledge and prohibit external assumptions. Public packets and grader keys use the
immutable `conclave_eval_v1` contracts and can be loaded with
`conclave.evals.dataset.load_public_tasks` and `load_grader_keys`.

## Frozen QA-only boundary

QA v1 may debug schemas, task clarity, rubric wording, grading workflows, execution
reliability, and report generation. It may not estimate unbiased model quality, establish
superiority, authorize product claims, tune a confirmatory holdout, or influence real
decisions. Any material task, key, prompt, rubric, or protocol change creates a new version
and new hashes; it must not silently replace this pack.

A paid execution requires a separately provisioned key artifact that model and runner
identities cannot read, an access-controlled grading workflow, and a frozen cryptographic
hash recorded before execution. The committed `grader_keys.json` must never be relabeled or
copied as that paid-study key artifact.
