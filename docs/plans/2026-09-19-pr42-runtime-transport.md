# PR42 Runtime Transport Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create and independently verify one private, immutable, source-bound runtime prerequisite transport release for PR #42 without updating PR source or changing product or production state.

**Architecture:** An off-repository Python helper validates the authenticated receipt, immutable source branch, PR tree boundary, unchanged local input bytes, and current source-scanner contract before it creates a private prerelease. It uploads the fixed four-asset set, verifies GitHub-reported digest and size, validates the resulting manifest with the repository's own runtime-prerequisite authority, and emits the exact manifest for the already-authorized workflow dispatch.

**Tech Stack:** Python 3 standard library; `gh` authenticated REST API; repository runtime prerequisite validators; GitHub private prerelease assets.

---

### Task 1: Freeze and test the operation contract

**Files:**
- Create: `/home/ubuntu/dev/conclave/pr42-refresh-runtime-transport-be31.py`
- Test: read-only `--source be31d2ad4cce3909f00bf6f99fcf345db4e836a9 --review-head ffe32cefc60dbe6830b3589b93801df2c89b110d`

**Step 1:** Bind exact source SHA, repository identity, authenticated GitHub receipt, approval expiry, allowed scope, and downstream stop boundary.

**Step 2:** Require protected source-branch identity, clean local worktree, PR head/tree boundary, no later owner revocation, and no duplicate source-specific transport release.

**Step 3:** Verify four source assets and the local expat lock identity byte-for-byte, then require at least three hours of valid source Trivy contract lifetime.

**Step 4:** Run read-only preflight. Expected: all validation passes and no GitHub release/artifact changes occur.

### Task 2: Independent security review

**Files:**
- Review: `/home/ubuntu/dev/conclave/pr42-refresh-runtime-transport-be31.py`

**Step 1:** Review the helper for scope expansion, credential leakage, unexpected workflow actions, mutable-input use, duplicate-release behavior, and post-upload verification.

**Step 2:** Repair only review findings, rerun preflight, and record the review verdict before execution.

### Task 3: Create and verify the private transport

**Files:**
- Create outside source repository: `/tmp/pr42-prereq-be31d2ad4cce3909f00bf6f99fcf345db4e836a9-rp85/transport-manifest.json`

**Step 1:** Execute helper once. Expected: private prerelease at exact source SHA and exactly four uploaded assets.

**Step 2:** Read GitHub release state and asset digest/size metadata back through REST API.

**Step 3:** Validate generated manifest through repository authority. Expected: runtime prerequisite proof is available, but Gate 3 stays blocked until a separate artifact-specific authenticated receipt.

### Task 4: Dispatch only the approved prerequisite workflow

**Files:**
- No repository source files changed.

**Step 1:** Dispatch `Runtime Prerequisites` on `codex/owned-image-remediation` with exact source SHA and generated strict manifest.

**Step 2:** Stop at the GitHub `security-scan` environment approval. Provide Ernest the exact run URL, the `Review deployments` / `security-scan` action, and pass/fail condition.

**Step 3:** After human gate completion, independently validate resulting artifact and obtain a new exact artifact-specific receipt before any Gate 3 dispatch.
