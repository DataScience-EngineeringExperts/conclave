# Security Policy

conclave is a **bring-your-own-keys** multi-model council. It calls foundation-model
APIs with the user's own credentials, read from the environment by **variable name
only**. The most security-sensitive surface in the project is therefore key
handling: a weakness that causes a real API key to be stored, logged, serialized,
or echoed back to the user breaks the core trust promise. We treat reports against
that surface as the highest priority.

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
