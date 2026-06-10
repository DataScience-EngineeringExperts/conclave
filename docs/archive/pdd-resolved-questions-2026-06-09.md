# Archived — PDD §12 Resolved Product Questions (as of 2026-06-09)

These design decisions were resolved during pre-dev planning (2026-06-08) and are
archived here for traceability so the live `docs/PRODUCT_DESIGN_DOCUMENT.md` §12 can
stay focused on genuinely open questions. The only still-open question (Q2, `vote`
answer schema) remains in the PDD.

1. **Synthesizer-in-council policy.** ✅ **RESOLVED (2026-06-08): allow, document the
   self-reinforcement caveat, no code gate.** The default synthesizer (`claude`) may also be
   a council member in the same run; this mirrors the common "chairman" precedent and is
   low-stakes. The self-reinforcement risk is documented, not enforced.
3. **Per-member model/temperature overrides.** ✅ **RESOLVED (2026-06-08): yes, at the
   config level**, with the council-wide value as the default. Members may carry per-member
   `model`/`temperature` overrides in config; call-args overrides are out of scope for now.
4. **Server mode scope.** ✅ **RESOLVED (2026-06-08): localhost-bind only, no token-proxy
   path, explicit no-middleman guard.** If a local HTTP mode ships (#8), it binds to
   `127.0.0.1`, never becomes a hosted token path, and carries an explicit non-goal guard.
   (Peer `llm-council-core` shipped MCP + HTTP, so precedent exists — but the no-middleman
   non-goal §8 is load-bearing and overrides convenience.) **Spike outcome (2026-06-09, #8):**
   recommendation is **no-go on an HTTP server** (even a `127.0.0.1` bind carries
   DNS-rebinding/CSRF surface and dilutes the "small" non-goal); if cross-process access is
   wanted, prefer a thin **stdio MCP server** (zero network bind, matches the mcp-warden
   sibling story). Final disposition is the maintainer's.
5. **First-class provider expansion criteria.** ✅ **RESOLVED (2026-06-08): promote a
   pass-through to a typed default when it is OpenAI-compatible (or a native adapter exists)
   AND has a stable public API AND shows common demand.** The long tail stays config-only via
   `endpoints:`. (Four vendors promoted under this rule in v0.3 #5: `groq`, `deepseek`,
   `mistral`, `together`.)
