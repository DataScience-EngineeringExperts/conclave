"""conclave -- bring-your-own-keys multi-model council.

Public library API::

    from conclave import Council
    council = Council(models=["grok", "perplexity"], synthesizer="claude")

    # synthesize (default) / raw
    result = council.ask_sync("Your prompt")            # sync
    result = await council.ask("Your prompt")            # async

    # streaming (synthesize/raw only): incremental StreamEvents
    async for event in council.ask_stream("Your prompt"):
        if event.type in ("member_delta", "synthesis_delta"):
            print(event.text, end="", flush=True)
        elif event.type == "done":
            result = event.result   # full CouncilResult, same shape as ask()

    # multi-round debate
    result = await council.debate("Your prompt", rounds=3)
    result = council.debate_sync("Your prompt", rounds=3)

    # adversarial: propose -> refute -> verdict
    result = await council.adversarial("Your prompt", proposer="grok")
    result = council.adversarial_sync("Your prompt")

The returned :class:`CouncilResult` carries each member's raw answer (with
latency, token usage, and any error) plus the merged synthesis. For ``debate``
it also carries per-round answers (``rounds``); for ``adversarial`` it carries
the proposal/critique/verdict structure (``adversarial``).

The ``*_sync`` wrappers close the pooled HTTP client automatically. Long-lived
consumers that drive the async API on their own event loop (e.g. a server) must
release the process-wide connection pool on shutdown::

    import conclave
    await conclave.aclose()   # or: await council.aclose()
"""

from __future__ import annotations

from .config import ConclaveConfig, load_config
from .council import Council
from .manifest import (
    ModelHarnessManifest,
    ProviderExecutionReceipt,
    ProviderSkip,
    VerdictExtraction,
)
from .models import (
    AdversarialResult,
    CouncilResult,
    DebateRound,
    ModelAnswer,
    StreamEvent,
    TokenUsage,
)
from .transport import aclose, guard_transport_logging
from .verdict import (
    VERDICT_EXTRACTION_PROMPT_VERSION,
    VERDICT_SCHEMA_VERSION,
    CouncilConflict,
    CouncilPosition,
    CouncilVerdict,
    MinorityReport,
    ProviderVote,
    VerdictExtractionModel,
    member_answer_json_schema,
    verdict_extraction_json_schema,
    verdict_json_schema,
)
from .verdict_synthesis import (
    VerdictSynthesisResult,
    extract_verdict,
)

__version__ = "1.0.0"

__all__ = [
    "Council",
    "CouncilResult",
    "ModelAnswer",
    "TokenUsage",
    "DebateRound",
    "AdversarialResult",
    "StreamEvent",
    "ConclaveConfig",
    "load_config",
    "aclose",
    "guard_transport_logging",
    # CAC-01 result contract v2 — verdict/member schema public surface.
    "CouncilVerdict",
    "CouncilConflict",
    "CouncilPosition",
    "ProviderVote",
    "MinorityReport",
    "VERDICT_SCHEMA_VERSION",
    "verdict_json_schema",
    "member_answer_json_schema",
    # CAC-05 disagreement extraction + verdict synthesis public surface.
    "extract_verdict",
    "VerdictSynthesisResult",
    "VerdictExtractionModel",
    "verdict_extraction_json_schema",
    "VERDICT_EXTRACTION_PROMPT_VERSION",
    # CAC-04 auditable manifest public surface.
    "ModelHarnessManifest",
    "ProviderExecutionReceipt",
    "VerdictExtraction",
    "ProviderSkip",
    "__version__",
]
