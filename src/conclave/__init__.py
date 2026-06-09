"""conclave -- bring-your-own-keys multi-model council.

Public library API::

    from conclave import Council
    council = Council(models=["grok", "perplexity"], synthesizer="claude")

    # synthesize (default) / raw
    result = council.ask_sync("Your prompt")            # sync
    result = await council.ask("Your prompt")            # async

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
from .models import (
    AdversarialResult,
    CouncilResult,
    DebateRound,
    ModelAnswer,
    TokenUsage,
)
from .transport import aclose

__version__ = "0.1.0"

__all__ = [
    "Council",
    "CouncilResult",
    "ModelAnswer",
    "TokenUsage",
    "DebateRound",
    "AdversarialResult",
    "ConclaveConfig",
    "load_config",
    "aclose",
    "__version__",
]
