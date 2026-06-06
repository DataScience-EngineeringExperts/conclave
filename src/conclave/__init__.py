"""conclave -- bring-your-own-keys multi-model council.

Public library API::

    from conclave import Council
    council = Council(models=["grok", "perplexity"], synthesizer="claude")
    result = council.ask_sync("Your prompt")          # sync
    result = await council.ask("Your prompt")          # async

The returned :class:`CouncilResult` carries each member's raw answer (with
latency, token usage, and any error) plus the merged synthesis.
"""

from __future__ import annotations

from .config import ConclaveConfig, load_config
from .council import Council
from .models import CouncilResult, ModelAnswer, TokenUsage

__version__ = "0.1.0"

__all__ = [
    "Council",
    "CouncilResult",
    "ModelAnswer",
    "TokenUsage",
    "ConclaveConfig",
    "load_config",
    "__version__",
]
