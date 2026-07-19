"""Experimental, offline contracts for budget-matched Conclave studies.

This package is intentionally not part of the public council API.  Its schemas
are versioned so recorded studies can be rejected when their contract changes.
"""

from .live import (
    BudgetExceededError,
    GatewayStoppedError,
    LiveProviderClient,
    PendingCall,
    ProviderCallCostBasis,
    ProviderCallReceipt,
    ReservationBreachError,
)
from .models import (
    EVAL_SCHEMA_VERSION,
    ConditionSpec,
    GraderKey,
    PlannedRun,
    ProtocolExecution,
    PublicTask,
    RunRecord,
    ScoreRecord,
    StudyManifest,
    StudyRun,
)
from .scoring import AdjudicationRecord, GraderJudgment, StudyScoreReport

__all__ = [
    "EVAL_SCHEMA_VERSION",
    "ConditionSpec",
    "GraderKey",
    "PlannedRun",
    "ProtocolExecution",
    "PublicTask",
    "RunRecord",
    "ScoreRecord",
    "StudyManifest",
    "StudyRun",
    "BudgetExceededError",
    "GatewayStoppedError",
    "LiveProviderClient",
    "PendingCall",
    "ProviderCallCostBasis",
    "ProviderCallReceipt",
    "ReservationBreachError",
    "AdjudicationRecord",
    "GraderJudgment",
    "StudyScoreReport",
]
