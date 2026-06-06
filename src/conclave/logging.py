"""Centralized logger configuration for conclave.

Provides a single module logger factory so every provider call path can log
consistently. Verbosity is controllable via the CONCLAVE_LOG_LEVEL env var.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "conclave") -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name; child loggers inherit the root ``conclave`` config.

    Returns:
        A ``logging.Logger`` writing to stderr at the level given by the
        ``CONCLAVE_LOG_LEVEL`` env var (default ``WARNING``).
    """
    global _CONFIGURED
    root = logging.getLogger("conclave")
    if not _CONFIGURED:
        level_name = os.environ.get("CONCLAVE_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_name, logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return root if name == "conclave" else root.getChild(name)
