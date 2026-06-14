"""Tests for the centralized logger factory and CONCLAVE_LOG_LEVEL resolution.

``get_logger`` configures the root ``conclave`` logger exactly once (guarded by a
module ``_CONFIGURED`` flag) and returns either that root or a named child. These
tests prove the level is read from ``CONCLAVE_LOG_LEVEL`` (default ``WARNING``),
that an unrecognized value falls back to ``WARNING``, and that the factory's
root-vs-child contract holds. Each test resets the one-shot configuration state so
the env-var branch actually runs instead of short-circuiting on the import-time
configuration done by ``conclave.transport``.
"""

from __future__ import annotations

import logging

import pytest

import conclave.logging as logging_mod
from conclave.logging import get_logger


def _app_stream_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Return only the handlers conclave itself installed on ``logger``.

    Under pytest's logging plugin the captured loggers also carry one or more
    ``LogCaptureHandler`` instances (a ``StreamHandler`` subclass) injected by the
    harness -- their presence is a property of the test runner, not of conclave's
    one-shot logger config. To assert on conclave's OWN handler set without
    coupling to the pytest version, we exclude any handler whose class name
    contains ``Capture``. What remains is exactly the ``StreamHandler``
    ``get_logger`` adds.
    """
    return [h for h in logger.handlers if "Capture" not in type(h).__name__]


@pytest.fixture
def fresh_logging(monkeypatch):
    """Reset the one-shot logger config so a fresh get_logger() reconfigures.

    Saves and restores ``_CONFIGURED`` plus the root ``conclave`` logger's
    handlers/level/propagate so the global logging state is left exactly as found.
    """
    root = logging.getLogger("conclave")
    saved_configured = logging_mod._CONFIGURED
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_propagate = root.propagate

    # Force reconfiguration on the next get_logger call.
    monkeypatch.setattr(logging_mod, "_CONFIGURED", False)
    root.handlers = []

    yield root

    # Restore prior state.
    root.handlers = saved_handlers
    root.setLevel(saved_level)
    root.propagate = saved_propagate
    logging_mod._CONFIGURED = saved_configured


def test_default_level_is_warning_when_env_unset(fresh_logging, monkeypatch):
    """With CONCLAVE_LOG_LEVEL unset the root logger is configured at WARNING."""
    monkeypatch.delenv("CONCLAVE_LOG_LEVEL", raising=False)

    logger = get_logger()

    assert logger.name == "conclave"
    assert logger.level == logging.WARNING
    assert logger.propagate is False
    app_handlers = _app_stream_handlers(logger)
    assert len(app_handlers) == 1
    assert isinstance(app_handlers[0], logging.StreamHandler)


def test_env_var_sets_level_case_insensitively(fresh_logging, monkeypatch):
    """A lowercase CONCLAVE_LOG_LEVEL is upper-cased and applied (DEBUG)."""
    monkeypatch.setenv("CONCLAVE_LOG_LEVEL", "debug")

    logger = get_logger()

    assert logger.level == logging.DEBUG


def test_unknown_level_falls_back_to_warning(fresh_logging, monkeypatch):
    """An unrecognized level name falls back to WARNING rather than crashing."""
    monkeypatch.setenv("CONCLAVE_LOG_LEVEL", "NOPE")

    logger = get_logger()

    assert logger.level == logging.WARNING


def test_named_logger_is_child_of_root(fresh_logging, monkeypatch):
    """A non-default name returns a child logger that inherits root config."""
    monkeypatch.setenv("CONCLAVE_LOG_LEVEL", "INFO")

    child = get_logger("transport")

    assert child.name == "conclave.transport"
    assert child.parent is logging.getLogger("conclave")
    # Child has no handler of its OWN; it propagates to the configured root.
    # (pytest may attach a capture handler; exclude it -- conclave adds none.)
    assert _app_stream_handlers(child) == []
    # Effective level is inherited from the root we just configured at INFO.
    assert child.getEffectiveLevel() == logging.INFO


def test_configuration_happens_once(fresh_logging, monkeypatch):
    """Repeated calls do not stack handlers -- configuration is one-shot."""
    monkeypatch.setenv("CONCLAVE_LOG_LEVEL", "ERROR")

    first = get_logger()
    assert len(_app_stream_handlers(first)) == 1
    assert logging_mod._CONFIGURED is True

    # Changing the env now must have no effect -- the guard short-circuits.
    monkeypatch.setenv("CONCLAVE_LOG_LEVEL", "DEBUG")
    second = get_logger()

    assert second is first
    assert len(_app_stream_handlers(second)) == 1  # not duplicated
    assert second.level == logging.ERROR  # unchanged from first config
