"""Tests for ORVYNTAL runtime initialization."""

import logging

import pytest

from orvyntal.config import Settings
from orvyntal.runtime import RuntimeContext, bootstrap_runtime


@pytest.fixture
def isolated_logger():
    logger = logging.getLogger("orvyntal")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate

    for handler in original_handlers:
        logger.removeHandler(handler)

    yield logger

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    for handler in original_handlers:
        logger.addHandler(handler)

    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_bootstrap_returns_context_with_supplied_settings(
    isolated_logger,
) -> None:
    settings = Settings(app_name="ORVYNTAL-Test", environment="test")

    context = bootstrap_runtime(settings)

    assert isinstance(context, RuntimeContext)
    assert context.settings is settings
    assert context.logger is isolated_logger


def test_bootstrap_uses_default_settings(isolated_logger, monkeypatch) -> None:
    monkeypatch.setenv("ORVYNTAL_ENVIRONMENT", "test")

    context = bootstrap_runtime()

    assert context.settings.environment == "test"
    assert context.logger is isolated_logger


def test_bootstrap_logs_successful_initialization(
    isolated_logger, capsys: pytest.CaptureFixture[str]
) -> None:
    bootstrap_runtime(Settings(app_name="ORVYNTAL-Test", environment="test"))

    output = capsys.readouterr().err
    assert "Runtime initialized" in output
    assert "ORVYNTAL-Test" in output
    assert "environment=test" in output


def test_runtime_context_is_immutable(isolated_logger) -> None:
    context = bootstrap_runtime(Settings(environment="test"))

    with pytest.raises(AttributeError):
        context.settings = Settings()  # type: ignore[misc]
