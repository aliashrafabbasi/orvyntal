"""Tests for centralized ORVYNTAL logging."""

import logging

import pytest

from orvyntal.config import Settings
from orvyntal.logging import configure_logging


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


def test_configure_logging_applies_configured_level(isolated_logger) -> None:
    logger = configure_logging(Settings(log_level="DEBUG"))

    assert logger.level == logging.DEBUG
    assert logger.handlers[0].level == logging.DEBUG


def test_configure_logging_does_not_duplicate_handlers(
    isolated_logger,
) -> None:
    configure_logging(Settings())
    initial_count = len(isolated_logger.handlers)

    configure_logging(Settings())

    assert len(isolated_logger.handlers) == initial_count == 1


def test_log_output_contains_timestamp_level_and_message(
    isolated_logger, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = configure_logging(Settings(log_level="INFO"))

    logger.info("logging cluster test")

    output = capsys.readouterr().err
    assert "INFO" in output
    assert "orvyntal" in output
    assert "logging cluster test" in output
    assert "|" in output


def test_reconfiguration_updates_existing_handler(isolated_logger) -> None:
    logger = configure_logging(Settings(log_level="INFO"))
    handler = logger.handlers[0]

    configure_logging(Settings(log_level="ERROR"))

    assert logger.level == logging.ERROR
    assert logger.handlers[0] is handler
    assert handler.level == logging.ERROR
