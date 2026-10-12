"""Runtime initialization and shared application context."""

import logging
from dataclasses import dataclass

from orvyntal.config.settings import Settings
from orvyntal.core.logging import configure_logging


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Validated configuration and initialized runtime services."""

    settings: Settings
    logger: logging.Logger


def bootstrap_runtime(settings: Settings | None = None) -> RuntimeContext:
    """Initialize core runtime services and return their shared context."""
    if settings is None:
        settings = Settings()

    logger = configure_logging(settings)
    logger.info(
        "Runtime initialized: app=%s environment=%s",
        settings.app_name,
        settings.environment,
    )

    return RuntimeContext(settings=settings, logger=logger)
