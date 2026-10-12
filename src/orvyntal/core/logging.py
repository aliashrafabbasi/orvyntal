"""Centralized logging setup for ORVYNTAL."""

import logging

from orvyntal.config.settings import Settings

_LOGGER_NAME = "orvyntal"
_HANDLER_MARKER = "_orvyntal_managed_handler"


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure the ORVYNTAL logger without adding duplicate handlers."""
    if settings is None:
        settings = Settings()

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(settings.log_level)
    logger.propagate = False

    handler = next(
        (item for item in logger.handlers if getattr(item, _HANDLER_MARKER, False)),
        None,
    )

    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    handler.setLevel(settings.log_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    return logger
