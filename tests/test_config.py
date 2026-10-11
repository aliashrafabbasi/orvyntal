"""Tests for validated runtime configuration."""

import pytest
from pydantic import ValidationError

from orvyntal.config import Settings


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ORVYNTAL_APP_NAME",
        "ORVYNTAL_ENVIRONMENT",
        "ORVYNTAL_LOG_LEVEL",
        "ORVYNTAL_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.app_name == "ORVYNTAL"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.debug is False


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORVYNTAL_APP_NAME", "Test Service")
    monkeypatch.setenv("ORVYNTAL_ENVIRONMENT", "test")
    monkeypatch.setenv("ORVYNTAL_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ORVYNTAL_DEBUG", "true")

    settings = Settings()

    assert settings.app_name == "Test Service"
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.debug is True


def test_invalid_environment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="invalid")  # type: ignore[arg-type]


def test_debug_mode_is_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="disabled in production"):
        Settings(environment="production", debug=True)
