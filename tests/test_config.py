"""Tests for validated runtime configuration."""

import pytest
from pydantic import ValidationError

from orvyntal.config.settings import Settings


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ORVYNTAL_APP_NAME",
        "ORVYNTAL_ENVIRONMENT",
        "ORVYNTAL_LOG_LEVEL",
        "ORVYNTAL_DEBUG",
        "ORVYNTAL_MAX_DATASET_ROWS",
        "ORVYNTAL_MAX_DATASET_COLUMNS",
        "ORVYNTAL_MAX_CONCURRENT_ANALYTICS_JOBS",
        "ORVYNTAL_ANALYTICS_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.app_name == "ORVYNTAL"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.debug is False
    assert settings.max_dataset_rows == 10_000
    assert settings.max_dataset_columns == 100
    assert settings.max_concurrent_analytics_jobs == 1
    assert settings.analytics_timeout_seconds == 60


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORVYNTAL_APP_NAME", "Test Service")
    monkeypatch.setenv("ORVYNTAL_ENVIRONMENT", "test")
    monkeypatch.setenv("ORVYNTAL_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ORVYNTAL_DEBUG", "true")
    monkeypatch.setenv("ORVYNTAL_MAX_DATASET_ROWS", "250")
    monkeypatch.setenv("ORVYNTAL_MAX_DATASET_COLUMNS", "12")
    monkeypatch.setenv("ORVYNTAL_MAX_CONCURRENT_ANALYTICS_JOBS", "3")
    monkeypatch.setenv("ORVYNTAL_ANALYTICS_TIMEOUT_SECONDS", "45")

    settings = Settings()

    assert settings.app_name == "Test Service"
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.debug is True
    assert settings.max_dataset_rows == 250
    assert settings.max_dataset_columns == 12
    assert settings.max_concurrent_analytics_jobs == 3
    assert settings.analytics_timeout_seconds == 45


def test_invalid_environment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="invalid")  # type: ignore[arg-type]


def test_debug_mode_is_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="disabled in production"):
        Settings(environment="production", debug=True)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_concurrent_analytics_jobs", 0),
        ("max_concurrent_analytics_jobs", -1),
        ("analytics_timeout_seconds", 0),
        ("analytics_timeout_seconds", -1),
    ],
)
def test_invalid_analytics_settings_are_rejected(
    setting: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{setting: value})
