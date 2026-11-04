"""Validated runtime configuration for ORVYNTAL."""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ORVYNTAL_",
        case_sensitive=False,
        str_strip_whitespace=True,
        extra="ignore",
    )

    app_name: str = Field(default="ORVYNTAL", min_length=1, max_length=100)
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    debug: bool = False
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_dataset_rows: int = Field(default=10_000, gt=0)
    max_dataset_columns: int = Field(default=100, gt=0)
    max_concurrent_analytics_jobs: int = Field(default=1, gt=0)
    analytics_timeout_seconds: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def validate_production_debug(self) -> "Settings":
        """Prevent debug mode from being enabled in production."""
        if self.environment == "production" and self.debug:
            raise ValueError("Debug mode must be disabled in production.")
        return self
