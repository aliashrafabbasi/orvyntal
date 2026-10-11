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

    @model_validator(mode="after")
    def validate_production_debug(self) -> "Settings":
        """Prevent debug mode from being enabled in production."""
        if self.environment == "production" and self.debug:
            raise ValueError("Debug mode must be disabled in production.")
        return self
