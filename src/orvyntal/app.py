"""FastAPI application factory and lifecycle management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orvyntal import __version__
from orvyntal.api.routes.health import router as health_router
from orvyntal.config.settings import Settings
from orvyntal.core.runtime import bootstrap_runtime


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independently configurable ORVYNTAL API application."""
    app_settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = bootstrap_runtime(app_settings)
        application.state.runtime = runtime

        try:
            yield
        finally:
            runtime.logger.info("Runtime shutdown complete")
            del application.state.runtime

    application = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        debug=app_settings.debug,
        lifespan=lifespan,
    )
    application.include_router(health_router, prefix="/api/v1")

    return application
