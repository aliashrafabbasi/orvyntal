"""Tests for the ORVYNTAL FastAPI application."""

import logging

import pytest
from fastapi.testclient import TestClient

from orvyntal import __version__
from orvyntal.app import create_app
from orvyntal.config.settings import Settings


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


@pytest.fixture
def client(isolated_logger):
    app = create_app(Settings(app_name="ORVYNTAL-Test", environment="test"))

    with TestClient(app) as test_client:
        yield test_client


def test_application_metadata(client) -> None:
    assert client.app.title == "ORVYNTAL-Test"
    assert client.app.version == __version__


def test_liveness_endpoint(client) -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint_after_startup(client) -> None:
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"application": "ok"},
    }


def test_readiness_reports_not_initialized(isolated_logger) -> None:
    app = create_app(Settings(environment="test"))

    response = TestClient(app).get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_api_routes_require_versioned_prefix(client) -> None:
    response = client.get("/health/live")

    assert response.status_code == 404


def test_openapi_contains_versioned_health_routes(client) -> None:
    response = client.get("/openapi.json")
    paths = response.json()["paths"]

    assert response.status_code == 200
    assert "/api/v1/health/live" in paths
    assert "/api/v1/health/ready" in paths
