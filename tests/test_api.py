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


def test_dataset_upload_returns_metadata_and_limited_preview(client) -> None:
    csv_content = (
        "name,score\nAli,90\nSara,85\nHamza,80\nNoor,95\nZara,88\nAhmed,76\nAyesha,91\n"
    )
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("students.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "headers": ["name", "score"],
        "row_count": 7,
        "preview": [
            {"name": "Ali", "score": "90"},
            {"name": "Sara", "score": "85"},
            {"name": "Hamza", "score": "80"},
            {"name": "Noor", "score": "95"},
            {"name": "Zara", "score": "88"},
        ],
    }


def test_dataset_upload_can_include_analytics(client) -> None:
    csv_content = "name,score\nAli,90\nSara,85\nAli,90\n"
    response = client.post(
        "/api/v1/datasets?include_analytics=true",
        files={"file": ("students.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["headers"] == ["name", "score"]
    assert body["row_count"] == 3
    assert body["analytics"]["column_count"] == 2
    assert body["analytics"]["duplicate_row_count"] == 1

    score = body["analytics"]["columns"][1]
    assert score["name"] == "score"
    assert score["inferred_type"] == "integer"
    assert score["numeric_statistics"]["count"] == 3
    assert score["numeric_statistics"]["total"] == "265"
    assert score["numeric_statistics"]["mean"] == "88.333333"
    assert score["numeric_statistics"]["minimum"] == "85"
    assert score["numeric_statistics"]["maximum"] == "90"


def test_dataset_upload_analytics_counts_missing_and_invalid_values(client) -> None:
    csv_content = "score,tag\n10,a\n,b\ninvalid,c\n"
    response = client.post(
        "/api/v1/datasets?include_analytics=true",
        files={"file": ("scores.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    analytics = response.json()["analytics"]
    score = analytics["columns"][0]

    assert score["missing_count"] == 1
    assert score["numeric_statistics"]["count"] == 1
    assert score["numeric_statistics"]["invalid_count"] == 1
    assert score["numeric_statistics"]["sample_variance"] is None


def test_dataset_upload_omits_analytics_by_default(client) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("scores.csv", "score\n10\n", "text/csv")},
    )

    assert response.status_code == 200
    assert "analytics" not in response.json()


def test_dataset_upload_header_only_csv_includes_empty_analytics(client) -> None:
    response = client.post(
        "/api/v1/datasets?include_analytics=true",
        files={"file": ("headers.csv", "amount,name\n", "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["headers"] == ["amount", "name"]
    assert body["row_count"] == 0
    assert body["preview"] == []
    assert body["analytics"] == {
        "column_count": 2,
        "duplicate_row_count": 0,
        "columns": [
            {
                "name": "amount",
                "inferred_type": "text",
                "missing_count": 0,
                "missing_percentage": "0.000000",
                "numeric_statistics": None,
            },
            {
                "name": "name",
                "inferred_type": "text",
                "missing_count": 0,
                "missing_percentage": "0.000000",
                "numeric_statistics": None,
            },
        ],
    }


def test_dataset_upload_mixed_values_reports_valid_and_invalid_numbers(
    client,
) -> None:
    csv_content = "amount\n12\n15\nunknown\nNaN\n"
    response = client.post(
        "/api/v1/datasets?include_analytics=true",
        files={"file": ("mixed.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    column = response.json()["analytics"]["columns"][0]
    assert column["inferred_type"] == "text"
    assert column["missing_count"] == 0
    assert column["numeric_statistics"] == {
        "count": 2,
        "invalid_count": 2,
        "total": "27",
        "mean": "13.5",
        "median": "13.5",
        "minimum": "12",
        "maximum": "15",
        "value_range": "3",
        "population_variance": "2.25",
        "population_standard_deviation": "1.5",
        "sample_variance": "4.5",
        "sample_standard_deviation": "2.12132",
    }


def test_dataset_upload_analytics_handles_out_of_policy_numbers(client) -> None:
    values = [
        "1e10000",
        "1e-1001",
        "1." + ("0" * 1000) + "1",
    ]

    for value in values:
        response = client.post(
            "/api/v1/datasets?include_analytics=true",
            files={
                "file": (
                    "numbers.csv",
                    "\n".join(("amount", "2", value, "")),
                    "text/csv",
                )
            },
        )

        assert response.status_code == 200
        column = response.json()["analytics"]["columns"][0]
        assert column["numeric_statistics"]["count"] == 1
        assert column["numeric_statistics"]["invalid_count"] == 1
        assert column["numeric_statistics"]["total"] == "2"


def test_dataset_upload_accepts_file_at_exact_size_limit(isolated_logger) -> None:
    csv_content = "score\n10\n"
    app = create_app(
        Settings(
            environment="test",
            max_upload_size_bytes=len(csv_content.encode("utf-8")),
        )
    )

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/datasets",
            files={"file": ("scores.csv", csv_content, "text/csv")},
        )

    assert response.status_code == 200
    assert response.json()["row_count"] == 1


def test_dataset_upload_rejects_unsupported_extension(client) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("students.txt", "name\nAli\n", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_dataset_upload_rejects_malformed_csv(client) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("broken.csv", 'name,score\n"Ali,90\n', "text/csv")},
    )

    assert response.status_code == 422


def test_dataset_upload_rejects_empty_csv(client) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("empty.csv", "", "text/csv")},
    )

    assert response.status_code == 422


def test_dataset_upload_enforces_configured_size_limit(isolated_logger) -> None:
    app = create_app(Settings(environment="test", max_upload_size_bytes=8))

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/datasets",
            files={"file": ("large.csv", "name\nAli,90\n", "text/csv")},
        )

    assert response.status_code == 413


def test_openapi_documents_dataset_upload_responses(client) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/api/v1/datasets"]["post"]
    assert {"200", "413", "415", "422"} <= set(operation["responses"])
    assert (
        operation["responses"]["413"]["description"]
        == "Uploaded file exceeds the configured size limit."
    )
    assert (
        operation["responses"]["415"]["description"]
        == "Unsupported file type. Upload a .csv file."
    )
    assert (
        operation["responses"]["422"]["description"]
        == "Request validation failed or the CSV is invalid."
    )


def test_dataset_upload_analytics_bounds_extreme_standard_deviation_output(
    client,
) -> None:
    response = client.post(
        "/api/v1/datasets?include_analytics=true",
        files={
            "file": (
                "extreme.csv",
                "amount\n-1e1000\n1e1000\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    stats = response.json()["analytics"]["columns"][0]["numeric_statistics"]

    assert stats["population_standard_deviation"] == "1E+1000"
    assert stats["sample_standard_deviation"] == "1.414214E+1000"
    assert len(stats["sample_standard_deviation"]) <= 32


def test_dataset_upload_rejects_rows_over_configured_limit(isolated_logger) -> None:
    app = create_app(Settings(environment="test", max_dataset_rows=1))

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/datasets",
            files={"file": ("rows.csv", "value\na\nb\n", "text/csv")},
        )

    assert response.status_code == 422
    assert "maximum of 1 data row" in response.json()["detail"]


def test_dataset_upload_rejects_columns_over_configured_limit(
    isolated_logger,
) -> None:
    app = create_app(Settings(environment="test", max_dataset_columns=1))

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/datasets",
            files={"file": ("columns.csv", "a,b\n1,2\n", "text/csv")},
        )

    assert response.status_code == 422
    assert "maximum of 1 column" in response.json()["detail"]
