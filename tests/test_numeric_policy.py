"""Tests for the shared analytics numeric safety policy."""

import pytest

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value


@pytest.mark.parametrize(
    "value",
    ["1e1000", "1e-1000", "0", "-0", "1.25"],
)
def test_accepts_supported_numeric_values(value: str) -> None:
    assert parse_numeric_value(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "1e1001",
        "1e-1001",
        "0e1001",
        "0e-1001",
        "-0e1001",
        "NaN",
        "Infinity",
        "1." + ("0" * 1000) + "1",
    ],
)
def test_rejects_values_outside_numeric_policy(value: str) -> None:
    assert parse_numeric_value(value) is None
