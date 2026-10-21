"""Tests for deterministic descriptive statistics."""

from decimal import Decimal

import pytest

from orvyntal.data_intelligence.analytics.descriptive import (
    describe_numeric_values,
)


def test_describes_basic_numeric_values() -> None:
    result = describe_numeric_values(["1", "2", "3", "4"])

    assert result is not None
    assert result.count == 4
    assert result.invalid_count == 0
    assert result.total == Decimal("10.000000")
    assert result.mean == Decimal("2.500000")
    assert result.median == Decimal("2.500000")
    assert result.minimum == Decimal("1")
    assert result.maximum == Decimal("4")
    assert result.value_range == Decimal("3.000000")
    assert result.population_variance == Decimal("1.250000")
    assert result.population_standard_deviation == Decimal("1.118034")
    assert result.sample_variance == Decimal("1.666667")
    assert result.sample_standard_deviation == Decimal("1.290994")


def test_ignores_missing_and_counts_invalid_values() -> None:
    result = describe_numeric_values(["10", "", " ", "invalid", "NaN", "20"])

    assert result is not None
    assert result.count == 2
    assert result.invalid_count == 2
    assert result.total == Decimal("30.000000")
    assert result.mean == Decimal("15.000000")
    assert result.median == Decimal("15.000000")


@pytest.mark.parametrize(
    "values",
    [[], [""], [" ", "NaN", "Infinity", "oops"]],
)
def test_returns_none_when_no_valid_numbers(values: list[str]) -> None:
    assert describe_numeric_values(values) is None


def test_single_observation_has_no_sample_statistics() -> None:
    result = describe_numeric_values(["7"])

    assert result is not None
    assert result.count == 1
    assert result.total == Decimal("7.000000")
    assert result.population_variance == Decimal("0.000000")
    assert result.population_standard_deviation == Decimal("0.000000")
    assert result.sample_variance is None
    assert result.sample_standard_deviation is None


def test_negative_values_and_zero_are_handled() -> None:
    result = describe_numeric_values(["-2", "0", "2"])

    assert result is not None
    assert result.total == Decimal("0.000000")
    assert result.mean == Decimal("0.000000")
    assert result.median == Decimal("0.000000")
    assert result.value_range == Decimal("4.000000")
    assert result.population_variance == Decimal("2.666667")


def test_even_count_median_is_average_of_middle_values() -> None:
    result = describe_numeric_values(["9", "1", "3", "7"])

    assert result is not None
    assert result.median == Decimal("5.000000")


def test_rounding_is_deterministic_and_half_even() -> None:
    first = describe_numeric_values(["0", "0.000001"])
    second = describe_numeric_values(["0", "0.000001"])

    assert first == second
    assert first is not None
    assert first.mean == Decimal("0.000000")


def test_large_finite_values_do_not_overflow_decimal_context() -> None:
    result = describe_numeric_values(["1e1000", "1e1000"])

    assert result is not None
    assert result.count == 2
    assert result.total == Decimal("2e1000")
    assert result.mean == Decimal("1e1000")
    assert len(result.total.as_tuple().digits) <= 2
    assert result.total.as_tuple().exponent == 1000
    assert result.population_variance == Decimal("0.000000")
    assert result.sample_variance == Decimal("0.000000")


def test_input_is_not_modified() -> None:
    values = ["3", "1", "2"]
    original = values.copy()

    describe_numeric_values(values)

    assert values == original


def test_result_is_immutable() -> None:
    result = describe_numeric_values(["1", "2"])

    assert result is not None
    with pytest.raises(AttributeError):
        result.count = 5
