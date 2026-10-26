"""Tests for deterministic Pearson correlation analysis."""

from decimal import Decimal

import pytest

from orvyntal.data_intelligence.analytics.correlation import (
    correlate_numeric_columns,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def make_dataset(
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> CsvDataset:
    """Build a dataset from ordered headers and rows."""
    return CsvDataset(
        headers=headers,
        records=tuple(dict(zip(headers, row, strict=True)) for row in rows),
    )


def test_perfect_positive_correlation() -> None:
    dataset = make_dataset(
        ("x", "y"),
        (("1", "10"), ("2", "20"), ("3", "30")),
    )

    result = correlate_numeric_columns(dataset)

    assert result.pairs[0].correlation == Decimal("1.000000")
    assert result.pairs[0].observation_count == 3


def test_perfect_negative_correlation() -> None:
    dataset = make_dataset(
        ("x", "y"),
        (("1", "30"), ("2", "20"), ("3", "10")),
    )

    result = correlate_numeric_columns(dataset)

    assert result.pairs[0].correlation == Decimal("-1.000000")


def test_zero_linear_correlation() -> None:
    dataset = make_dataset(
        ("x", "y"),
        (("-1", "1"), ("0", "-2"), ("1", "1")),
    )

    result = correlate_numeric_columns(dataset)

    assert result.pairs[0].correlation == Decimal("0.000000")


def test_missing_and_invalid_values_use_pairwise_complete_rows() -> None:
    dataset = make_dataset(
        ("x", "y"),
        (
            ("1", "10"),
            ("2", ""),
            ("3", "30"),
            ("unknown", "40"),
            ("5", "50"),
        ),
    )

    result = correlate_numeric_columns(dataset)

    assert result.pairs[0].observation_count == 3
    assert result.pairs[0].correlation == Decimal("1.000000")


@pytest.mark.parametrize(
    ("rows", "expected_count"),
    [
        ((("1", "10"),), 1),
        ((("1", "10"), ("2", "")), 1),
        ((("1", "10"), ("2", "20e1001")), 1),
        ((("1", "10"), ("", "20")), 1),
    ],
)
def test_insufficient_valid_pairs_return_none(
    rows: tuple[tuple[str, ...], ...],
    expected_count: int,
) -> None:
    dataset = make_dataset(("x", "y"), rows)

    result = correlate_numeric_columns(dataset)

    assert result.pairs[0].observation_count == expected_count
    assert result.pairs[0].correlation is None


def test_constant_column_has_no_correlation() -> None:
    dataset = make_dataset(
        ("constant", "varying"),
        (("7", "1"), ("7", "2"), ("7", "3")),
    )

    result = correlate_numeric_columns(dataset)

    assert result.pairs[0].correlation is None
    assert result.pairs[0].observation_count == 3


def test_extreme_supported_values_do_not_overflow() -> None:
    dataset = make_dataset(
        ("x", "y"),
        (
            ("1e1000", "2e1000"),
            ("2e1000", "4e1000"),
            ("3e1000", "6e1000"),
        ),
    )

    result = correlate_numeric_columns(dataset)

    assert result.pairs[0].correlation == Decimal("1.000000")


def test_supported_exponent_boundaries_are_deterministic() -> None:
    dataset = make_dataset(
        ("x", "y"),
        (
            ("1e-1000", "2e-1000"),
            ("1e1000", "2e1000"),
            ("5e999", "1e1000"),
        ),
    )

    first = correlate_numeric_columns(dataset)
    second = correlate_numeric_columns(dataset)

    assert first == second
    assert len(first.pairs) == 1
    assert first.pairs[0].observation_count == 3
    assert first.pairs[0].correlation is not None
    assert Decimal("-1.000000") <= first.pairs[0].correlation <= Decimal("1.000000")


def test_result_is_deterministic_and_input_is_unchanged() -> None:
    dataset = make_dataset(
        ("x", "y"),
        (("1", "4"), ("2", "1"), ("3", "5"), ("4", "2")),
    )
    original_records = dataset.records

    first = correlate_numeric_columns(dataset)
    second = correlate_numeric_columns(dataset)

    assert first == second
    assert dataset.records == original_records


def test_columns_are_paired_in_header_order() -> None:
    dataset = make_dataset(
        ("z", "a", "label"),
        (("1", "3", "x"), ("2", "2", "y"), ("3", "1", "z")),
    )

    result = correlate_numeric_columns(dataset)

    assert [(pair.column_x, pair.column_y) for pair in result.pairs] == [("z", "a")]
