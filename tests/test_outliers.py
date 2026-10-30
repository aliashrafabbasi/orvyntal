"""Tests for deterministic Tukey IQR outlier analysis."""

from decimal import Decimal
from fractions import Fraction

from orvyntal.data_intelligence.analytics.outliers import (
    analyze_numeric_outliers,
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


def test_detects_high_and_low_outliers_with_original_row_indices() -> None:
    dataset = make_dataset(
        ("value",),
        (("1",), ("2",), ("3",), ("4",), ("5",), ("100",), ("-100",)),
    )

    result = analyze_numeric_outliers(dataset).columns[0]

    assert result.q1 == Fraction(1)
    assert result.q3 == Fraction(5)
    assert result.iqr == Fraction(4)
    assert result.lower_bound == Fraction(-5)
    assert result.upper_bound == Fraction(11)
    assert [(item.row_index, item.value) for item in result.outliers] == [
        (5, Decimal("100")),
        (6, Decimal("-100")),
    ]


def test_values_on_iqr_bounds_are_not_outliers() -> None:
    dataset = make_dataset(
        ("value",),
        (("-1",), ("0",), ("1",), ("1",), ("2",), ("5",)),
    )

    result = analyze_numeric_outliers(dataset).columns[0]

    assert result.outliers == ()


def test_missing_and_invalid_values_are_excluded() -> None:
    dataset = make_dataset(
        ("value",),
        (
            ("1",),
            ("2",),
            ("3",),
            ("4",),
            ("5",),
            ("100",),
            ("",),
            ("unknown",),
        ),
    )

    result = analyze_numeric_outliers(dataset).columns[0]

    assert result.observation_count == 6
    assert result.missing_count == 1
    assert result.invalid_count == 1
    assert [(item.row_index, item.value) for item in result.outliers] == [
        (5, Decimal("100"))
    ]


def test_columns_preserve_header_order_and_non_numeric_columns() -> None:
    dataset = make_dataset(
        ("label", "amount"),
        (("a", "1"), ("b", "2"), ("c", "3")),
    )

    result = analyze_numeric_outliers(dataset)

    assert [column.column_name for column in result.columns] == [
        "label",
        "amount",
    ]
    assert result.columns[0].observation_count == 0
    assert result.columns[0].q1 is None
    assert result.columns[1].q1 == Fraction(1)
    assert result.columns[1].q3 == Fraction(3)


def test_fewer_than_two_valid_observations_have_no_bounds() -> None:
    dataset = make_dataset(
        ("value",),
        (("1",), ("",), ("not-a-number",)),
    )

    result = analyze_numeric_outliers(dataset).columns[0]

    assert result.observation_count == 1
    assert result.missing_count == 1
    assert result.invalid_count == 1
    assert result.q1 is None
    assert result.q3 is None
    assert result.iqr is None
    assert result.lower_bound is None
    assert result.upper_bound is None
    assert result.outliers == ()


def test_extreme_supported_values_do_not_overflow() -> None:
    dataset = make_dataset(
        ("value",),
        (
            ("1e-1000",),
            ("2e-1000",),
            ("3e-1000",),
            ("4e-1000",),
            ("5e-1000",),
            ("1e1000",),
        ),
    )

    result = analyze_numeric_outliers(dataset).columns[0]

    assert [(item.row_index, item.value) for item in result.outliers] == [
        (5, Decimal("1e1000"))
    ]


def test_analysis_is_deterministic_and_does_not_mutate_input() -> None:
    dataset = make_dataset(
        ("value",),
        (("1",), ("2",), ("3",), ("4",), ("5",), ("100",)),
    )
    original_records = dataset.records

    first = analyze_numeric_outliers(dataset)
    second = analyze_numeric_outliers(dataset)

    assert first == second
    assert dataset.records == original_records


def test_constant_values_have_zero_iqr_and_no_outliers() -> None:
    dataset = make_dataset(
        ("value",),
        (("7",), ("7",), ("7",), ("7",)),
    )

    result = analyze_numeric_outliers(dataset).columns[0]

    assert result.iqr == 0
    assert result.lower_bound == 7
    assert result.upper_bound == 7
    assert result.outliers == ()


def test_result_objects_are_immutable() -> None:
    dataset = make_dataset(
        ("value",),
        (("1",), ("2",), ("3",), ("4",)),
    )

    result = analyze_numeric_outliers(dataset)

    try:
        result.columns[0].column_name = "changed"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("Expected immutable analysis result")
