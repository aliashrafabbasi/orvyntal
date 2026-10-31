"""Tests for deterministic CSV dataset profiling."""

from decimal import Decimal
from fractions import Fraction

import pytest

from orvyntal.data_intelligence.analytics.profiling import (
    _round_fraction,
    profile_dataset,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def test_profiles_types_missing_values_and_numeric_summaries() -> None:
    dataset = CsvDataset(
        headers=("age", "score", "active", "name"),
        records=(
            {"age": "20", "score": "1.25", "active": "yes", "name": "Ali"},
            {"age": "30", "score": "2.75", "active": "no", "name": ""},
            {"age": "40", "score": "3.00", "active": "true", "name": "Sara"},
            {"age": " ", "score": "4.00", "active": "false", "name": "Zara"},
        ),
    )

    profile = profile_dataset(dataset)

    assert (profile.row_count, profile.column_count) == (4, 4)
    assert profile.duplicate_row_count == 0
    age, score, active, name = profile.columns
    assert age.inferred_type == "integer"
    assert age.numeric_summary is not None
    assert age.numeric_summary.mean == Decimal("30.000000")
    assert age.numeric_summary.median == Decimal("30.000000")
    assert age.numeric_summary.minimum == Decimal("20")
    assert age.numeric_summary.maximum == Decimal("40")
    assert age.missing_count == 1
    assert age.missing_percentage == Decimal("25.000000")
    assert score.inferred_type == "numeric"
    assert score.numeric_summary is not None
    assert score.numeric_summary.count == 4
    assert score.numeric_summary.mean == Decimal("2.750000")
    assert score.numeric_summary.median == Decimal("2.875000")
    assert active.inferred_type == "boolean"
    assert active.numeric_summary is None
    assert name.inferred_type == "text"
    assert name.missing_count == 1


def test_mixed_column_reports_numeric_and_invalid_values_separately() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=(
            {"value": "12"},
            {"value": "15"},
            {"value": "unknown"},
            {"value": "NaN"},
            {"value": ""},
        ),
    )

    column = profile_dataset(dataset).columns[0]

    assert column.inferred_type == "text"
    assert column.missing_count == 1
    assert column.numeric_summary is not None
    assert column.numeric_summary.count == 2
    assert column.numeric_summary.invalid_count == 2
    assert column.numeric_summary.mean == Decimal("13.500000")
    assert column.numeric_summary.median == Decimal("13.500000")
    assert column.numeric_summary.minimum == Decimal("12")
    assert column.numeric_summary.maximum == Decimal("15")


def test_duplicate_rows_count_only_repeated_occurrences() -> None:
    dataset = CsvDataset(
        headers=("name", "score"),
        records=(
            {"name": "Ali", "score": "90"},
            {"name": "Sara", "score": "85"},
            {"name": "Ali", "score": "90"},
            {"name": "Ali", "score": "90"},
        ),
    )

    assert profile_dataset(dataset).duplicate_row_count == 2


def test_header_only_dataset_has_zero_counts_and_no_numeric_summary() -> None:
    profile = profile_dataset(CsvDataset(headers=("age",), records=()))

    assert profile.row_count == 0
    assert profile.column_count == 1
    assert profile.duplicate_row_count == 0
    assert profile.columns[0].inferred_type == "text"
    assert profile.columns[0].missing_count == 0
    assert profile.columns[0].missing_percentage == Decimal("0.000000")
    assert profile.columns[0].numeric_summary is None


def test_all_missing_column_has_no_numeric_summary() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": ""}, {"value": "   "}),
    )

    column = profile_dataset(dataset).columns[0]

    assert column.inferred_type == "text"
    assert column.missing_count == 2
    assert column.missing_percentage == Decimal("100.000000")
    assert column.numeric_summary is None


@pytest.mark.parametrize(
    ("values", "expected_type"),
    [
        (("1", "2", "3"), "integer"),
        (("1.5", "2", "3.25"), "numeric"),
        (("true", "FALSE", "yes"), "boolean"),
        (("1", "two"), "text"),
        (("NaN", "Infinity"), "text"),
    ],
)
def test_type_inference_is_conservative(
    values: tuple[str, ...],
    expected_type: str,
) -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=tuple({"value": value} for value in values),
    )

    assert profile_dataset(dataset).columns[0].inferred_type == expected_type


def test_profiling_does_not_mutate_dataset_records() -> None:
    records = ({"value": "1"}, {"value": "not-a-number"})
    dataset = CsvDataset(headers=("value",), records=records)

    profile_dataset(dataset)

    assert dataset.records == records


def test_profiling_is_deterministic() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": "1"}, {"value": "2"}),
    )

    assert profile_dataset(dataset) == profile_dataset(dataset)


def test_large_finite_numbers_are_profiled_without_decimal_overflow() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": "1e1000"}, {"value": "2"}),
    )

    summary = profile_dataset(dataset).columns[0].numeric_summary

    assert summary is not None
    assert summary.count == 2
    from fractions import Fraction

    expected = (Fraction(Decimal("1e1000")) + Fraction(Decimal("2"))) / 2
    assert Fraction(summary.mean) == expected
    assert Fraction(summary.median) == expected
    assert summary.mean.is_finite()
    assert summary.median.is_finite()
    assert summary.minimum == Decimal("2")
    assert summary.maximum == Decimal("1e1000")


def test_tiny_finite_numbers_follow_six_decimal_rounding_policy() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": "1e-1000"},),
    )

    summary = profile_dataset(dataset).columns[0].numeric_summary

    assert summary is not None
    assert summary.mean == Decimal("0.000000")
    assert summary.median == Decimal("0.000000")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(1, 2_000_000), Decimal("0.000000")),
        (Fraction(3, 2_000_000), Decimal("0.000002")),
        (Fraction(-3, 2_000_000), Decimal("-0.000002")),
    ],
)
def test_fraction_rounding_uses_half_even(
    value: Fraction,
    expected: Decimal,
) -> None:
    assert _round_fraction(value) == expected


def test_missing_percentage_uses_six_decimal_precision() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": ""}, {"value": "1"}, {"value": "2"}),
    )

    column = profile_dataset(dataset).columns[0]

    assert column.missing_percentage == Decimal("33.333333")


def test_text_column_reports_category_frequencies_and_percentages() -> None:
    dataset = CsvDataset(
        headers=("name",),
        records=(
            {"name": "Ali"},
            {"name": "Sara"},
            {"name": "Ali"},
            {"name": ""},
            {"name": "Sara"},
            {"name": "Ali"},
        ),
    )

    column = profile_dataset(dataset).columns[0]

    assert column.inferred_type == "text"
    assert column.missing_count == 1
    assert column.categorical_summary is not None
    assert column.categorical_summary.unique_count == 2
    assert [
        (item.value, item.count, item.percentage)
        for item in column.categorical_summary.top_categories
    ] == [
        ("Ali", 3, Decimal("60.000000")),
        ("Sara", 2, Decimal("40.000000")),
    ]


def test_category_frequency_ties_use_deterministic_value_order() -> None:
    dataset = CsvDataset(
        headers=("name",),
        records=(
            {"name": "Zara"},
            {"name": "Ali"},
            {"name": "Sara"},
            {"name": "Ali"},
            {"name": "Zara"},
            {"name": "Sara"},
        ),
    )

    summary = profile_dataset(dataset).columns[0].categorical_summary

    assert summary is not None
    assert [item.value for item in summary.top_categories] == [
        "Ali",
        "Sara",
        "Zara",
    ]
    assert all(
        item.percentage == Decimal("33.333333") for item in summary.top_categories
    )


def test_boolean_column_reports_category_frequencies() -> None:
    dataset = CsvDataset(
        headers=("active",),
        records=(
            {"active": "yes"},
            {"active": "no"},
            {"active": "yes"},
            {"active": ""},
        ),
    )

    column = profile_dataset(dataset).columns[0]

    assert column.inferred_type == "boolean"
    assert column.categorical_summary is not None
    assert [
        (item.value, item.count) for item in column.categorical_summary.top_categories
    ] == [("yes", 2), ("no", 1)]


def test_numeric_column_does_not_report_category_frequencies() -> None:
    dataset = CsvDataset(
        headers=("score",),
        records=({"score": "10"}, {"score": "20"}, {"score": "10"}),
    )

    column = profile_dataset(dataset).columns[0]

    assert column.inferred_type == "integer"
    assert column.numeric_summary is not None
    assert column.categorical_summary is None


def test_high_cardinality_column_returns_only_top_twenty_categories() -> None:
    dataset = CsvDataset(
        headers=("label",),
        records=tuple({"label": f"category-{i:02d}"} for i in range(25)),
    )

    summary = profile_dataset(dataset).columns[0].categorical_summary

    assert summary is not None
    assert summary.unique_count == 25
    assert len(summary.top_categories) == 20
    assert [item.value for item in summary.top_categories] == [
        f"category-{i:02d}" for i in range(20)
    ]


def test_all_missing_text_column_has_empty_category_summary() -> None:
    dataset = CsvDataset(
        headers=("label",),
        records=({"label": ""}, {"label": "  "}),
    )

    column = profile_dataset(dataset).columns[0]

    assert column.categorical_summary is not None
    assert column.categorical_summary.unique_count == 0
    assert column.categorical_summary.top_categories == ()
