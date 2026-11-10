"""Tests for deterministic, non-destructive dataset cleaning."""

from decimal import Decimal

import pytest

from orvyntal.data_intelligence.cleaning import (
    CleaningResult,
    ColumnCleaningPolicy,
    DataCleaningError,
    clean_dataset,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def test_constant_imputation_and_audit_report() -> None:
    dataset = CsvDataset(
        headers=("name", "city"),
        records=(
            {"name": "Ali", "city": ""},
            {"name": "Sara", "city": "Karachi"},
            {"name": "Zara", "city": " "},
        ),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "city": ColumnCleaningPolicy("constant", "Unknown"),
        },
    )

    assert result.dataset.records == (
        {"name": "Ali", "city": "Unknown"},
        {"name": "Sara", "city": "Karachi"},
        {"name": "Zara", "city": "Unknown"},
    )
    assert result.report.filled_values == (("city", 2),)
    assert result.report.output_row_count == 3


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("mean", "2.6666666666666666666666666666666666666666666666667"),
        ("median", "2"),
        ("mode", "2"),
    ],
)
def test_imputation_strategies(
    strategy: str,
    expected: str,
) -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=(
            {"value": "2"},
            {"value": ""},
            {"value": "4"},
            {"value": "2"},
        ),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "value": ColumnCleaningPolicy(strategy),  # type: ignore[arg-type]
        },
    )

    assert result.dataset.records[1]["value"] == expected
    assert result.report.filled_values == (("value", 1),)


def test_drop_strategy_removes_rows_with_original_indexes() -> None:
    dataset = CsvDataset(
        headers=("name", "age"),
        records=(
            {"name": "Ali", "age": "20"},
            {"name": "", "age": "25"},
            {"name": "Sara", "age": "30"},
            {"name": "Zara", "age": ""},
        ),
    )

    result = clean_dataset(
        dataset,
        column_policies={"name": ColumnCleaningPolicy("drop")},
    )

    assert result.dataset.records == (
        {"name": "Ali", "age": "20"},
        {"name": "Sara", "age": "30"},
        {"name": "Zara", "age": ""},
    )
    assert result.report.removed_row_indices == (1,)
    assert result.report.missing_rows_removed == 1


def test_duplicate_removal_keeps_first_occurrence() -> None:
    dataset = CsvDataset(
        headers=("name", "score"),
        records=(
            {"name": "Ali", "score": "90"},
            {"name": "Sara", "score": "80"},
            {"name": "Ali", "score": "90"},
            {"name": "Ali", "score": "90"},
        ),
    )

    result = clean_dataset(dataset, remove_duplicates=True)

    assert result.dataset.records == (
        {"name": "Ali", "score": "90"},
        {"name": "Sara", "score": "80"},
    )
    assert result.report.duplicate_row_indices == (2, 3)
    assert result.report.duplicate_rows_removed == 2


def test_cleaning_applies_imputation_before_duplicate_removal() -> None:
    dataset = CsvDataset(
        headers=("name",),
        records=(
            {"name": "Ali"},
            {"name": ""},
        ),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "name": ColumnCleaningPolicy("constant", "Ali"),
        },
        remove_duplicates=True,
    )

    assert result.dataset.records == ({"name": "Ali"},)
    assert result.report.filled_values == (("name", 1),)
    assert result.report.duplicate_row_indices == (1,)


def test_mode_tie_breaking_is_deterministic() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=(
            {"value": "z"},
            {"value": "a"},
            {"value": ""},
        ),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "value": ColumnCleaningPolicy("mode"),
        },
    )

    assert result.dataset.records[-1]["value"] == "a"


def test_numeric_imputation_uses_decimal_arithmetic() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=(
            {"value": "0.1"},
            {"value": ""},
            {"value": "0.2"},
        ),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "value": ColumnCleaningPolicy("mean"),
        },
    )

    assert Decimal(result.dataset.records[1]["value"]) == Decimal("0.15")


def test_numeric_imputation_rejects_invalid_observed_values() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": "10"}, {"value": "invalid"}, {"value": ""}),
    )

    with pytest.raises(DataCleaningError, match="valid finite numbers"):
        clean_dataset(
            dataset,
            column_policies={
                "value": ColumnCleaningPolicy("mean"),
            },
        )


def test_imputation_rejects_columns_without_observed_values() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": ""}, {"value": "  "}),
    )

    with pytest.raises(DataCleaningError, match="no observed values"):
        clean_dataset(
            dataset,
            column_policies={
                "value": ColumnCleaningPolicy("mode"),
            },
        )


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            ColumnCleaningPolicy("constant"),
            "non-empty fill_value",
        ),
        (
            ColumnCleaningPolicy("mean", "0"),
            "only valid",
        ),
    ],
)
def test_invalid_policy_options_are_rejected(
    policy: ColumnCleaningPolicy,
    message: str,
) -> None:
    dataset = CsvDataset(headers=("value",), records=({"value": ""},))

    with pytest.raises(DataCleaningError, match=message):
        clean_dataset(dataset, column_policies={"value": policy})


def test_unknown_columns_are_rejected() -> None:
    dataset = CsvDataset(headers=("name",), records=({"name": "Ali"},))

    with pytest.raises(DataCleaningError, match="unknown columns"):
        clean_dataset(
            dataset,
            column_policies={
                "age": ColumnCleaningPolicy("drop"),
            },
        )


def test_unsupported_strategy_is_rejected() -> None:
    dataset = CsvDataset(headers=("value",), records=({"value": ""},))

    with pytest.raises(DataCleaningError, match="Unsupported"):
        clean_dataset(
            dataset,
            column_policies={
                "value": ColumnCleaningPolicy("unknown"),  # type: ignore[arg-type]
            },
        )


def test_cleaning_does_not_mutate_original_dataset() -> None:
    original_record = {"name": "Ali", "city": ""}
    dataset = CsvDataset(
        headers=("name", "city"),
        records=(original_record,),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "city": ColumnCleaningPolicy("constant", "Karachi"),
        },
    )

    assert dataset.records == ({"name": "Ali", "city": ""},)
    assert original_record["city"] == ""
    assert result.dataset.records[0]["city"] == "Karachi"
    assert result.dataset.records[0] is not original_record


def test_cleaning_is_deterministic_and_returns_immutable_result() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": "1"}, {"value": ""}, {"value": "3"}),
    )
    policy = {"value": ColumnCleaningPolicy("median")}

    first = clean_dataset(dataset, column_policies=policy)
    second = clean_dataset(dataset, column_policies=policy)

    assert first == second
    assert isinstance(first, CleaningResult)
    with pytest.raises(AttributeError):
        first.report.output_row_count = 99  # type: ignore[misc]


def test_empty_dataset_is_supported_without_imputation() -> None:
    dataset = CsvDataset(headers=("name",), records=())

    result = clean_dataset(dataset)

    assert result.dataset.records == ()
    assert result.report.input_row_count == 0
    assert result.report.output_row_count == 0


def test_unconfigured_columns_remain_unchanged() -> None:
    dataset = CsvDataset(
        headers=("name", "city"),
        records=({"name": "", "city": ""},),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "city": ColumnCleaningPolicy("constant", "Karachi"),
        },
    )

    assert result.dataset.records == ({"name": "", "city": "Karachi"},)


def test_imputation_uses_only_rows_retained_by_drop_policy() -> None:
    dataset = CsvDataset(
        headers=("value", "required"),
        records=(
            {"value": "100", "required": ""},
            {"value": "2", "required": "yes"},
            {"value": "4", "required": "yes"},
            {"value": "", "required": "yes"},
        ),
    )

    result = clean_dataset(
        dataset,
        column_policies={
            "value": ColumnCleaningPolicy("mean"),
            "required": ColumnCleaningPolicy("drop"),
        },
    )

    assert result.dataset.records == (
        {"value": "2", "required": "yes"},
        {"value": "4", "required": "yes"},
        {"value": "3", "required": "yes"},
    )
    assert result.report.removed_row_indices == (0,)


@pytest.mark.parametrize("policy", [None, "mean", object()])
def test_invalid_policy_objects_are_rejected(policy: object) -> None:
    dataset = CsvDataset(headers=("value",), records=({"value": "1"},))

    with pytest.raises(DataCleaningError, match="ColumnCleaningPolicy"):
        clean_dataset(
            dataset,
            column_policies={"value": policy},  # type: ignore[dict-item]
        )
