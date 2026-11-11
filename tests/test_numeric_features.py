"""Tests for deterministic numeric feature extraction."""

from decimal import Decimal

import pytest

from orvyntal.data_intelligence.features.numeric import (
    NumericFeatureError,
    extract_numeric_features,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def test_extracts_selected_columns_in_dataset_header_order() -> None:
    dataset = CsvDataset(
        headers=("name", "score", "age"),
        records=(
            {"name": "Ali", "score": "95.5", "age": "21"},
            {"name": "Sara", "score": "88", "age": "22"},
        ),
    )

    result = extract_numeric_features(dataset, ("age", "score"))

    assert result.columns == ("score", "age")
    assert result.values == (
        (Decimal("95.5"), Decimal("21")),
        (Decimal("88"), Decimal("22")),
    )
    assert result.row_indices == (0, 1)
    assert result.dropped_row_indices == ()


def test_strict_policy_rejects_invalid_numeric_values() -> None:
    dataset = CsvDataset(
        headers=("score",),
        records=({"score": "95"}, {"score": "invalid"}),
    )

    with pytest.raises(
        NumericFeatureError,
        match="row 1, column 'score'",
    ):
        extract_numeric_features(dataset, ("score",))


@pytest.mark.parametrize("value", ["", "  ", "NaN", "Infinity", "-Infinity"])
def test_strict_policy_rejects_missing_and_non_finite_values(
    value: str,
) -> None:
    dataset = CsvDataset(
        headers=("score",),
        records=({"score": value},),
    )

    with pytest.raises(NumericFeatureError):
        extract_numeric_features(dataset, ("score",))


def test_drop_policy_preserves_original_row_indexes() -> None:
    dataset = CsvDataset(
        headers=("score", "age"),
        records=(
            {"score": "90", "age": "20"},
            {"score": "invalid", "age": "21"},
            {"score": "80", "age": "invalid"},
            {"score": "70", "age": "23"},
        ),
    )

    result = extract_numeric_features(
        dataset,
        ("score", "age"),
        invalid_policy="drop",
    )

    assert result.values == (
        (Decimal("90"), Decimal("20")),
        (Decimal("70"), Decimal("23")),
    )
    assert result.row_indices == (0, 3)
    assert result.dropped_row_indices == (1, 2)


def test_unselected_invalid_columns_do_not_affect_extraction() -> None:
    dataset = CsvDataset(
        headers=("name", "score"),
        records=(
            {"name": "Ali", "score": "90"},
            {"name": "Sara", "score": "80"},
        ),
    )

    result = extract_numeric_features(dataset, ("score",))

    assert result.values == ((Decimal("90"),), (Decimal("80"),))


def test_extraction_does_not_mutate_input_dataset() -> None:
    dataset = CsvDataset(
        headers=("score",),
        records=({"score": "90"}, {"score": "invalid"}),
    )
    original_records = tuple(dict(row) for row in dataset.records)

    extract_numeric_features(
        dataset,
        ("score",),
        invalid_policy="drop",
    )

    assert dataset.records == original_records


def test_empty_dataset_returns_empty_matrix() -> None:
    dataset = CsvDataset(headers=("score",), records=())

    result = extract_numeric_features(dataset, ("score",))

    assert result.values == ()
    assert result.row_indices == ()
    assert result.dropped_row_indices == ()


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ((), "non-empty tuple"),
        (("score", "score"), "duplicates"),
        (("unknown",), "unknown columns"),
        (("",), "non-empty strings"),
    ],
)
def test_invalid_column_selection_is_rejected(
    columns: tuple[str, ...],
    message: str,
) -> None:
    dataset = CsvDataset(
        headers=("score",),
        records=({"score": "90"},),
    )

    with pytest.raises(NumericFeatureError, match=message):
        extract_numeric_features(dataset, columns)


def test_non_tuple_column_selection_is_rejected() -> None:
    dataset = CsvDataset(headers=("score",), records=())

    with pytest.raises(NumericFeatureError, match="non-empty tuple"):
        extract_numeric_features(dataset, ["score"])  # type: ignore[arg-type]


def test_invalid_policy_is_rejected() -> None:
    dataset = CsvDataset(headers=("score",), records=())

    with pytest.raises(NumericFeatureError, match="invalid_policy"):
        extract_numeric_features(
            dataset,
            ("score",),
            invalid_policy="ignore",  # type: ignore[arg-type]
        )


def test_non_dataset_input_is_rejected() -> None:
    with pytest.raises(NumericFeatureError, match="CsvDataset"):
        extract_numeric_features(
            None,  # type: ignore[arg-type]
            ("score",),
        )
