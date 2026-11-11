from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


class NumericFeatureError(ValueError):
    """Raised when a dataset cannot be converted into numeric features."""


@dataclass(frozen=True, slots=True)
class NumericFeatureMatrix:
    """Immutable numeric feature matrix with original row-index tracking."""

    columns: tuple[str, ...]
    values: tuple[tuple[Decimal, ...], ...]
    row_indices: tuple[int, ...]
    dropped_row_indices: tuple[int, ...]


def extract_numeric_features(
    dataset: CsvDataset,
    columns: tuple[str, ...],
    *,
    invalid_policy: Literal["error", "drop"] = "error",
) -> NumericFeatureMatrix:
    """Extract selected columns as Decimal values without mutating the dataset."""
    if not isinstance(dataset, CsvDataset):
        raise NumericFeatureError("dataset must be a CsvDataset")

    if invalid_policy not in ("error", "drop"):
        raise NumericFeatureError("invalid_policy must be 'error' or 'drop'")

    if not isinstance(columns, tuple) or not columns:
        raise NumericFeatureError("columns must be a non-empty tuple")

    if any(not isinstance(column, str) or not column.strip() for column in columns):
        raise NumericFeatureError("column names must be non-empty strings")

    if len(set(columns)) != len(columns):
        raise NumericFeatureError("columns must not contain duplicates")

    unknown_columns = set(columns).difference(dataset.headers)
    if unknown_columns:
        unknown = ", ".join(sorted(unknown_columns))
        raise NumericFeatureError(f"unknown columns: {unknown}")

    selected_columns = tuple(header for header in dataset.headers if header in columns)

    extracted_rows: list[tuple[Decimal, ...]] = []
    row_indices: list[int] = []
    dropped_row_indices: list[int] = []

    for row_index, record in enumerate(dataset.records):
        numeric_row: list[Decimal] = []
        invalid_column: str | None = None

        for column in selected_columns:
            raw_value = record.get(column)
            number = (
                parse_numeric_value(raw_value) if isinstance(raw_value, str) else None
            )

            if number is None:
                invalid_column = column
                break

            numeric_row.append(number)

        if invalid_column is not None:
            if invalid_policy == "error":
                raise NumericFeatureError(
                    f"row {row_index}, column {invalid_column!r}: "
                    "value is missing or not a valid numeric value"
                )

            dropped_row_indices.append(row_index)
            continue

        extracted_rows.append(tuple(numeric_row))
        row_indices.append(row_index)

    return NumericFeatureMatrix(
        columns=selected_columns,
        values=tuple(extracted_rows),
        row_indices=tuple(row_indices),
        dropped_row_indices=tuple(dropped_row_indices),
    )
