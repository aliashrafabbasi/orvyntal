"""Deterministic categorical feature encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


class CategoricalFeatureError(ValueError):
    """Raised when categorical features cannot be encoded safely."""


@dataclass(frozen=True, slots=True)
class CategoricalFeatureEncoder:
    """Immutable category mappings learned from a dataset."""

    columns: tuple[str, ...]
    categories: tuple[tuple[str, tuple[str, ...]], ...]

    def mappings(self) -> dict[str, dict[str, int]]:
        """Return independent category-to-integer mappings."""
        return {
            column: {category: index for index, category in enumerate(values)}
            for column, values in self.categories
        }


@dataclass(frozen=True, slots=True)
class CategoricalFeatureMatrix:
    """Immutable encoded matrix with original row-index tracking."""

    columns: tuple[str, ...]
    values: tuple[tuple[int, ...], ...]
    row_indices: tuple[int, ...]


def _validate_columns(dataset: CsvDataset, columns: tuple[str, ...]) -> None:
    if not isinstance(dataset, CsvDataset):
        raise CategoricalFeatureError("dataset must be a CsvDataset")

    if not isinstance(columns, tuple) or not columns:
        raise CategoricalFeatureError("columns must be a non-empty tuple")

    if any(not isinstance(column, str) or not column.strip() for column in columns):
        raise CategoricalFeatureError("column names must be non-empty strings")

    if len(set(columns)) != len(columns):
        raise CategoricalFeatureError("columns must not contain duplicates")

    unknown_columns = set(columns).difference(dataset.headers)
    if unknown_columns:
        unknown = ", ".join(sorted(unknown_columns))
        raise CategoricalFeatureError(f"unknown columns: {unknown}")


def _validate_encoder(encoder: CategoricalFeatureEncoder) -> None:
    columns = encoder.columns
    categories = encoder.categories

    if (
        not isinstance(columns, tuple)
        or not columns
        or any(not isinstance(column, str) or not column.strip() for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise CategoricalFeatureError("encoder contains invalid columns")

    if not isinstance(categories, tuple) or len(categories) != len(columns):
        raise CategoricalFeatureError("encoder category definitions are inconsistent")

    for expected_column, definition in zip(columns, categories, strict=True):
        if (
            not isinstance(definition, tuple)
            or len(definition) != 2
            or definition[0] != expected_column
        ):
            raise CategoricalFeatureError(
                "encoder category definitions are inconsistent"
            )

        values = definition[1]
        if (
            not isinstance(values, tuple)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
            or tuple(sorted(values)) != values
        ):
            raise CategoricalFeatureError(
                f"encoder categories for {expected_column!r} are invalid"
            )


def fit_categorical_encoder(
    dataset: CsvDataset,
    columns: tuple[str, ...],
) -> CategoricalFeatureEncoder:
    """Learn deterministic, lexicographically ordered category mappings."""
    _validate_columns(dataset, columns)
    selected_columns = tuple(header for header in dataset.headers if header in columns)

    categories: list[tuple[str, tuple[str, ...]]] = []

    for column in selected_columns:
        observed: set[str] = set()

        for row_index, record in enumerate(dataset.records):
            value = record.get(column)

            if not isinstance(value, str) or not value.strip():
                raise CategoricalFeatureError(
                    f"row {row_index}, column {column!r}: "
                    "category is missing or invalid"
                )

            observed.add(value)

        categories.append((column, tuple(sorted(observed))))

    return CategoricalFeatureEncoder(
        columns=selected_columns,
        categories=tuple(categories),
    )


def transform_categorical_features(
    dataset: CsvDataset,
    encoder: CategoricalFeatureEncoder,
    *,
    unknown_policy: Literal["error", "ignore"] = "error",
) -> CategoricalFeatureMatrix:
    """Apply fitted mappings without learning from or mutating the input.

    Unknown categories raise an error by default. With ``unknown_policy="ignore"``,
    they are encoded as ``-1``; rows are never dropped.
    """
    if not isinstance(dataset, CsvDataset):
        raise CategoricalFeatureError("dataset must be a CsvDataset")

    if not isinstance(encoder, CategoricalFeatureEncoder):
        raise CategoricalFeatureError("encoder must be a CategoricalFeatureEncoder")

    if unknown_policy not in ("error", "ignore"):
        raise CategoricalFeatureError("unknown_policy must be 'error' or 'ignore'")

    _validate_encoder(encoder)

    if tuple(column for column in dataset.headers if column in encoder.columns) != (
        encoder.columns
    ):
        raise CategoricalFeatureError(
            "dataset must contain all encoder columns in the original header order"
        )

    mappings = encoder.mappings()

    encoded_rows: list[tuple[int, ...]] = []
    row_indices: list[int] = []

    for row_index, record in enumerate(dataset.records):
        encoded_row: list[int] = []
        invalid_column: str | None = None

        for column in encoder.columns:
            raw_value = record.get(column)

            if not isinstance(raw_value, str) or not raw_value.strip():
                invalid_column = column
                break

            category_id = mappings[column].get(raw_value)
            if category_id is None:
                if unknown_policy == "error":
                    raise CategoricalFeatureError(
                        f"row {row_index}, column {column!r}: "
                        f"unknown category {raw_value!r}"
                    )

                encoded_row.append(-1)
                continue

            encoded_row.append(category_id)

        if invalid_column is not None:
            raise CategoricalFeatureError(
                f"row {row_index}, column {invalid_column!r}: "
                "category is missing or invalid"
            )

        encoded_rows.append(tuple(encoded_row))
        row_indices.append(row_index)

    return CategoricalFeatureMatrix(
        columns=encoder.columns,
        values=tuple(encoded_rows),
        row_indices=tuple(row_indices),
    )
