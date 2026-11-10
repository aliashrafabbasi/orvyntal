"""Explicit, deterministic CSV dataset cleaning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Literal

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset

CleaningStrategy = Literal["drop", "constant", "mean", "median", "mode"]


class DataCleaningError(ValueError):
    """Raised when a requested cleaning operation cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class ColumnCleaningPolicy:
    """A missing-value strategy for one explicitly selected column."""

    strategy: CleaningStrategy
    fill_value: str | None = None


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """Audit details for a cleaning operation."""

    input_row_count: int
    output_row_count: int
    missing_rows_removed: int
    duplicate_rows_removed: int
    filled_values: tuple[tuple[str, int], ...]
    strategies: tuple[tuple[str, str], ...]
    removed_row_indices: tuple[int, ...]
    duplicate_row_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """A cleaned dataset and its audit report."""

    dataset: CsvDataset
    report: CleaningReport


def _is_missing(value: str) -> bool:
    return not value.strip()


def _imputation_value(
    column: str,
    values: list[str],
    strategy: CleaningStrategy,
    fill_value: str | None,
) -> str:
    if strategy == "constant":
        if fill_value is None or not fill_value.strip():
            raise DataCleaningError(
                f"Column {column!r} requires a non-empty fill_value "
                "for the constant strategy."
            )
        return fill_value

    if strategy == "mode":
        if not values:
            raise DataCleaningError(
                f"Cannot calculate mode for column {column!r}: no observed values."
            )
        counts = Counter(values)
        highest_count = max(counts.values())
        return min(value for value, count in counts.items() if count == highest_count)

    numbers = [parse_numeric_value(value) for value in values]
    if not values or any(number is None for number in numbers):
        raise DataCleaningError(
            f"Cannot apply {strategy} imputation to column {column!r}: "
            "all observed values must be valid finite numbers."
        )

    numeric_values = sorted(number for number in numbers if number is not None)

    with localcontext() as context:
        context.prec = 50

        if strategy == "median":
            middle = len(numeric_values) // 2
            if len(numeric_values) % 2:
                result = numeric_values[middle]
            else:
                result = (numeric_values[middle - 1] + numeric_values[middle]) / 2
        else:
            result = sum(numeric_values, Decimal(0)) / Decimal(len(numeric_values))

        return str(+result)


def clean_dataset(
    dataset: CsvDataset,
    *,
    column_policies: Mapping[str, ColumnCleaningPolicy] | None = None,
    remove_duplicates: bool = False,
) -> CleaningResult:
    """Apply explicit missing-value and duplicate-row operations.

    Missing values are empty or whitespace-only strings. The input is never
    mutated. Policies are evaluated in header order. Duplicate removal keeps
    the first occurrence and runs after missing-value handling.
    """
    policies = dict(column_policies or {})
    unknown_columns = sorted(set(policies) - set(dataset.headers))
    if unknown_columns:
        raise DataCleaningError(
            "Cleaning policies reference unknown columns: "
            f"{', '.join(unknown_columns)}."
        )

    allowed_strategies = {"drop", "constant", "mean", "median", "mode"}
    for column, policy in policies.items():
        if not isinstance(policy, ColumnCleaningPolicy):
            raise DataCleaningError(
                f"Column {column!r} requires a ColumnCleaningPolicy."
            )
        if policy.strategy not in allowed_strategies:
            raise DataCleaningError(
                f"Unsupported cleaning strategy for column {column!r}: "
                f"{policy.strategy!r}."
            )
        if policy.strategy != "constant" and policy.fill_value is not None:
            raise DataCleaningError(
                f"Column {column!r}: fill_value is only valid "
                "with the constant strategy."
            )

    ordered_policies = tuple(
        (column, policies[column]) for column in dataset.headers if column in policies
    )
    drop_columns = {
        column for column, policy in ordered_policies if policy.strategy == "drop"
    }

    retained_records: list[tuple[int, dict[str, str]]] = []
    removed_indices: list[int] = []

    for row_index, record in enumerate(dataset.records):
        if any(_is_missing(record[column]) for column in drop_columns):
            removed_indices.append(row_index)
        else:
            retained_records.append((row_index, record))

    imputation_values: dict[str, str] = {}
    for column, policy in ordered_policies:
        if policy.strategy == "drop":
            continue

        observed = [
            record[column]
            for _, record in retained_records
            if not _is_missing(record[column])
        ]
        imputation_values[column] = _imputation_value(
            column, observed, policy.strategy, policy.fill_value
        )

    output_records: list[dict[str, str]] = []
    retained_indices: list[int] = []
    filled_counts = {column: 0 for column, _ in ordered_policies}

    for row_index, record in retained_records:
        cleaned_record = dict(record)
        for column, _policy in ordered_policies:
            if _is_missing(cleaned_record[column]):
                cleaned_record[column] = imputation_values[column]
                filled_counts[column] += 1

        output_records.append(cleaned_record)
        retained_indices.append(row_index)

    duplicate_indices: list[int] = []
    if remove_duplicates:
        unique_records: list[dict[str, str]] = []
        seen: set[tuple[str, ...]] = set()

        for row_index, record in zip(retained_indices, output_records, strict=True):
            key = tuple(record[column] for column in dataset.headers)
            if key in seen:
                duplicate_indices.append(row_index)
                continue

            seen.add(key)
            unique_records.append(record)

        output_records = unique_records

    cleaned_dataset = CsvDataset(
        headers=tuple(dataset.headers),
        records=tuple(dict(record) for record in output_records),
    )

    report = CleaningReport(
        input_row_count=len(dataset.records),
        output_row_count=len(cleaned_dataset.records),
        missing_rows_removed=len(removed_indices),
        duplicate_rows_removed=len(duplicate_indices),
        filled_values=tuple(
            (column, filled_counts[column])
            for column, _ in ordered_policies
            if filled_counts[column]
        ),
        strategies=tuple(
            (column, policy.strategy) for column, policy in ordered_policies
        ),
        removed_row_indices=tuple(removed_indices),
        duplicate_row_indices=tuple(duplicate_indices),
    )

    return CleaningResult(dataset=cleaned_dataset, report=report)
