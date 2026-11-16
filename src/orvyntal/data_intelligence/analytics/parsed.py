"""Shared, immutable parsed representations for dataset analytics."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


@dataclass(frozen=True, slots=True)
class ParsedColumn:
    """Parsed numeric values aligned with the source dataset's row order."""

    name: str
    values: tuple[Decimal | None, ...]
    missing_count: int
    invalid_count: int

    @property
    def valid_count(self) -> int:
        """Return the number of successfully parsed numeric values."""
        return len(self.values) - self.missing_count - self.invalid_count


@dataclass(frozen=True, slots=True)
class ParsedDataset:
    """Read-only parsed columns for one source dataset."""

    headers: tuple[str, ...]
    records: tuple[tuple[str, ...], ...]
    columns: tuple[ParsedColumn, ...]
    by_name: Mapping[str, ParsedColumn]


def parse_dataset(
    dataset: CsvDataset,
    columns: tuple[str, ...] | None = None,
) -> ParsedDataset:
    """Parse selected non-empty cells once and retain their row alignment."""
    if not isinstance(dataset, CsvDataset):
        raise TypeError("dataset must be a CsvDataset")

    if columns is None:
        selected_headers = dataset.headers
    else:
        if not isinstance(columns, tuple):
            raise TypeError("columns must be a tuple of column names")
        if len(set(columns)) != len(columns):
            raise ValueError("columns must not contain duplicates")

        unknown_columns = set(columns).difference(dataset.headers)
        if unknown_columns:
            unknown = ", ".join(sorted(unknown_columns))
            raise ValueError(f"unknown columns: {unknown}")

        selected_headers = tuple(
            header for header in dataset.headers if header in columns
        )

    headers = dataset.headers
    records = tuple(
        tuple(record.get(header, "") for header in headers)
        for record in dataset.records
    )
    header_indices = {header: index for index, header in enumerate(headers)}
    parsed_columns: list[ParsedColumn] = []

    for header in selected_headers:
        values: list[Decimal | None] = []
        missing_count = 0
        invalid_count = 0

        column_index = header_indices[header]
        for record in records:
            raw_value = record[column_index]

            if not isinstance(raw_value, str) or not raw_value.strip():
                values.append(None)
                missing_count += 1
                continue

            number = parse_numeric_value(raw_value)
            values.append(number)

            if number is None:
                invalid_count += 1

        parsed_columns.append(
            ParsedColumn(
                name=header,
                values=tuple(values),
                missing_count=missing_count,
                invalid_count=invalid_count,
            )
        )

    columns = tuple(parsed_columns)
    by_name = MappingProxyType({column.name: column for column in columns})

    return ParsedDataset(
        headers=headers,
        records=records,
        columns=columns,
        by_name=by_name,
    )


def scale_decimal_values(
    values: tuple[Decimal, ...] | list[Decimal],
) -> tuple[tuple[int, ...], int]:
    """Convert validated finite decimals to integers with one common exponent.

    Values must satisfy the shared numeric parsing policy. Each original value
    is exactly equal to its returned integer multiplied by 10 raised to the
    returned exponent.
    """
    if not values:
        return (), 0

    exponent = min(value.as_tuple().exponent for value in values)
    scaled: list[int] = []

    for value in values:
        parts = value.as_tuple()
        coefficient = int("".join(str(digit) for digit in parts.digits) or "0")

        if parts.sign:
            coefficient = -coefficient

        scaled.append(coefficient * 10 ** (parts.exponent - exponent))

    return tuple(scaled), exponent
