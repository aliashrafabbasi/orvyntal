"""Deterministic Pearson correlation analysis for CSV datasets."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from orvyntal.data_intelligence.analytics.descriptive import (
    _round_sqrt_fraction,
)
from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


@dataclass(frozen=True, slots=True)
class CorrelationPair:
    """Pearson correlation between two columns."""

    column_x: str
    column_y: str
    observation_count: int
    correlation: Decimal | None


@dataclass(frozen=True, slots=True)
class CorrelationAnalysis:
    """Ordered pairwise correlations for numeric columns."""

    pairs: tuple[CorrelationPair, ...]


def _scaled_integer_values(
    values: tuple[str, ...],
) -> tuple[int | None, ...]:
    """Convert valid decimal strings to exact integers at a shared scale."""
    parsed: list[Decimal | None] = []
    minimum_exponent: int | None = None

    for value in values:
        number = parse_numeric_value(value) if value.strip() else None
        parsed.append(number)

        if number is not None:
            exponent = number.as_tuple().exponent
            if isinstance(exponent, int):
                minimum_exponent = (
                    exponent
                    if minimum_exponent is None
                    else min(minimum_exponent, exponent)
                )

    if minimum_exponent is None:
        return tuple(None for _ in parsed)

    scaled: list[int | None] = []

    for number in parsed:
        if number is None:
            scaled.append(None)
            continue

        sign, digits, exponent = number.as_tuple()
        coefficient = int("".join(map(str, digits)) or "0")
        integer = coefficient * 10 ** (exponent - minimum_exponent)
        scaled.append(-integer if sign else integer)

    return tuple(scaled)


def _correlate_pair(
    column_x: str,
    column_y: str,
    values_x: tuple[int | None, ...],
    values_y: tuple[int | None, ...],
) -> CorrelationPair:
    """Calculate Pearson correlation using pairwise complete observations."""
    count = 0
    sum_x = sum_y = sum_x_squared = sum_y_squared = sum_xy = 0

    for x, y in zip(values_x, values_y, strict=True):
        if x is None or y is None:
            continue

        count += 1
        sum_x += x
        sum_y += y
        sum_x_squared += x * x
        sum_y_squared += y * y
        sum_xy += x * y

    if count < 2:
        return CorrelationPair(column_x, column_y, count, None)

    centered_x = count * sum_x_squared - sum_x * sum_x
    centered_y = count * sum_y_squared - sum_y * sum_y
    centered_xy = count * sum_xy - sum_x * sum_y

    if centered_x == 0 or centered_y == 0:
        return CorrelationPair(column_x, column_y, count, None)

    squared_correlation = Fraction(
        centered_xy * centered_xy,
        centered_x * centered_y,
    )
    correlation = _round_sqrt_fraction(squared_correlation)

    if centered_xy < 0 and correlation != 0:
        correlation = correlation.copy_negate()

    return CorrelationPair(column_x, column_y, count, correlation)


def correlate_numeric_columns(dataset: CsvDataset) -> CorrelationAnalysis:
    """Calculate Pearson correlations for columns containing numeric values.

    Missing and policy-invalid values are excluded pairwise. Columns with no
    valid numeric observations are omitted. A pair has no coefficient when
    fewer than two complete observations exist or either column is constant.
    Results are deterministic and do not mutate the dataset.
    """
    numeric_columns: list[tuple[str, tuple[int | None, ...]]] = []

    for header in dataset.headers:
        values = tuple(record[header] for record in dataset.records)
        scaled = _scaled_integer_values(values)

        if any(value is not None for value in scaled):
            numeric_columns.append((header, scaled))

    pairs = tuple(
        _correlate_pair(name_x, name_y, values_x, values_y)
        for index, (name_x, values_x) in enumerate(numeric_columns)
        for name_y, values_y in numeric_columns[index + 1 :]
    )

    return CorrelationAnalysis(pairs=pairs)
