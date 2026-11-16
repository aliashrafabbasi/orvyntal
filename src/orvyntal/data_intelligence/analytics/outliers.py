"""Deterministic numeric outlier analysis using Tukey's IQR rule."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from orvyntal.data_intelligence.analytics.parsed import (
    ParsedColumn,
    ParsedDataset,
    parse_dataset,
    scale_decimal_values,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


@dataclass(frozen=True, slots=True)
class OutlierObservation:
    """An outlier value and its zero-based index in the dataset."""

    row_index: int
    value: Decimal


@dataclass(frozen=True, slots=True)
class ColumnOutlierAnalysis:
    """IQR statistics and detected outliers for one dataset column."""

    column_name: str
    observation_count: int
    missing_count: int
    invalid_count: int
    q1: Fraction | None
    q3: Fraction | None
    iqr: Fraction | None
    lower_bound: Fraction | None
    upper_bound: Fraction | None
    outliers: tuple[OutlierObservation, ...]


@dataclass(frozen=True, slots=True)
class OutlierAnalysis:
    """Immutable outlier analysis in original column order."""

    columns: tuple[ColumnOutlierAnalysis, ...]


def _median_scaled(values: list[int]) -> Fraction:
    """Return the exact median of a non-empty sorted integer sequence."""
    middle = len(values) // 2
    if len(values) % 2:
        return Fraction(values[middle])
    return Fraction(values[middle - 1] + values[middle], 2)


def _scaled_fraction(
    value: Fraction,
    exponent: int,
) -> Fraction:
    """Convert a rational in scaled-integer units to its exact decimal value."""
    if exponent >= 0:
        return value * 10**exponent
    return value / 10**-exponent


def _analyze_parsed_column(
    column: ParsedColumn,
) -> ColumnOutlierAnalysis:
    """Analyze one shared parsed column without modifying source records."""
    valid = [
        (row_index, value)
        for row_index, value in enumerate(column.values)
        if value is not None
    ]
    observation_count = len(valid)

    if observation_count < 2:
        return ColumnOutlierAnalysis(
            column_name=column.name,
            observation_count=observation_count,
            missing_count=column.missing_count,
            invalid_count=column.invalid_count,
            q1=None,
            q3=None,
            iqr=None,
            lower_bound=None,
            upper_bound=None,
            outliers=(),
        )

    scaled, exponent = scale_decimal_values([value for _, value in valid])
    ordered = sorted(
        zip(valid, scaled, strict=True),
        key=lambda item: item[1],
    )
    values = [scaled_value for _, scaled_value in ordered]
    midpoint = observation_count // 2

    lower_half = values[:midpoint]
    upper_half = values[(observation_count + 1) // 2 :]

    q1_scaled = _median_scaled(lower_half)
    q3_scaled = _median_scaled(upper_half)
    iqr_scaled = q3_scaled - q1_scaled
    lower_bound_scaled = q1_scaled - Fraction(3, 2) * iqr_scaled
    upper_bound_scaled = q3_scaled + Fraction(3, 2) * iqr_scaled

    q1 = _scaled_fraction(q1_scaled, exponent)
    q3 = _scaled_fraction(q3_scaled, exponent)
    iqr = _scaled_fraction(iqr_scaled, exponent)
    lower_bound = _scaled_fraction(lower_bound_scaled, exponent)
    upper_bound = _scaled_fraction(upper_bound_scaled, exponent)

    outliers = tuple(
        OutlierObservation(row_index=row_index, value=value)
        for (row_index, value), scaled_value in zip(valid, scaled, strict=True)
        if (
            scaled_value * lower_bound_scaled.denominator < lower_bound_scaled.numerator
            or scaled_value * upper_bound_scaled.denominator
            > upper_bound_scaled.numerator
        )
    )

    return ColumnOutlierAnalysis(
        column_name=column.name,
        observation_count=observation_count,
        missing_count=column.missing_count,
        invalid_count=column.invalid_count,
        q1=q1,
        q3=q3,
        iqr=iqr,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        outliers=outliers,
    )


def analyze_parsed_outliers(dataset: ParsedDataset) -> OutlierAnalysis:
    """Analyze every shared parsed column using Tukey's 1.5-IQR rule."""
    return OutlierAnalysis(
        columns=tuple(_analyze_parsed_column(column) for column in dataset.columns)
    )


def analyze_numeric_outliers(dataset: CsvDataset) -> OutlierAnalysis:
    """Analyze every column using Tukey's 1.5-IQR outlier rule.

    Missing and invalid numeric values are excluded from the calculation.
    Columns with fewer than two valid observations have no quartile bounds.
    Source records are never changed.
    """
    return analyze_parsed_outliers(parse_dataset(dataset))
