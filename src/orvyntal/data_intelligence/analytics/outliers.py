"""Deterministic numeric outlier analysis using Tukey's IQR rule."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value
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


def _median(values: list[Fraction]) -> Fraction:
    """Return the exact median of a non-empty sorted sequence."""
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _analyze_column(
    dataset: CsvDataset,
    column_name: str,
) -> ColumnOutlierAnalysis:
    """Analyze one column without modifying its source records."""
    valid: list[tuple[int, Decimal, Fraction]] = []
    missing_count = 0
    invalid_count = 0

    for row_index, record in enumerate(dataset.records):
        raw_value = record[column_name]
        if not raw_value.strip():
            missing_count += 1
            continue

        value = parse_numeric_value(raw_value)
        if value is None:
            invalid_count += 1
            continue

        valid.append((row_index, value, Fraction(value)))

    observation_count = len(valid)

    if observation_count < 2:
        return ColumnOutlierAnalysis(
            column_name=column_name,
            observation_count=observation_count,
            missing_count=missing_count,
            invalid_count=invalid_count,
            q1=None,
            q3=None,
            iqr=None,
            lower_bound=None,
            upper_bound=None,
            outliers=(),
        )

    ordered = sorted(valid, key=lambda item: item[2])
    values = [item[2] for item in ordered]
    midpoint = observation_count // 2

    lower_half = values[:midpoint]
    upper_half = values[(observation_count + 1) // 2 :]

    q1 = _median(lower_half)
    q3 = _median(upper_half)
    iqr = q3 - q1
    lower_bound = q1 - Fraction(3, 2) * iqr
    upper_bound = q3 + Fraction(3, 2) * iqr

    outliers = tuple(
        OutlierObservation(row_index=row_index, value=value)
        for row_index, value, exact_value in valid
        if exact_value < lower_bound or exact_value > upper_bound
    )

    return ColumnOutlierAnalysis(
        column_name=column_name,
        observation_count=observation_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        q1=q1,
        q3=q3,
        iqr=iqr,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        outliers=outliers,
    )


def analyze_numeric_outliers(dataset: CsvDataset) -> OutlierAnalysis:
    """Analyze every column using Tukey's 1.5-IQR outlier rule.

    Missing and invalid numeric values are excluded from the calculation.
    Columns with fewer than two valid observations have no quartile bounds.
    Source records are never changed.
    """
    return OutlierAnalysis(
        columns=tuple(
            _analyze_column(dataset, column_name) for column_name in dataset.headers
        )
    )
