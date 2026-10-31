"""Deterministic, dependency-free dataset profiling."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset

_SUMMARY_QUANTUM = Decimal("0.000001")
_BOOLEAN_VALUES = frozenset({"true", "false", "yes", "no"})


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Statistics for values successfully parsed as finite decimals."""

    count: int
    invalid_count: int
    mean: Decimal
    median: Decimal
    minimum: Decimal
    maximum: Decimal


@dataclass(frozen=True, slots=True)
class CategoryFrequency:
    """Frequency and percentage for one categorical value."""

    value: str
    count: int
    percentage: Decimal


@dataclass(frozen=True, slots=True)
class CategoricalSummary:
    """Bounded category frequencies for a categorical column."""

    unique_count: int
    top_categories: tuple[CategoryFrequency, ...]


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Missing-value, type-inference, and summary information for one column."""

    name: str
    inferred_type: str
    missing_count: int
    missing_percentage: Decimal
    numeric_summary: NumericSummary | None
    categorical_summary: CategoricalSummary | None


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """A deterministic profile of a CSV dataset."""

    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: tuple[ColumnProfile, ...]


def _parse_decimal(value: str) -> Decimal | None:
    """Return a policy-compliant finite Decimal, or None."""
    return parse_numeric_value(value)


def _round_fraction(value: Fraction) -> Decimal:
    """Round an exact rational to six decimal places using half-even."""
    places = abs(_SUMMARY_QUANTUM.as_tuple().exponent)
    scale = 10**places
    numerator = abs(value.numerator) * scale
    quotient, remainder = divmod(numerator, value.denominator)

    doubled_remainder = remainder * 2
    if doubled_remainder > value.denominator or (
        doubled_remainder == value.denominator and quotient % 2
    ):
        quotient += 1

    scaled = -quotient if value < 0 else quotient
    digits = str(abs(scaled)).rjust(places + 1, "0")
    sign = "-" if scaled < 0 else ""
    return Decimal(f"{sign}{digits[:-places]}.{digits[-places:]}")


def _infer_type(
    values: list[str],
    numbers: list[Decimal | None],
) -> str:
    """Infer a conservative type from non-missing column values."""
    if not values:
        return "text"

    normalized = [value.strip().casefold() for value in values]
    if all(value in _BOOLEAN_VALUES for value in normalized):
        return "boolean"

    if all(number is not None for number in numbers):
        if all(number == number.to_integral_value() for number in numbers):
            return "integer"
        return "numeric"

    return "text"


def _numeric_summary(numbers: list[Decimal | None]) -> NumericSummary | None:
    """Summarize finite numbers using enough precision for six-place rounding."""
    valid = sorted(number for number in numbers if number is not None)
    if not valid:
        return None

    count = len(valid)
    exact_numbers = [Fraction(number) for number in valid]
    midpoint = count // 2

    if count % 2:
        exact_median = exact_numbers[midpoint]
    else:
        exact_median = (exact_numbers[midpoint - 1] + exact_numbers[midpoint]) / 2

    exact_mean = sum(exact_numbers, Fraction(0)) / count
    mean = _round_fraction(exact_mean)
    median = _round_fraction(exact_median)

    return NumericSummary(
        count=count,
        invalid_count=len(numbers) - count,
        mean=mean,
        median=median,
        minimum=valid[0],
        maximum=valid[-1],
    )


def _categorical_summary(
    values: list[str],
    inferred_type: str,
) -> CategoricalSummary | None:
    """Summarize text and boolean values with deterministic top-20 ordering."""
    if inferred_type not in {"text", "boolean"}:
        return None

    frequencies: dict[str, int] = {}
    for value in values:
        frequencies[value] = frequencies.get(value, 0) + 1

    total = len(values)
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    top_categories = (
        tuple(
            CategoryFrequency(
                value=value,
                count=count,
                percentage=_round_fraction(Fraction(count * 100, total)),
            )
            for value, count in ranked[:20]
        )
        if total
        else ()
    )

    return CategoricalSummary(
        unique_count=len(frequencies),
        top_categories=top_categories,
    )


def profile_dataset(dataset: CsvDataset) -> DatasetProfile:
    """Profile a CSV dataset without changing its records.

    Empty and whitespace-only cells count as missing. Numeric summaries include
    finite Decimal values only. Other non-missing values count as invalid when
    at least one value in that column can be parsed as a number.
    """
    row_count = len(dataset.records)
    column_profiles: list[ColumnProfile] = []

    for header in dataset.headers:
        values = [record.get(header, "") for record in dataset.records]
        missing_count = sum(not value.strip() for value in values)
        non_missing = [value for value in values if value.strip()]
        parsed = [_parse_decimal(value) for value in non_missing]
        inferred_type = _infer_type(non_missing, parsed)

        column_profiles.append(
            ColumnProfile(
                name=header,
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_percentage=(
                    _round_fraction(Fraction(missing_count * 100, row_count))
                    if row_count
                    else Decimal("0.000000")
                ),
                numeric_summary=_numeric_summary(parsed),
                categorical_summary=_categorical_summary(
                    non_missing,
                    inferred_type,
                ),
            )
        )

    seen_rows: set[tuple[str, ...]] = set()
    duplicate_row_count = 0
    for record in dataset.records:
        row_key = tuple(record.get(header, "") for header in dataset.headers)
        if row_key in seen_rows:
            duplicate_row_count += 1
        else:
            seen_rows.add(row_key)

    return DatasetProfile(
        row_count=row_count,
        column_count=len(dataset.headers),
        duplicate_row_count=duplicate_row_count,
        columns=tuple(column_profiles),
    )
