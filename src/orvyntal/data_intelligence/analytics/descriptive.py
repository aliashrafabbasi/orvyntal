"""Deterministic descriptive statistics for numeric CSV values."""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import isqrt

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value
from orvyntal.data_intelligence.analytics.parsed import (
    ParsedColumn,
    scale_decimal_values,
)

_PLACES = 6
_SCALE = 10**_PLACES


@dataclass(frozen=True, slots=True)
class DescriptiveStatistics:
    """Descriptive statistics for valid finite numeric observations."""

    count: int
    invalid_count: int
    total: Decimal
    mean: Decimal
    median: Decimal
    minimum: Decimal
    maximum: Decimal
    value_range: Decimal
    population_variance: Decimal
    population_standard_deviation: Decimal
    sample_variance: Decimal | None
    sample_standard_deviation: Decimal | None


def _parse_number(value: str) -> Decimal | None:
    """Parse a policy-compliant finite decimal value."""
    return parse_numeric_value(value)


def _decimal_from_scaled_integer(value: int) -> Decimal:
    """Create an exact, compact Decimal without context-dependent rounding."""
    if value == 0:
        return Decimal("0.000000")

    sign = 1 if value < 0 else 0
    coefficient = abs(value)
    exponent = -_PLACES

    while coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1

    digits = tuple(int(digit) for digit in str(coefficient))
    return Decimal((sign, digits, exponent))


def _round_fraction(value: Fraction) -> Decimal:
    """Round an exact rational to six decimal places using half-even."""
    numerator = abs(value.numerator) * _SCALE
    quotient, remainder = divmod(numerator, value.denominator)
    doubled = 2 * remainder

    if doubled > value.denominator or (doubled == value.denominator and quotient % 2):
        quotient += 1

    return _decimal_from_scaled_integer(-quotient if value < 0 else quotient)


def _round_sqrt_fraction(value: Fraction) -> Decimal:
    """Round the square root of a non-negative rational to six places."""
    if value < 0:
        raise ValueError("Square root requires a non-negative value.")

    scaled_numerator = value.numerator * _SCALE**2
    floor_root = isqrt(scaled_numerator // value.denominator)

    # Compare sqrt(value) * SCALE with floor_root + 0.5 exactly.
    left = 4 * scaled_numerator
    right = value.denominator * (2 * floor_root + 1) ** 2
    if left > right or (left == right and floor_root % 2):
        floor_root += 1

    return _decimal_from_scaled_integer(floor_root)


def _scaled_fraction(
    numerator: int,
    denominator: int,
    exponent: int,
) -> Fraction:
    """Build an exact rational from an integer and a decimal exponent."""
    if exponent >= 0:
        return Fraction(numerator * 10**exponent, denominator)
    return Fraction(numerator, denominator * 10**-exponent)


def _describe_parsed_values(
    valid: list[Decimal],
    invalid_count: int,
) -> DescriptiveStatistics | None:
    """Calculate exact statistics using shared-scale integer arithmetic."""
    if not valid:
        return None

    ordered = sorted(valid)
    scaled, exponent = scale_decimal_values(ordered)
    count = len(scaled)
    scaled_total = sum(scaled)
    total = _scaled_fraction(scaled_total, 1, exponent)
    mean = _scaled_fraction(scaled_total, count, exponent)

    midpoint = count // 2
    if count % 2:
        median = _scaled_fraction(scaled[midpoint], 1, exponent)
    else:
        median = _scaled_fraction(
            scaled[midpoint - 1] + scaled[midpoint],
            2,
            exponent,
        )

    squared_deviation_sum = sum(
        (number * count - scaled_total) ** 2 for number in scaled
    )
    population_variance = _scaled_fraction(
        squared_deviation_sum,
        count**3,
        exponent * 2,
    )
    sample_variance = (
        _scaled_fraction(
            squared_deviation_sum,
            count**2 * (count - 1),
            exponent * 2,
        )
        if count >= 2
        else None
    )

    value_range = _scaled_fraction(
        scaled[-1] - scaled[0],
        1,
        exponent,
    )

    return DescriptiveStatistics(
        count=count,
        invalid_count=invalid_count,
        total=_round_fraction(total),
        mean=_round_fraction(mean),
        median=_round_fraction(median),
        minimum=ordered[0],
        maximum=ordered[-1],
        value_range=_round_fraction(value_range),
        population_variance=_round_fraction(population_variance),
        population_standard_deviation=_round_sqrt_fraction(population_variance),
        sample_variance=(
            _round_fraction(sample_variance) if sample_variance is not None else None
        ),
        sample_standard_deviation=(
            _round_sqrt_fraction(sample_variance)
            if sample_variance is not None
            else None
        ),
    )


def describe_parsed_column(
    column: ParsedColumn,
) -> DescriptiveStatistics | None:
    """Describe a column using its shared parsed representation."""
    valid = [number for number in column.values if number is not None]
    return _describe_parsed_values(valid, column.invalid_count)


def describe_numeric_values(
    values: Iterable[str],
) -> DescriptiveStatistics | None:
    """Describe finite numeric observations from strings.

    Empty and whitespace-only values are missing and are ignored. Invalid or
    non-finite non-empty values are counted separately. If no valid numeric
    observations exist, return None. Sample statistics require at least two
    observations; population statistics require at least one.
    """
    valid: list[Decimal] = []
    invalid_count = 0

    for value in values:
        if not value.strip():
            continue
        number = _parse_number(value)
        if number is None:
            invalid_count += 1
        else:
            valid.append(number)

    return _describe_parsed_values(valid, invalid_count)
