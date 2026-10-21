"""Deterministic descriptive statistics for numeric CSV values."""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import isqrt

from orvyntal.data_intelligence.analytics.numeric import parse_numeric_value

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

    if not valid:
        return None

    exact = sorted(Fraction(number) for number in valid)
    count = len(exact)
    total = sum(exact, Fraction(0))
    mean = total / count
    midpoint = count // 2

    if count % 2:
        median = exact[midpoint]
    else:
        median = (exact[midpoint - 1] + exact[midpoint]) / 2

    population_variance = (
        sum(((number - mean) ** 2 for number in exact), Fraction(0)) / count
    )
    sample_variance = (
        sum(((number - mean) ** 2 for number in exact), Fraction(0)) / (count - 1)
        if count >= 2
        else None
    )

    minimum = valid[0]
    maximum = valid[0]
    for number in valid[1:]:
        if number < minimum:
            minimum = number
        if number > maximum:
            maximum = number

    return DescriptiveStatistics(
        count=count,
        invalid_count=invalid_count,
        total=_round_fraction(total),
        mean=_round_fraction(mean),
        median=_round_fraction(median),
        minimum=minimum,
        maximum=maximum,
        value_range=_round_fraction(exact[-1] - exact[0]),
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
