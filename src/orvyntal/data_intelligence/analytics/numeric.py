"""Shared numeric parsing policy for dataset analytics."""

from decimal import Decimal, InvalidOperation

MIN_ADJUSTED_EXPONENT = -1000
MAX_ADJUSTED_EXPONENT = 1000
MAX_COEFFICIENT_DIGITS = 1000


def parse_numeric_value(value: str) -> Decimal | None:
    """Parse a finite decimal within the supported exponent range.

    Values outside the range are treated as invalid for analytics.
    """
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return None

    if not number.is_finite():
        return None

    if len(number.as_tuple().digits) > MAX_COEFFICIENT_DIGITS:
        return None

    if not (MIN_ADJUSTED_EXPONENT <= number.adjusted() <= MAX_ADJUSTED_EXPONENT):
        return None

    return number
