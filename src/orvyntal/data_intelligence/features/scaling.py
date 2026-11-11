"""Deterministic numeric feature scaling."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import isqrt

from orvyntal.data_intelligence.features.numeric import NumericFeatureMatrix

_PLACES = 6
_SCALE = 10**_PLACES


class ScalingFeatureError(ValueError):
    """Raised when numeric features cannot be scaled safely."""


@dataclass(frozen=True, slots=True)
class ScaledFeatureMatrix:
    """Immutable scaled features with original row-index tracking."""

    columns: tuple[str, ...]
    values: tuple[tuple[Decimal, ...], ...]
    row_indices: tuple[int, ...]
    dropped_row_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StandardScaler:
    """Fitted population-standardization parameters."""

    columns: tuple[str, ...]
    means: tuple[Fraction, ...]
    variances: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class MinMaxScaler:
    """Fitted minimum and maximum parameters."""

    columns: tuple[str, ...]
    minima: tuple[Fraction, ...]
    maxima: tuple[Fraction, ...]


def _decimal_from_scaled_integer(value: int) -> Decimal:
    """Build a Decimal with exactly six fractional places without context rounding."""
    sign = 1 if value < 0 else 0
    digits = tuple(int(digit) for digit in str(abs(value))) if value else (0,)
    return Decimal((sign, digits, -_PLACES))


def _round_fraction(value: Fraction) -> Decimal:
    """Round an exact rational to six decimal places using half-even."""
    numerator = abs(value.numerator) * _SCALE
    quotient, remainder = divmod(numerator, value.denominator)
    doubled = 2 * remainder

    if doubled > value.denominator or (doubled == value.denominator and quotient % 2):
        quotient += 1

    return _decimal_from_scaled_integer(-quotient if value < 0 else quotient)


def _round_standard_score(delta: Fraction, variance: Fraction) -> Decimal:
    """Round delta / sqrt(variance) exactly to six decimal places."""
    if variance <= 0:
        raise ScalingFeatureError("variance must be positive")

    if delta == 0:
        return _decimal_from_scaled_integer(0)

    numerator = delta.numerator**2 * _SCALE**2 * variance.denominator
    denominator = delta.denominator**2 * variance.numerator

    quotient = isqrt(numerator // denominator)
    left = 4 * numerator
    right = denominator * (2 * quotient + 1) ** 2

    if left > right or (left == right and quotient % 2):
        quotient += 1

    return _decimal_from_scaled_integer(-quotient if delta < 0 else quotient)


def _validate_matrix(matrix: NumericFeatureMatrix) -> None:
    if not isinstance(matrix, NumericFeatureMatrix):
        raise ScalingFeatureError("matrix must be a NumericFeatureMatrix")

    columns = matrix.columns
    if (
        not isinstance(columns, tuple)
        or not columns
        or any(not isinstance(column, str) or not column.strip() for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ScalingFeatureError("matrix contains invalid columns")

    if not isinstance(matrix.values, tuple) or not isinstance(
        matrix.row_indices, tuple
    ):
        raise ScalingFeatureError("matrix values and row indices must be tuples")

    if not isinstance(matrix.dropped_row_indices, tuple):
        raise ScalingFeatureError("dropped row indices must be a tuple")

    if len(matrix.values) != len(matrix.row_indices):
        raise ScalingFeatureError("matrix values and row indices are inconsistent")

    for row in matrix.values:
        if (
            not isinstance(row, tuple)
            or len(row) != len(columns)
            or any(
                not isinstance(value, Decimal) or not value.is_finite() for value in row
            )
        ):
            raise ScalingFeatureError("matrix contains invalid numeric rows")

    for indices in (matrix.row_indices, matrix.dropped_row_indices):
        if any(type(index) is not int or index < 0 for index in indices) or len(
            set(indices)
        ) != len(indices):
            raise ScalingFeatureError("matrix contains invalid row indices")

    if set(matrix.row_indices).intersection(matrix.dropped_row_indices):
        raise ScalingFeatureError("retained and dropped row indices must not overlap")


def _validate_fitted_columns(
    matrix: NumericFeatureMatrix,
    columns: tuple[str, ...],
) -> None:
    if (
        not isinstance(columns, tuple)
        or not columns
        or any(not isinstance(column, str) or not column.strip() for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ScalingFeatureError("fitted scaler contains invalid columns")

    if matrix.columns != columns:
        raise ScalingFeatureError(
            "matrix columns must match the fitted scaler columns and order"
        )


def fit_standard_scaler(matrix: NumericFeatureMatrix) -> StandardScaler:
    """Fit population mean and variance using exact rational arithmetic."""
    _validate_matrix(matrix)

    if not matrix.values:
        raise ScalingFeatureError("at least one row is required to fit a scaler")

    column_count = len(matrix.columns)
    row_count = len(matrix.values)
    sums = [Fraction(0) for _ in range(column_count)]
    sums_of_squares = [Fraction(0) for _ in range(column_count)]

    for row in matrix.values:
        for column_index, value in enumerate(row):
            exact_value = Fraction(value)
            sums[column_index] += exact_value
            sums_of_squares[column_index] += exact_value * exact_value

    means = [total / row_count for total in sums]
    variances = [
        squares / row_count - mean * mean
        for squares, mean in zip(sums_of_squares, means, strict=True)
    ]

    return StandardScaler(
        columns=matrix.columns,
        means=tuple(means),
        variances=tuple(variances),
    )


def transform_standard_features(
    matrix: NumericFeatureMatrix,
    scaler: StandardScaler,
) -> ScaledFeatureMatrix:
    """Apply fitted population standardization without refitting."""
    _validate_matrix(matrix)

    if not isinstance(scaler, StandardScaler):
        raise ScalingFeatureError("scaler must be a fitted StandardScaler")

    _validate_fitted_columns(matrix, scaler.columns)

    if (
        not isinstance(scaler.means, tuple)
        or not isinstance(scaler.variances, tuple)
        or len(scaler.means) != len(scaler.columns)
        or len(scaler.variances) != len(scaler.columns)
        or any(not isinstance(value, Fraction) for value in scaler.means)
        or any(
            not isinstance(value, Fraction) or value < 0 for value in scaler.variances
        )
    ):
        raise ScalingFeatureError("fitted standard scaler parameters are invalid")

    scaled_rows: list[tuple[Decimal, ...]] = []

    for row in matrix.values:
        scaled_row: list[Decimal] = []

        for index, value in enumerate(row):
            variance = scaler.variances[index]

            delta = Fraction(value) - scaler.means[index]

            if variance == 0:
                scaled_row.append(_round_fraction(delta))
                continue

            scaled_row.append(_round_standard_score(delta, variance))

        scaled_rows.append(tuple(scaled_row))

    return ScaledFeatureMatrix(
        columns=matrix.columns,
        values=tuple(scaled_rows),
        row_indices=matrix.row_indices,
        dropped_row_indices=matrix.dropped_row_indices,
    )


def fit_minmax_scaler(matrix: NumericFeatureMatrix) -> MinMaxScaler:
    """Fit exact per-column minimum and maximum values."""
    _validate_matrix(matrix)

    if not matrix.values:
        raise ScalingFeatureError("at least one row is required to fit a scaler")

    minima: list[Fraction | None] = [None] * len(matrix.columns)
    maxima: list[Fraction | None] = [None] * len(matrix.columns)

    for row in matrix.values:
        for index, value in enumerate(row):
            exact_value = Fraction(value)
            current_minimum = minima[index]
            current_maximum = maxima[index]

            if current_minimum is None or exact_value < current_minimum:
                minima[index] = exact_value

            if current_maximum is None or exact_value > current_maximum:
                maxima[index] = exact_value

    fitted_minima = tuple(value for value in minima if value is not None)
    fitted_maxima = tuple(value for value in maxima if value is not None)

    return MinMaxScaler(
        columns=matrix.columns,
        minima=fitted_minima,
        maxima=fitted_maxima,
    )


def transform_minmax_features(
    matrix: NumericFeatureMatrix,
    scaler: MinMaxScaler,
) -> ScaledFeatureMatrix:
    """Apply fitted min-max scaling; unseen values may extrapolate beyond [0, 1]."""
    _validate_matrix(matrix)

    if not isinstance(scaler, MinMaxScaler):
        raise ScalingFeatureError("scaler must be a fitted MinMaxScaler")

    _validate_fitted_columns(matrix, scaler.columns)

    if (
        not isinstance(scaler.minima, tuple)
        or not isinstance(scaler.maxima, tuple)
        or len(scaler.minima) != len(scaler.columns)
        or len(scaler.maxima) != len(scaler.columns)
        or any(not isinstance(value, Fraction) for value in scaler.minima)
        or any(not isinstance(value, Fraction) for value in scaler.maxima)
        or any(
            minimum > maximum
            for minimum, maximum in zip(scaler.minima, scaler.maxima, strict=True)
        )
    ):
        raise ScalingFeatureError("fitted min-max scaler parameters are invalid")

    scaled_rows: list[tuple[Decimal, ...]] = []

    for row in matrix.values:
        scaled_row: list[Decimal] = []

        for index, value in enumerate(row):
            minimum = scaler.minima[index]
            maximum = scaler.maxima[index]

            numerator = Fraction(value) - minimum
            denominator = maximum - minimum

            if denominator == 0:
                scaled_row.append(_round_fraction(numerator))
                continue

            scaled_value = numerator / denominator
            scaled_row.append(_round_fraction(scaled_value))

        scaled_rows.append(tuple(scaled_row))

    return ScaledFeatureMatrix(
        columns=matrix.columns,
        values=tuple(scaled_rows),
        row_indices=matrix.row_indices,
        dropped_row_indices=matrix.dropped_row_indices,
    )
