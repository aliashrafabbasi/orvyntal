"""Tests for deterministic numeric feature scaling."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from orvyntal.data_intelligence.features.numeric import NumericFeatureMatrix
from orvyntal.data_intelligence.features.scaling import (
    ScaledFeatureMatrix,
    ScalingFeatureError,
    fit_minmax_scaler,
    fit_standard_scaler,
    transform_minmax_features,
    transform_standard_features,
)


def make_matrix(
    values: tuple[tuple[str, ...], ...],
    *,
    columns: tuple[str, ...] = ("score",),
    row_indices: tuple[int, ...] | None = None,
    dropped_row_indices: tuple[int, ...] = (),
) -> NumericFeatureMatrix:
    """Build a numeric matrix for focused scaler tests."""
    return NumericFeatureMatrix(
        columns=columns,
        values=tuple(tuple(Decimal(value) for value in row) for row in values),
        row_indices=(tuple(range(len(values))) if row_indices is None else row_indices),
        dropped_row_indices=dropped_row_indices,
    )


def test_standard_scaler_uses_population_variance() -> None:
    matrix = make_matrix((("1",), ("2",), ("3",)))

    scaler = fit_standard_scaler(matrix)
    result = transform_standard_features(matrix, scaler)

    assert scaler.means == (Fraction(2),)
    assert scaler.variances == (Fraction(2, 3),)
    assert result.values == (
        (Decimal("-1.224745"),),
        (Decimal("0.000000"),),
        (Decimal("1.224745"),),
    )


def test_standard_scaler_handles_negative_values() -> None:
    matrix = make_matrix((("-3",), ("-2",), ("-1",)))

    result = transform_standard_features(
        matrix,
        fit_standard_scaler(matrix),
    )

    assert result.values == (
        (Decimal("-1.224745"),),
        (Decimal("0.000000"),),
        (Decimal("1.224745"),),
    )


def test_standard_scaler_maps_constant_columns_to_zero() -> None:
    matrix = make_matrix(
        (("7", "2"), ("7", "4"), ("7", "6")),
        columns=("constant", "score"),
    )

    result = transform_standard_features(
        matrix,
        fit_standard_scaler(matrix),
    )

    assert result.values == (
        (Decimal("0.000000"), Decimal("-1.224745")),
        (Decimal("0.000000"), Decimal("0.000000")),
        (Decimal("0.000000"), Decimal("1.224745")),
    )


def test_standard_scaler_preserves_unseen_constant_column_deviation() -> None:
    training = make_matrix((("5",), ("5",), ("5",)))
    unseen = make_matrix((("7",),))

    result = transform_standard_features(
        unseen,
        fit_standard_scaler(training),
    )

    assert result.values == ((Decimal("2.000000"),),)


def test_standard_scaler_handles_very_small_variance() -> None:
    matrix = make_matrix(
        (("0",), ("1e-1000",)),
    )

    result = transform_standard_features(
        matrix,
        fit_standard_scaler(matrix),
    )

    assert result.values == (
        (Decimal("-1.000000"),),
        (Decimal("1.000000"),),
    )


def test_standard_scaler_handles_very_large_values() -> None:
    matrix = make_matrix(
        (("1e1000",), ("2e1000",)),
    )

    result = transform_standard_features(
        matrix,
        fit_standard_scaler(matrix),
    )

    assert result.values == (
        (Decimal("-1.000000"),),
        (Decimal("1.000000"),),
    )


def test_standard_scaler_preserves_row_metadata() -> None:
    matrix = make_matrix(
        (("10",), ("30",)),
        row_indices=(0, 3),
        dropped_row_indices=(1, 2),
    )

    result = transform_standard_features(
        matrix,
        fit_standard_scaler(matrix),
    )

    assert result.row_indices == (0, 3)
    assert result.dropped_row_indices == (1, 2)


def test_standard_scaler_transform_uses_fitted_parameters() -> None:
    training = make_matrix((("0",), ("10",)))
    unseen = make_matrix((("5",), ("20",)))

    scaler = fit_standard_scaler(training)
    result = transform_standard_features(unseen, scaler)

    assert result.values == (
        (Decimal("0.000000"),),
        (Decimal("3.000000"),),
    )


def test_minmax_scaler_maps_training_values_to_unit_interval() -> None:
    matrix = make_matrix((("10",), ("20",), ("30",)))

    result = transform_minmax_features(
        matrix,
        fit_minmax_scaler(matrix),
    )

    assert result.values == (
        (Decimal("0.000000"),),
        (Decimal("0.500000"),),
        (Decimal("1.000000"),),
    )


def test_minmax_scaler_handles_negative_values() -> None:
    matrix = make_matrix((("-10",), ("0",), ("10",)))

    result = transform_minmax_features(
        matrix,
        fit_minmax_scaler(matrix),
    )

    assert result.values == (
        (Decimal("0.000000"),),
        (Decimal("0.500000"),),
        (Decimal("1.000000"),),
    )


def test_minmax_scaler_maps_constant_columns_to_zero() -> None:
    matrix = make_matrix((("5",), ("5",), ("5",)))

    result = transform_minmax_features(
        matrix,
        fit_minmax_scaler(matrix),
    )

    assert result.values == (
        (Decimal("0.000000"),),
        (Decimal("0.000000"),),
        (Decimal("0.000000"),),
    )


def test_minmax_scaler_preserves_unseen_constant_column_deviation() -> None:
    training = make_matrix((("5",), ("5",), ("5",)))
    unseen = make_matrix((("7",),))

    result = transform_minmax_features(
        unseen,
        fit_minmax_scaler(training),
    )

    assert result.values == ((Decimal("2.000000"),),)


def test_minmax_scaler_preserves_column_order_and_row_metadata() -> None:
    matrix = make_matrix(
        (("10", "100"), ("20", "200")),
        columns=("age", "score"),
        row_indices=(2, 5),
        dropped_row_indices=(0, 1, 3, 4),
    )

    result = transform_minmax_features(
        matrix,
        fit_minmax_scaler(matrix),
    )

    assert result.columns == ("age", "score")
    assert result.row_indices == (2, 5)
    assert result.dropped_row_indices == (0, 1, 3, 4)


def test_minmax_scaler_transform_uses_fitted_parameters() -> None:
    training = make_matrix((("0",), ("10",)))
    unseen = make_matrix((("5",), ("20",)))

    scaler = fit_minmax_scaler(training)
    result = transform_minmax_features(unseen, scaler)

    assert result.values == (
        (Decimal("0.500000"),),
        (Decimal("2.000000"),),
    )


def test_minmax_scaler_rounds_half_even() -> None:
    matrix = make_matrix(
        (("0",), ("1",), ("3",), ("2000000",)),
    )

    result = transform_minmax_features(
        matrix,
        fit_minmax_scaler(matrix),
    )

    assert result.values == (
        (Decimal("0.000000"),),
        (Decimal("0.000000"),),
        (Decimal("0.000002"),),
        (Decimal("1.000000"),),
    )


def test_minmax_scaler_handles_very_small_ranges() -> None:
    matrix = make_matrix((("0",), ("1e-1000",)))

    result = transform_minmax_features(
        matrix,
        fit_minmax_scaler(matrix),
    )

    assert result.values == (
        (Decimal("0.000000"),),
        (Decimal("1.000000"),),
    )


def test_scalers_do_not_mutate_input_matrix() -> None:
    matrix = make_matrix((("1",), ("2",), ("3",)))
    original = matrix.values

    transform_standard_features(matrix, fit_standard_scaler(matrix))
    transform_minmax_features(matrix, fit_minmax_scaler(matrix))

    assert matrix.values == original


@pytest.mark.parametrize(
    "values",
    [
        (),
    ],
)
def test_scalers_reject_empty_training_matrices(
    values: tuple[tuple[str, ...], ...],
) -> None:
    matrix = make_matrix(values)

    with pytest.raises(ScalingFeatureError, match="at least one row"):
        fit_standard_scaler(matrix)

    with pytest.raises(ScalingFeatureError, match="at least one row"):
        fit_minmax_scaler(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        object(),
    ],
)
def test_scalers_reject_non_numeric_matrices(matrix: object) -> None:
    with pytest.raises(ScalingFeatureError, match="NumericFeatureMatrix"):
        fit_standard_scaler(matrix)  # type: ignore[arg-type]

    with pytest.raises(ScalingFeatureError, match="NumericFeatureMatrix"):
        fit_minmax_scaler(matrix)  # type: ignore[arg-type]


def test_scalers_reject_malformed_numeric_rows() -> None:
    matrix = NumericFeatureMatrix(
        columns=("score",),
        values=((Decimal("1"), Decimal("2")),),
        row_indices=(0,),
        dropped_row_indices=(),
    )

    with pytest.raises(ScalingFeatureError, match="invalid numeric rows"):
        fit_standard_scaler(matrix)


def test_scalers_reject_non_finite_values() -> None:
    matrix = NumericFeatureMatrix(
        columns=("score",),
        values=((Decimal("NaN"),),),
        row_indices=(0,),
        dropped_row_indices=(),
    )

    with pytest.raises(ScalingFeatureError, match="invalid numeric rows"):
        fit_minmax_scaler(matrix)


def test_scalers_reject_inconsistent_row_indices() -> None:
    matrix = NumericFeatureMatrix(
        columns=("score",),
        values=((Decimal("1"),), (Decimal("2"),)),
        row_indices=(0,),
        dropped_row_indices=(),
    )

    with pytest.raises(ScalingFeatureError, match="inconsistent"):
        fit_standard_scaler(matrix)


def test_transform_rejects_different_column_order() -> None:
    training = make_matrix(
        (("1", "10"), ("2", "20")),
        columns=("age", "score"),
    )
    reordered = make_matrix(
        (("10", "1"), ("20", "2")),
        columns=("score", "age"),
    )

    standard_scaler = fit_standard_scaler(training)
    minmax_scaler = fit_minmax_scaler(training)

    with pytest.raises(ScalingFeatureError, match="columns must match"):
        transform_standard_features(reordered, standard_scaler)

    with pytest.raises(ScalingFeatureError, match="columns must match"):
        transform_minmax_features(reordered, minmax_scaler)


def test_standard_transform_rejects_invalid_fitted_parameters() -> None:
    matrix = make_matrix((("1",), ("2",)))
    scaler = fit_standard_scaler(matrix)
    malformed = replace(scaler, variances=(Fraction(-1),))

    with pytest.raises(ScalingFeatureError, match="parameters are invalid"):
        transform_standard_features(matrix, malformed)


def test_minmax_transform_rejects_invalid_fitted_parameters() -> None:
    matrix = make_matrix((("1",), ("2",)))
    scaler = fit_minmax_scaler(matrix)
    malformed = replace(
        scaler,
        minima=(Fraction(2),),
        maxima=(Fraction(1),),
    )

    with pytest.raises(ScalingFeatureError, match="parameters are invalid"):
        transform_minmax_features(matrix, malformed)


@pytest.mark.parametrize(
    "columns",
    [
        ["score"],
        (),
        ("",),
        ("score", "score"),
    ],
)
def test_standard_transform_rejects_invalid_fitted_columns(columns: object) -> None:
    matrix = make_matrix((("1",), ("2",)))
    scaler = replace(fit_standard_scaler(matrix), columns=columns)

    with pytest.raises(ScalingFeatureError, match="invalid columns"):
        transform_standard_features(matrix, scaler)


@pytest.mark.parametrize(
    "columns",
    [
        ["score"],
        (),
        ("",),
        ("score", "score"),
    ],
)
def test_minmax_transform_rejects_invalid_fitted_columns(columns: object) -> None:
    matrix = make_matrix((("1",), ("2",)))
    scaler = replace(fit_minmax_scaler(matrix), columns=columns)

    with pytest.raises(ScalingFeatureError, match="invalid columns"):
        transform_minmax_features(matrix, scaler)


def test_transform_rejects_wrong_scaler_type() -> None:
    matrix = make_matrix((("1",), ("2",)))

    with pytest.raises(ScalingFeatureError, match="StandardScaler"):
        transform_standard_features(matrix, object())  # type: ignore[arg-type]

    with pytest.raises(ScalingFeatureError, match="MinMaxScaler"):
        transform_minmax_features(matrix, object())  # type: ignore[arg-type]


def test_transform_returns_scaled_feature_matrix() -> None:
    matrix = make_matrix((("1",), ("2",)))

    result = transform_minmax_features(
        matrix,
        fit_minmax_scaler(matrix),
    )

    assert isinstance(result, ScaledFeatureMatrix)


def test_fitted_scalers_are_immutable() -> None:
    matrix = make_matrix((("1",), ("2",)))
    standard = fit_standard_scaler(matrix)
    minmax = fit_minmax_scaler(matrix)

    with pytest.raises((AttributeError, TypeError)):
        standard.means = (Fraction(0),)  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        minmax.minima = (Fraction(0),)  # type: ignore[misc]
