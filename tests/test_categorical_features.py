"""Tests for deterministic categorical feature encoding."""

import pytest

from orvyntal.data_intelligence.features.encoding import (
    CategoricalFeatureError,
    fit_categorical_encoder,
    transform_categorical_features,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def test_fit_learns_deterministic_lexicographic_mappings() -> None:
    dataset = CsvDataset(
        headers=("city", "name"),
        records=(
            {"city": "Karachi", "name": "Ali"},
            {"city": "Lahore", "name": "Sara"},
            {"city": "Karachi", "name": "Zara"},
        ),
    )

    encoder = fit_categorical_encoder(dataset, ("name", "city"))

    assert encoder.columns == ("city", "name")
    assert encoder.mappings() == {
        "city": {"Karachi": 0, "Lahore": 1},
        "name": {"Ali": 0, "Sara": 1, "Zara": 2},
    }


def test_mapping_is_independent_of_record_order() -> None:
    first = CsvDataset(
        headers=("city",),
        records=({"city": "Lahore"}, {"city": "Karachi"}),
    )
    second = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"}, {"city": "Lahore"}),
    )

    assert fit_categorical_encoder(first, ("city",)).mappings() == (
        fit_categorical_encoder(second, ("city",)).mappings()
    )


def test_transform_uses_fitted_mapping_and_preserves_row_indexes() -> None:
    training = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"}, {"city": "Lahore"}),
    )
    inference = CsvDataset(
        headers=("city",),
        records=(
            {"city": "Lahore"},
            {"city": "Karachi"},
            {"city": "Lahore"},
        ),
    )

    encoder = fit_categorical_encoder(training, ("city",))
    result = transform_categorical_features(inference, encoder)

    assert result.columns == ("city",)
    assert result.values == ((1,), (0,), (1,))
    assert result.row_indices == (0, 1, 2)


def test_unknown_category_raises_by_default() -> None:
    training = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"},),
    )
    inference = CsvDataset(
        headers=("city",),
        records=({"city": "Lahore"},),
    )

    encoder = fit_categorical_encoder(training, ("city",))

    with pytest.raises(CategoricalFeatureError, match="unknown category"):
        transform_categorical_features(inference, encoder)


def test_ignore_policy_encodes_unknown_category_as_minus_one() -> None:
    training = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"},),
    )
    inference = CsvDataset(
        headers=("city",),
        records=({"city": "Lahore"}, {"city": "Karachi"}),
    )

    encoder = fit_categorical_encoder(training, ("city",))
    result = transform_categorical_features(
        inference,
        encoder,
        unknown_policy="ignore",
    )

    assert result.values == ((-1,), (0,))
    assert result.row_indices == (0, 1)


@pytest.mark.parametrize("value", ["", "  "])
def test_fit_rejects_missing_categories(value: str) -> None:
    dataset = CsvDataset(
        headers=("city",),
        records=({"city": value},),
    )

    with pytest.raises(CategoricalFeatureError, match="missing or invalid"):
        fit_categorical_encoder(dataset, ("city",))


@pytest.mark.parametrize("value", ["", "  "])
def test_transform_rejects_missing_categories(value: str) -> None:
    training = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"},),
    )
    inference = CsvDataset(
        headers=("city",),
        records=({"city": value},),
    )

    encoder = fit_categorical_encoder(training, ("city",))

    with pytest.raises(CategoricalFeatureError, match="missing or invalid"):
        transform_categorical_features(inference, encoder)


def test_unselected_invalid_columns_do_not_affect_encoding() -> None:
    dataset = CsvDataset(
        headers=("city", "score"),
        records=({"city": "Karachi", "score": ""},),
    )

    encoder = fit_categorical_encoder(dataset, ("city",))
    result = transform_categorical_features(dataset, encoder)

    assert result.values == ((0,),)


def test_fit_and_transform_do_not_mutate_input() -> None:
    dataset = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"}, {"city": "Lahore"}),
    )
    original_records = tuple(dict(row) for row in dataset.records)

    encoder = fit_categorical_encoder(dataset, ("city",))
    transform_categorical_features(dataset, encoder)

    assert dataset.records == original_records


def test_empty_dataset_produces_empty_matrix() -> None:
    dataset = CsvDataset(headers=("city",), records=())

    encoder = fit_categorical_encoder(dataset, ("city",))
    result = transform_categorical_features(dataset, encoder)

    assert encoder.mappings() == {"city": {}}
    assert result.values == ()
    assert result.row_indices == ()


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ((), "non-empty tuple"),
        (("city", "city"), "duplicates"),
        (("unknown",), "unknown columns"),
        (("",), "non-empty strings"),
    ],
)
def test_invalid_column_selection_is_rejected(
    columns: tuple[str, ...],
    message: str,
) -> None:
    dataset = CsvDataset(headers=("city",), records=())

    with pytest.raises(CategoricalFeatureError, match=message):
        fit_categorical_encoder(dataset, columns)


def test_non_tuple_column_selection_is_rejected() -> None:
    dataset = CsvDataset(headers=("city",), records=())

    with pytest.raises(CategoricalFeatureError, match="non-empty tuple"):
        fit_categorical_encoder(
            dataset,
            ["city"],  # type: ignore[arg-type]
        )


def test_invalid_unknown_policy_is_rejected() -> None:
    dataset = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"},),
    )
    encoder = fit_categorical_encoder(dataset, ("city",))

    with pytest.raises(CategoricalFeatureError, match="unknown_policy"):
        transform_categorical_features(
            dataset,
            encoder,
            unknown_policy="drop",  # type: ignore[arg-type]
        )


def test_transform_rejects_dataset_with_missing_encoder_column() -> None:
    training = CsvDataset(
        headers=("city",),
        records=({"city": "Karachi"},),
    )
    inference = CsvDataset(headers=("name",), records=())

    encoder = fit_categorical_encoder(training, ("city",))

    with pytest.raises(CategoricalFeatureError, match="header order"):
        transform_categorical_features(inference, encoder)


def test_transform_rejects_non_dataset_input() -> None:
    with pytest.raises(CategoricalFeatureError, match="CsvDataset"):
        transform_categorical_features(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("columns", "categories"),
    [
        ((), ()),
        (("city", "city"), (("city", ("Karachi",)), ("city", ("Lahore",)))),
        (("city",), (("name", ("Karachi",)),)),
        (
            ("city",),
            (
                (
                    "city",
                    ["Karachi"],
                ),
            ),
        ),
        (("city",), (("city", ("Lahore", "Karachi")),)),
        (("city",), (("city", ("Karachi", "Karachi")),)),
        (("city",), (("city", ("",)),)),
    ],
)
def test_transform_rejects_malformed_encoder_definitions(
    columns: tuple[str, ...],
    categories: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    from orvyntal.data_intelligence.features.encoding import CategoricalFeatureEncoder

    dataset = CsvDataset(headers=("city",), records=())
    encoder = CategoricalFeatureEncoder(columns=columns, categories=categories)

    with pytest.raises(CategoricalFeatureError):
        transform_categorical_features(dataset, encoder)
