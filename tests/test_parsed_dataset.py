"""Tests for shared parsed dataset representations."""

from decimal import Decimal

import pytest

from orvyntal.data_intelligence.analytics import parsed
from orvyntal.data_intelligence.analytics.parsed import (
    ParsedColumn,
    parse_dataset,
    scale_decimal_values,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def test_parse_dataset_preserves_column_and_row_order() -> None:
    dataset = CsvDataset(
        headers=("age", "score"),
        records=(
            {"age": "20", "score": "1.5"},
            {"age": "", "score": "invalid"},
            {"age": "30", "score": "2.5"},
        ),
    )

    result = parse_dataset(dataset)

    assert tuple(column.name for column in result.columns) == ("age", "score")
    assert result.by_name["age"].values == (Decimal("20"), None, Decimal("30"))
    assert result.by_name["score"].values == (
        Decimal("1.5"),
        None,
        Decimal("2.5"),
    )


def test_parse_dataset_tracks_missing_invalid_and_valid_counts() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=(
            {"value": "10"},
            {"value": ""},
            {"value": "  "},
            {"value": "invalid"},
            {"value": "NaN"},
            {"value": "20"},
        ),
    )

    column = parse_dataset(dataset).by_name["value"]

    assert column.missing_count == 2
    assert column.invalid_count == 2
    assert column.valid_count == 2


def test_parse_dataset_parses_each_non_empty_cell_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original_parser = parsed.parse_numeric_value

    def tracking_parser(value: str) -> Decimal | None:
        calls.append(value)
        return original_parser(value)

    monkeypatch.setattr(parsed, "parse_numeric_value", tracking_parser)

    dataset = CsvDataset(
        headers=("a", "b"),
        records=(
            {"a": "1", "b": ""},
            {"a": "invalid", "b": "2"},
        ),
    )

    parse_dataset(dataset)

    assert calls == ["1", "invalid", "2"]


def test_parsed_dataset_mapping_is_immutable() -> None:
    dataset = CsvDataset(headers=("value",), records=({"value": "1"},))
    result = parse_dataset(dataset)

    with pytest.raises(TypeError):
        result.by_name["other"] = ParsedColumn(
            name="other",
            values=(Decimal("2"),),
            missing_count=0,
            invalid_count=0,
        )


def test_parse_dataset_handles_empty_records() -> None:
    dataset = CsvDataset(headers=("value",), records=())

    result = parse_dataset(dataset)

    assert len(result.columns) == 1
    assert result.by_name["value"].values == ()
    assert result.by_name["value"].valid_count == 0


def test_parse_dataset_rejects_invalid_dataset_type() -> None:
    with pytest.raises(TypeError, match="dataset must be a CsvDataset"):
        parse_dataset(None)  # type: ignore[arg-type]


def test_parse_dataset_only_parses_requested_columns() -> None:
    dataset = CsvDataset(
        headers=("first", "second", "third"),
        records=({"first": "1", "second": "2", "third": "3"},),
    )

    result = parse_dataset(dataset, ("third", "first"))

    assert tuple(column.name for column in result.columns) == ("first", "third")
    assert result.by_name["first"].values == (Decimal("1"),)
    assert result.by_name["third"].values == (Decimal("3"),)
    assert "second" not in result.by_name


def test_parse_dataset_rejects_duplicate_selected_columns() -> None:
    dataset = CsvDataset(headers=("value",), records=({"value": "1"},))

    with pytest.raises(ValueError, match="must not contain duplicates"):
        parse_dataset(dataset, ("value", "value"))


def test_parse_dataset_rejects_unknown_selected_columns() -> None:
    dataset = CsvDataset(headers=("value",), records=({"value": "1"},))

    with pytest.raises(ValueError, match="unknown columns: missing"):
        parse_dataset(dataset, ("missing",))


def test_parse_dataset_rejects_non_tuple_column_selection() -> None:
    dataset = CsvDataset(headers=("value",), records=({"value": "1"},))

    with pytest.raises(TypeError, match="columns must be a tuple"):
        parse_dataset(dataset, ["value"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        (Decimal("1e999"), Decimal("2e999"), Decimal("-3e999")),
        (Decimal("1.25"), Decimal("-2.50"), Decimal("0")),
        (Decimal("1e1000"), Decimal("1e-1000")),
        (Decimal("-0"), Decimal("0"), Decimal("5")),
    ],
)
def test_scale_decimal_values_preserves_exact_numeric_values(
    values: tuple[Decimal, ...],
) -> None:
    from fractions import Fraction

    scaled, exponent = scale_decimal_values(values)

    assert len(scaled) == len(values)
    for original, integer in zip(values, scaled, strict=True):
        reconstructed = (
            Fraction(integer * 10**exponent)
            if exponent >= 0
            else Fraction(integer, 10**-exponent)
        )
        assert reconstructed == Fraction(original)


def test_scale_decimal_values_handles_empty_input() -> None:
    assert scale_decimal_values(()) == ((), 0)
