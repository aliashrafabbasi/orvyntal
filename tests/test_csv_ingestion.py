"""Tests for CSV ingestion."""

import pytest

from orvyntal.data_intelligence.ingestion.csv_reader import (
    CsvDataset,
    CsvIngestionError,
    read_csv,
)


def test_reads_csv_records(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,score\nAli,90\nSara,85\n", encoding="utf-8")

    dataset = read_csv(path)

    assert isinstance(dataset, CsvDataset)
    assert dataset.headers == ("name", "score")
    assert dataset.records == (
        {"name": "Ali", "score": "90"},
        {"name": "Sara", "score": "85"},
    )


def test_reads_utf8_bom(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("\ufeffname\nAli\n", encoding="utf-8")

    assert read_csv(path).headers == ("name",)


def test_missing_file_raises_ingestion_error(tmp_path) -> None:
    with pytest.raises(CsvIngestionError, match="not found"):
        read_csv(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    "content", ["", "\n", "name,name\nAli,90\n", " ,score\nAli,90\n"]
)
def test_invalid_headers_raise_ingestion_error(tmp_path, content: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CsvIngestionError):
        read_csv(path)


@pytest.mark.parametrize("content", ["name,score\nAli\n", "name,score\nAli,90,extra\n"])
def test_inconsistent_rows_raise_ingestion_error(tmp_path, content: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CsvIngestionError):
        read_csv(path)


def test_malformed_quoted_csv_raises_ingestion_error(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text('name,score\n"Ali,90\n', encoding="utf-8")

    with pytest.raises(CsvIngestionError):
        read_csv(path)


def test_preserves_quoted_commas_and_multiline_fields(tmp_path) -> None:
    path = tmp_path / "quoted.csv"
    path.write_text(
        'name,notes\nAli,"Karachi, Pakistan"\nSara,"first line\nsecond line"\n',
        encoding="utf-8",
    )

    dataset = read_csv(path)

    assert dataset.records == (
        {"name": "Ali", "notes": "Karachi, Pakistan"},
        {"name": "Sara", "notes": "first line\nsecond line"},
    )


def test_header_only_csv_returns_no_records(tmp_path) -> None:
    path = tmp_path / "header_only.csv"
    path.write_text("name,score\n", encoding="utf-8")

    dataset = read_csv(path)

    assert dataset.headers == ("name", "score")
    assert dataset.records == ()


def test_ignores_blank_physical_lines(tmp_path) -> None:
    path = tmp_path / "blank_lines.csv"
    path.write_text("\nname,score\n\nAli,90\n\n", encoding="utf-8")

    dataset = read_csv(path)

    assert dataset.records == ({"name": "Ali", "score": "90"},)


def test_preserves_original_header_whitespace(tmp_path) -> None:
    path = tmp_path / "headers.csv"
    path.write_text(" name ,name\nAli,90\n", encoding="utf-8")

    dataset = read_csv(path)

    assert dataset.headers == (" name ", "name")
    assert dataset.records == ({" name ": "Ali", "name": "90"},)


def test_rejects_whitespace_only_header(tmp_path) -> None:
    path = tmp_path / "blank_header.csv"
    path.write_text("name,   \nAli,90\n", encoding="utf-8")

    with pytest.raises(CsvIngestionError, match="must not be blank"):
        read_csv(path)
