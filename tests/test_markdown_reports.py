"""Tests for deterministic Markdown dataset reports."""

from orvyntal.data_intelligence.analytics.profiling import profile_dataset
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset
from orvyntal.data_intelligence.reports.markdown import render_profile_report


def test_report_contains_dataset_and_column_summaries() -> None:
    dataset = CsvDataset(
        headers=("age", "name"),
        records=(
            {"age": "20", "name": "Ali"},
            {"age": "30", "name": "Sara"},
            {"age": "30", "name": "Ali"},
        ),
    )

    report = render_profile_report(profile_dataset(dataset))

    assert "# Dataset Profile Report" in report
    assert "| Rows | 3 |" in report
    assert "| Columns | 2 |" in report
    assert "| Duplicate rows | 0 |" in report
    assert "| age | integer | 0 |" in report
    assert "## Top Categories: name" in report
    assert "| Ali | 2 | 66.666667 |" in report


def test_report_is_deterministic() -> None:
    dataset = CsvDataset(
        headers=("value",),
        records=({"value": "1"}, {"value": "2"}),
    )
    profile = profile_dataset(dataset)

    assert render_profile_report(profile) == render_profile_report(profile)


def test_report_supports_header_only_dataset() -> None:
    profile = profile_dataset(CsvDataset(headers=("value",), records=()))

    report = render_profile_report(profile)

    assert "| Rows | 0 |" in report
    assert "| Columns | 1 |" in report
    assert "| value | text | 0 |" in report


def test_report_escapes_markdown_table_values() -> None:
    dataset = CsvDataset(
        headers=("category|name",),
        records=(
            {"category|name": "A|B"},
            {"category|name": "A|B"},
            {"category|name": "line\nbreak"},
        ),
    )

    report = render_profile_report(profile_dataset(dataset))

    assert r"category\|name" in report
    assert r"A\|B" in report
    assert "line break" in report
