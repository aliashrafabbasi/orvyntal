"""Deterministic Markdown reports for dataset profiles."""

from orvyntal.data_intelligence.analytics.profiling import DatasetProfile


def _escape(value: object) -> str:
    """Escape values for safe, single-line Markdown table cells."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def render_profile_report(profile: DatasetProfile) -> str:
    """Render a deterministic Markdown report from a dataset profile."""
    lines = [
        "# Dataset Profile Report",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Rows | {profile.row_count} |",
        f"| Columns | {profile.column_count} |",
        f"| Duplicate rows | {profile.duplicate_row_count} |",
        "",
        "## Column Summary",
        "",
        "| Column | Type | Missing | Missing % | Unique | Mean | Median | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for column in profile.columns:
        numeric = column.numeric_summary
        categorical = column.categorical_summary
        lines.append(
            "| "
            + " | ".join(
                _escape(value)
                for value in (
                    column.name,
                    column.inferred_type,
                    column.missing_count,
                    column.missing_percentage,
                    categorical.unique_count if categorical else "-",
                    numeric.mean if numeric else "-",
                    numeric.median if numeric else "-",
                    numeric.minimum if numeric else "-",
                    numeric.maximum if numeric else "-",
                )
            )
            + " |"
        )

    for column in profile.columns:
        categorical = column.categorical_summary
        if categorical is None or not categorical.top_categories:
            continue

        lines.extend(
            [
                "",
                f"## Top Categories: {_escape(column.name)}",
                "",
                "| Category | Count | Percentage |",
                "| --- | ---: | ---: |",
            ]
        )
        for item in categorical.top_categories:
            lines.append(
                "| "
                + " | ".join(
                    _escape(value)
                    for value in (item.value, item.count, item.percentage)
                )
                + " |"
            )

    return "\n".join(lines) + "\n"
