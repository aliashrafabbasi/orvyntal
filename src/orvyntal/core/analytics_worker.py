"""Isolated process execution for CPU-bound dataset analytics."""

from multiprocessing.connection import Connection

from orvyntal.data_intelligence.analytics.correlation import (
    correlate_numeric_columns,
)
from orvyntal.data_intelligence.analytics.descriptive import (
    describe_numeric_values,
)
from orvyntal.data_intelligence.analytics.outliers import (
    analyze_numeric_outliers,
)
from orvyntal.data_intelligence.analytics.profiling import profile_dataset
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def run_analytics_worker(
    connection: Connection,
    dataset: CsvDataset,
    include_correlations: bool,
) -> None:
    """Calculate analytics in a child process and return serializable results."""
    try:
        profile = profile_dataset(dataset)

        descriptive_statistics = tuple(
            (
                column.name,
                describe_numeric_values(
                    record.get(column.name, "") for record in dataset.records
                ),
            )
            for column in profile.columns
            if column.numeric_summary is not None
        )

        correlations = (
            correlate_numeric_columns(dataset) if include_correlations else None
        )
        outliers = analyze_numeric_outliers(dataset)

        connection.send(
            ("ok", (profile, descriptive_statistics, correlations, outliers))
        )
    except Exception:
        # Do not send exception messages or internal paths to the API process.
        try:
            connection.send(("error",))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
