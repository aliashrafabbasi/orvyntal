"""Isolated process execution for CPU-bound dataset analytics."""

from multiprocessing.connection import Connection

from orvyntal.data_intelligence.analytics.correlation import (
    correlate_parsed_dataset,
)
from orvyntal.data_intelligence.analytics.descriptive import (
    describe_parsed_column,
)
from orvyntal.data_intelligence.analytics.outliers import analyze_parsed_outliers
from orvyntal.data_intelligence.analytics.parsed import parse_dataset
from orvyntal.data_intelligence.analytics.profiling import profile_parsed_dataset
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def run_analytics_worker(
    connection: Connection,
    dataset: CsvDataset,
    include_correlations: bool,
) -> None:
    """Calculate analytics in a child process and return serializable results."""
    try:
        parsed_dataset = parse_dataset(dataset)
        profile = profile_parsed_dataset(parsed_dataset)

        descriptive_statistics = tuple(
            (column.name, describe_parsed_column(parsed_dataset.by_name[column.name]))
            for column in profile.columns
            if column.numeric_summary is not None
        )

        correlations = (
            correlate_parsed_dataset(parsed_dataset) if include_correlations else None
        )
        outliers = analyze_parsed_outliers(parsed_dataset)

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
