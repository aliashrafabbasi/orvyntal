"""Opt-in performance regression benchmarks for analytics and numeric scaling."""

import json
import subprocess
import sys
from statistics import median
from time import perf_counter, sleep

import pytest

from orvyntal.data_intelligence.analytics.correlation import (
    correlate_parsed_dataset,
)
from orvyntal.data_intelligence.analytics.descriptive import (
    describe_parsed_column,
)
from orvyntal.data_intelligence.analytics.outliers import analyze_parsed_outliers
from orvyntal.data_intelligence.analytics.parsed import parse_dataset
from orvyntal.data_intelligence.analytics.profiling import profile_parsed_dataset
from orvyntal.data_intelligence.features.numeric import extract_numeric_features
from orvyntal.data_intelligence.features.scaling import (
    fit_minmax_scaler,
    fit_standard_scaler,
    transform_minmax_features,
    transform_standard_features,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def make_analytics_dataset(
    rows: int,
    *,
    extreme: bool,
    columns: int = 2,
) -> CsvDataset:
    if columns < 1:
        raise ValueError("columns must be positive")

    headers = tuple(f"column_{index}" for index in range(columns))
    records = []

    for row_index in range(rows):
        row = {}

        for column_index, header in enumerate(headers):
            value = (row_index + column_index) % 9 + 1
            raw_value = f"{value}e999" if extreme else str(value)
            row[header] = raw_value

        records.append(row)

    return CsvDataset(headers=headers, records=tuple(records))


def run_analytics_pipeline(dataset: CsvDataset) -> None:
    parsed = parse_dataset(dataset)
    profile_parsed_dataset(parsed)

    for column in parsed.columns:
        describe_parsed_column(column)

    correlate_parsed_dataset(parsed)
    analyze_parsed_outliers(parsed)


def measure(function, *, repetitions: int = 2) -> float:
    durations = []

    for _ in range(repetitions):
        started = perf_counter()
        function()
        durations.append(perf_counter() - started)

    return median(durations)


@pytest.mark.slow
def test_extreme_exponents_do_not_disproportionately_slow_analytics() -> None:
    """100k cells with extreme exponents should stay within 2x normal runtime."""
    normal = make_analytics_dataset(50_000, extreme=False)
    extreme = make_analytics_dataset(50_000, extreme=True)

    normal_seconds = measure(lambda: run_analytics_pipeline(normal))
    extreme_seconds = measure(lambda: run_analytics_pipeline(extreme))

    ratio = extreme_seconds / normal_seconds

    print(
        "\nAnalytics benchmark (100k cells): "
        f"normal={normal_seconds:.3f}s, "
        f"extreme={extreme_seconds:.3f}s, "
        f"ratio={ratio:.2f}x"
    )

    assert extreme_seconds <= normal_seconds * 2.0, (
        f"Extreme-exponent analytics took {ratio:.2f}x normal runtime "
        f"(normal={normal_seconds:.3f}s, extreme={extreme_seconds:.3f}s)"
    )


def make_scaling_dataset(rows: int) -> CsvDataset:
    return CsvDataset(
        headers=("value",),
        records=tuple({"value": str(index % 9 + 1)} for index in range(rows)),
    )


def run_scaling_pipeline(dataset: CsvDataset) -> None:
    matrix = extract_numeric_features(dataset, ("value",))

    standard = fit_standard_scaler(matrix)
    transform_standard_features(matrix, standard)

    minmax = fit_minmax_scaler(matrix)
    transform_minmax_features(matrix, minmax)


@pytest.mark.slow
def test_numeric_scaling_runtime_scales_with_input_size() -> None:
    """A 10x row increase should not cause more than 15x runtime."""
    small = make_scaling_dataset(10_000)
    large = make_scaling_dataset(100_000)

    small_seconds = measure(lambda: run_scaling_pipeline(small))
    large_seconds = measure(lambda: run_scaling_pipeline(large))

    ratio = large_seconds / small_seconds

    print(
        "\nNumeric scaling benchmark: "
        f"10k={small_seconds:.3f}s, "
        f"100k={large_seconds:.3f}s, "
        f"ratio={ratio:.2f}x"
    )

    assert large_seconds <= small_seconds * 15.0, (
        f"10x more rows took {ratio:.2f}x runtime "
        f"(10k={small_seconds:.3f}s, 100k={large_seconds:.3f}s)"
    )


def measure_isolated_analytics(rows: int, columns: int) -> dict[str, float | None]:
    """Measure analytics in a fresh process to isolate peak RSS per workload."""
    child_code = r"""
import json
import sys
import tracemalloc

from time import perf_counter

from orvyntal.data_intelligence.analytics.correlation import correlate_parsed_dataset
from orvyntal.data_intelligence.analytics.descriptive import describe_parsed_column
from orvyntal.data_intelligence.analytics.outliers import analyze_parsed_outliers
from orvyntal.data_intelligence.analytics.parsed import parse_dataset
from orvyntal.data_intelligence.analytics.profiling import profile_parsed_dataset
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset

rows = int(sys.argv[1])
columns = int(sys.argv[2])
headers = tuple(f"column_{index}" for index in range(columns))
records = tuple(
    {
        header: str((row_index + column_index) % 9 + 1)
        for column_index, header in enumerate(headers)
    }
    for row_index in range(rows)
)
dataset = CsvDataset(headers=headers, records=records)

def run_pipeline():
    parsed = parse_dataset(dataset)
    profile_parsed_dataset(parsed)
    for column in parsed.columns:
        describe_parsed_column(column)
    correlate_parsed_dataset(parsed)
    analyze_parsed_outliers(parsed)

started = perf_counter()
run_pipeline()
runtime_seconds = perf_counter() - started
tracemalloc.start()
try:
    run_pipeline()
    _, peak_pipeline_python_bytes = tracemalloc.get_traced_memory()
finally:
    tracemalloc.stop()

print(json.dumps({
    "runtime_seconds": runtime_seconds,
    "peak_pipeline_python_mib": peak_pipeline_python_bytes / (1024 * 1024),
}))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(rows), str(columns)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    peak_sampled_rss_mib = None
    communicated = False

    try:
        while process.poll() is None:
            try:
                with open(f"/proc/{process.pid}/status", encoding="utf-8") as status:
                    for line in status:
                        if line.startswith("VmRSS:"):
                            rss_mib = int(line.split()[1]) / 1024
                            peak_sampled_rss_mib = max(
                                peak_sampled_rss_mib or 0.0, rss_mib
                            )
                            break
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
            sleep(0.005)

        stdout, stderr = process.communicate()
        communicated = True

        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, process.args, output=stdout, stderr=stderr
            )

        metrics = json.loads(stdout)
        metrics["peak_sampled_rss_mib"] = peak_sampled_rss_mib
        return metrics
    finally:
        if not communicated:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.communicate()


@pytest.mark.slow
def test_analytics_pipeline_handles_10k_rows_and_20_columns() -> None:
    """Measure and enforce resource ceilings for the 200k-cell workload."""
    metrics = measure_isolated_analytics(10_000, 20)

    rss = metrics["peak_sampled_rss_mib"]
    rss_text = f"{rss:.2f} MiB" if rss is not None else "unavailable"
    print(
        "\nWide analytics benchmark (10k x 20 = 200k cells): "
        f"runtime={metrics['runtime_seconds']:.3f}s, "
        f"peak_pipeline_python_memory={metrics['peak_pipeline_python_mib']:.2f} MiB, "
        f"peak_sampled_rss={rss_text}"
    )

    assert metrics["peak_pipeline_python_mib"] <= 40.0, (
        f"20-column workload exceeded Python memory ceiling: "
        f"{metrics['peak_pipeline_python_mib']:.2f} MiB > 40 MiB"
    )
    if rss is not None:
        assert rss <= 90.0, (
            f"20-column workload exceeded RSS ceiling: {rss:.2f} MiB > 90 MiB"
        )


@pytest.mark.slow
def test_analytics_pipeline_handles_10k_rows_and_100_columns() -> None:
    """Measure and enforce resource ceilings for the 1m-cell workload."""
    metrics = measure_isolated_analytics(10_000, 100)

    rss = metrics["peak_sampled_rss_mib"]
    rss_text = f"{rss:.2f} MiB" if rss is not None else "unavailable"
    print(
        "\nWide analytics benchmark (10k x 100 = 1m cells): "
        f"runtime={metrics['runtime_seconds']:.3f}s, "
        f"peak_pipeline_python_memory={metrics['peak_pipeline_python_mib']:.2f} MiB, "
        f"peak_sampled_rss={rss_text}"
    )

    assert metrics["peak_pipeline_python_mib"] <= 200.0, (
        f"100-column workload exceeded Python memory ceiling: "
        f"{metrics['peak_pipeline_python_mib']:.2f} MiB > 200 MiB"
    )
    if rss is not None:
        assert rss <= 350.0, (
            f"100-column workload exceeded RSS ceiling: {rss:.2f} MiB > 350 MiB"
        )
