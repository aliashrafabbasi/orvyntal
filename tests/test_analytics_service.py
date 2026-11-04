"""Tests for isolated and bounded analytics execution."""

import asyncio
import os
import time
from multiprocessing import Pipe
from pathlib import Path

import pytest

from orvyntal.core import analytics_worker
from orvyntal.core.analytics_service import (
    AnalyticsBusyError,
    AnalyticsExecutionError,
    AnalyticsService,
    AnalyticsTimeoutError,
)
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


def _success_worker(connection, dataset, include_correlations):
    """Return a small deterministic result from a child process."""
    connection.send(("ok", (dataset.headers, include_correlations)))
    connection.close()


def _cpu_worker(connection, dataset, include_correlations):
    """Burn CPU for a fixed interval, then return a result."""
    deadline = time.monotonic() + 0.5
    value = 0

    while time.monotonic() < deadline:
        value += 1

    connection.send(("ok", (value > 0,)))
    connection.close()


def _slow_worker(connection, dataset, include_correlations):
    """Remain active until the parent terminates the worker."""
    time.sleep(10)


@pytest.fixture
def dataset():
    return CsvDataset(
        headers=("score",),
        records=({"score": "10"}, {"score": "20"}),
    )


def test_analytics_service_returns_child_process_result(dataset):
    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_success_worker,
        )

        try:
            result = await service.analyze(
                dataset,
                include_correlations=False,
            )
            assert result == (("score",), False)
        finally:
            await service.shutdown()

    asyncio.run(scenario())


def test_cpu_analytics_does_not_block_event_loop(dataset):
    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_cpu_worker,
        )

        try:
            task = asyncio.create_task(
                service.analyze(dataset, include_correlations=False)
            )

            started = time.monotonic()
            await asyncio.wait_for(asyncio.sleep(0.05), timeout=0.25)
            elapsed = time.monotonic() - started

            assert elapsed < 0.25
            assert await task == (True,)
        finally:
            await service.shutdown()

    asyncio.run(scenario())


def test_analytics_service_rejects_jobs_when_capacity_is_full(dataset):
    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_slow_worker,
        )

        first_task = asyncio.create_task(
            service.analyze(dataset, include_correlations=False)
        )

        try:
            for _ in range(100):
                if service._active == 1:
                    break
                await asyncio.sleep(0.01)

            assert service._active == 1

            with pytest.raises(AnalyticsBusyError):
                await service.analyze(
                    dataset,
                    include_correlations=False,
                )
        finally:
            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task
            await service.shutdown()

    asyncio.run(scenario())


def test_analytics_timeout_reclaims_worker_and_releases_capacity(dataset):
    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=1,
            worker_target=_slow_worker,
        )

        try:
            with pytest.raises(AnalyticsTimeoutError):
                await service.analyze(
                    dataset,
                    include_correlations=False,
                )

            assert service._active == 0
            assert service._processes == {}
        finally:
            await service.shutdown()

    asyncio.run(scenario())


def test_worker_skips_correlation_calculation_when_disabled(dataset, monkeypatch):
    def fail_if_called(_dataset):
        raise AssertionError("Correlation calculation should have been skipped.")

    monkeypatch.setattr(
        analytics_worker,
        "correlate_numeric_columns",
        fail_if_called,
    )

    receiver, sender = Pipe(duplex=False)
    try:
        analytics_worker.run_analytics_worker(
            sender,
            dataset,
            include_correlations=False,
        )
        status, result = receiver.recv()
    finally:
        receiver.close()

    assert status == "ok"
    assert result[2] is None


def test_shutdown_terminates_active_worker_and_releases_capacity(dataset):
    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=10,
            worker_target=_slow_worker,
        )

        task = asyncio.create_task(service.analyze(dataset, include_correlations=False))

        try:
            for _ in range(200):
                if service._processes:
                    break
                await asyncio.sleep(0.01)

            assert service._processes, "Worker process did not start."

            await asyncio.wait_for(service.shutdown(), timeout=3)

            with pytest.raises(AnalyticsExecutionError):
                await task

            assert service._active == 0
            assert service._processes == {}

            with pytest.raises(AnalyticsExecutionError):
                await service.analyze(
                    dataset,
                    include_correlations=False,
                )
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await service.shutdown()

    asyncio.run(scenario())


def _pid_worker(connection, dataset, include_correlations):
    """Return the child process ID."""
    connection.send(("ok", (os.getpid(),)))
    connection.close()


def test_successful_worker_process_is_reaped(dataset):
    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_pid_worker,
        )

        try:
            result = await service.analyze(
                dataset,
                include_correlations=False,
            )

            worker_pid = result[0]

            with pytest.raises(ProcessLookupError):
                os.kill(worker_pid, 0)

            assert service._active == 0
            assert service._processes == {}
        finally:
            await service.shutdown()

    asyncio.run(scenario())


def _cleanup_worker(connection, dataset, include_correlations):
    """Send a result, then finish a small cleanup operation."""
    marker = Path(dataset.records[0]["marker"])
    connection.send(("ok", (True,)))
    connection.close()
    time.sleep(0.2)
    marker.write_text("finished")


def test_successful_worker_finishes_graceful_cleanup(tmp_path):
    async def scenario():
        marker = tmp_path / "worker-cleanup.txt"
        dataset = CsvDataset(
            headers=("marker",),
            records=({"marker": str(marker)},),
        )
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_cleanup_worker,
        )

        try:
            assert await service.analyze(
                dataset,
                include_correlations=False,
            ) == (True,)
            assert marker.read_text() == "finished"
        finally:
            await service.shutdown()

    asyncio.run(scenario())
