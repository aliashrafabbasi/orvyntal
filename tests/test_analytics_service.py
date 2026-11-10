"""Tests for isolated and bounded analytics execution."""

import asyncio
import os
import threading
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


def test_process_start_does_not_hold_service_condition(dataset, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class BlockingProcess:
        def __init__(self, target, args, daemon):
            self.args = args
            self.pid = None

        def start(self):
            started.set()
            if not release.wait(timeout=2):
                raise RuntimeError("Timed out waiting to release process startup.")

            connection = self.args[0]
            connection.send(("ok", (("score",), False)))
            connection.close()
            self.pid = os.getpid()

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def close(self):
            pass

    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_success_worker,
        )
        monkeypatch.setattr(service._context, "Process", BlockingProcess)

        task = asyncio.create_task(service.analyze(dataset, include_correlations=False))

        try:
            assert await asyncio.to_thread(started.wait, 2), (
                "Process startup did not begin."
            )

            assert service._condition.acquire(blocking=False), (
                "The service condition is held during process startup."
            )
            service._condition.release()

            with pytest.raises(AnalyticsBusyError):
                await service.analyze(dataset, include_correlations=False)
        finally:
            release.set()
            try:
                await task
            finally:
                await service.shutdown()

    asyncio.run(scenario())


def test_shutdown_during_process_start_reaps_unregistered_process(dataset, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    process_closed = threading.Event()

    class BlockingProcess:
        def __init__(self, target, args, daemon):
            self.pid = None

        def start(self):
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Timed out waiting to release process startup.")
            self.pid = os.getpid()

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def close(self):
            process_closed.set()

    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_success_worker,
        )
        monkeypatch.setattr(service._context, "Process", BlockingProcess)

        analyze_task = asyncio.create_task(
            service.analyze(dataset, include_correlations=False)
        )
        shutdown_task = None

        try:
            assert await asyncio.to_thread(started.wait, 2), (
                "Process startup did not begin."
            )

            shutdown_task = asyncio.create_task(service.shutdown())

            for _ in range(200):
                if service._closed:
                    break
                await asyncio.sleep(0.01)

            assert service._closed, "Shutdown did not close the service."

            release.set()

            with pytest.raises(AnalyticsExecutionError):
                await asyncio.wait_for(analyze_task, timeout=3)

            await asyncio.wait_for(shutdown_task, timeout=3)

            assert process_closed.is_set(), "Started process was not closed."
            assert service._processes == {}, "Process registry was not cleared."
            assert service._active == 0, "Analytics slot was not released."
        finally:
            release.set()
            if not analyze_task.done():
                analyze_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await analyze_task
            if shutdown_task is not None and not shutdown_task.done():
                await shutdown_task
            elif shutdown_task is None:
                await service.shutdown()

    asyncio.run(scenario())


def test_admission_lock_wait_does_not_block_event_loop(dataset):
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_success_worker,
        )

        def hold_condition():
            with service._condition:
                lock_acquired.set()
                release_lock.wait(timeout=3)

        holder = threading.Thread(target=hold_condition)
        holder.start()

        try:
            assert await asyncio.to_thread(lock_acquired.wait, 2)

            task = asyncio.create_task(
                service.analyze(dataset, include_correlations=False)
            )

            await asyncio.wait_for(asyncio.sleep(0.05), timeout=0.25)
            assert not task.done(), "Admission unexpectedly bypassed the held lock."

            release_lock.set()
            assert await task == (("score",), False)
        finally:
            release_lock.set()
            await asyncio.to_thread(holder.join, 2)
            await service.shutdown()

    asyncio.run(scenario())


def test_cancelled_admission_releases_slot_after_lock_wait(dataset):
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_success_worker,
        )

        def hold_condition():
            with service._condition:
                lock_acquired.set()
                release_lock.wait(timeout=3)

        holder = threading.Thread(target=hold_condition)
        holder.start()

        try:
            assert await asyncio.to_thread(lock_acquired.wait, 2)

            task = asyncio.create_task(
                service.analyze(dataset, include_correlations=False)
            )
            await asyncio.sleep(0.05)
            task.cancel()

            await asyncio.sleep(0.05)
            assert not task.done(), "Cancellation skipped pending admission cleanup."

            release_lock.set()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert service._active == 0
        finally:
            release_lock.set()
            await asyncio.to_thread(holder.join, 2)
            await service.shutdown()

    asyncio.run(scenario())


def test_repeated_cancellation_during_admission_does_not_leak_slot(dataset):
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_success_worker,
        )

        def hold_condition():
            with service._condition:
                lock_acquired.set()
                release_lock.wait(timeout=3)

        holder = threading.Thread(target=hold_condition)
        holder.start()

        try:
            assert await asyncio.to_thread(lock_acquired.wait, 2)

            task = asyncio.create_task(
                service.analyze(dataset, include_correlations=False)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.sleep(0.05)
            task.cancel()

            release_lock.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            for _ in range(100):
                if service._active:
                    break
                await asyncio.sleep(0.01)

            assert service._active == 0, "Cancelled admission leaked a slot."
        finally:
            release_lock.set()
            await asyncio.to_thread(holder.join, 2)
            assert service._active == 0, "Analytics slot leaked during cleanup."
            await service.shutdown()

    asyncio.run(scenario())


def test_repeated_cancellation_waits_for_admission_slot_release(
    dataset,
    monkeypatch,
):
    async def scenario():
        service = AnalyticsService(
            max_concurrent_jobs=1,
            timeout_seconds=5,
            worker_target=_success_worker,
        )
        condition_locked = threading.Event()
        unlock_condition = threading.Event()
        release_started = threading.Event()
        allow_release = threading.Event()

        original_release = service._release_slot

        def hold_condition():
            with service._condition:
                condition_locked.set()
                unlock_condition.wait(timeout=3)

        def blocking_release():
            release_started.set()
            if not allow_release.wait(timeout=3):
                raise RuntimeError("Timed out waiting for slot release.")
            original_release()

        monkeypatch.setattr(service, "_release_slot", blocking_release)
        holder = threading.Thread(target=hold_condition)
        task = None

        try:
            holder.start()
            assert await asyncio.to_thread(condition_locked.wait, 2)

            task = asyncio.create_task(
                service.analyze(dataset, include_correlations=False)
            )
            await asyncio.sleep(0.05)
            task.cancel()

            await asyncio.sleep(0.05)
            unlock_condition.set()

            assert await asyncio.to_thread(release_started.wait, 2)

            task.cancel()
            await asyncio.sleep(0.05)

            assert not task.done(), "Cancellation interrupted slot cleanup."

            allow_release.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert service._active == 0
        finally:
            unlock_condition.set()
            allow_release.set()
            await asyncio.to_thread(holder.join, 2)

            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await service.shutdown()

    asyncio.run(scenario())
