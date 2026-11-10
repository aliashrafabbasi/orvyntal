"""Bounded, timeout-aware process isolation for dataset analytics."""

import asyncio
import multiprocessing
import queue
import threading
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from orvyntal.core.analytics_worker import run_analytics_worker
from orvyntal.data_intelligence.ingestion.csv_reader import CsvDataset


class AnalyticsBusyError(Exception):
    """Raised when all analytics worker slots are occupied."""


class AnalyticsTimeoutError(Exception):
    """Raised when an analytics worker exceeds its time budget."""


class AnalyticsExecutionError(Exception):
    """Raised when an analytics worker fails unexpectedly."""


WorkerTarget = Callable[[Connection, CsvDataset, bool], None]


async def _await_task_completion(task: asyncio.Task[Any]) -> Any:
    """Wait for a task to finish despite repeated caller cancellation."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


class AnalyticsService:
    """Run CPU-bound analytics in bounded, independently reclaimable processes."""

    def __init__(
        self,
        *,
        max_concurrent_jobs: int,
        timeout_seconds: int,
        worker_target: WorkerTarget = run_analytics_worker,
    ) -> None:
        if max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be positive.")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive.")

        self._max_concurrent_jobs = max_concurrent_jobs
        self._timeout_seconds = timeout_seconds
        self._worker_target = worker_target
        self._context = multiprocessing.get_context("spawn")
        self._condition = threading.Condition()
        self._active = 0
        self._processes: dict[int, multiprocessing.Process] = {}
        self._closed = False

    async def analyze(
        self,
        dataset: CsvDataset,
        *,
        include_correlations: bool,
    ) -> tuple[Any, ...]:
        """Run analytics without blocking the application's event loop."""
        admission = asyncio.create_task(asyncio.to_thread(self._acquire_slot))

        try:
            await asyncio.shield(admission)
        except asyncio.CancelledError:
            try:
                await _await_task_completion(admission)
            except Exception:
                pass
            else:
                release_task = asyncio.create_task(
                    asyncio.to_thread(self._release_slot)
                )
                await _await_task_completion(release_task)
            raise

        cancellation = threading.Event()

        try:
            supervisor = asyncio.create_task(
                asyncio.to_thread(
                    self._execute,
                    dataset,
                    include_correlations,
                    cancellation,
                )
            )
        except BaseException:
            await asyncio.to_thread(self._release_slot)
            raise

        try:
            return await asyncio.shield(supervisor)
        except asyncio.CancelledError:
            cancellation.set()
            try:
                await _await_task_completion(supervisor)
            except Exception:
                pass
            raise

    def _execute(
        self,
        dataset: CsvDataset,
        include_correlations: bool,
        cancellation: threading.Event,
    ) -> tuple[Any, ...]:
        """Supervise one child process from a bounded worker thread."""
        deadline = time.monotonic() + self._timeout_seconds
        receiver = None
        sender = None
        process = None
        reader = None
        reader_started = False
        messages: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        registered = False
        result_received = False

        try:
            receiver, sender = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=self._worker_target,
                args=(sender, dataset, include_correlations),
                daemon=True,
            )

            with self._condition:
                if self._closed or cancellation.is_set():
                    raise AnalyticsExecutionError("Analytics service is shutting down.")

            process.start()

            with self._condition:
                self._processes[id(process)] = process
                registered = True
                if self._closed or cancellation.is_set():
                    raise AnalyticsExecutionError("Analytics service is shutting down.")

            sender.close()
            sender = None

            def receive_result() -> None:
                try:
                    messages.put(("result", receiver.recv()))
                except (EOFError, OSError) as exc:
                    messages.put(("error", exc))
                finally:
                    try:
                        receiver.close()
                    except OSError:
                        pass

            reader = threading.Thread(
                target=receive_result,
                name="orvyntal-analytics-reader",
                daemon=True,
            )
            reader.start()
            reader_started = True

            while True:
                if cancellation.is_set():
                    raise AnalyticsExecutionError("Analytics request was cancelled.")

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AnalyticsTimeoutError

                try:
                    kind, message = messages.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue

                if kind == "error":
                    raise AnalyticsExecutionError from message

                if kind != "result" or not message or message[0] != "ok":
                    raise AnalyticsExecutionError

                result_received = True
                return message[1]

        finally:
            try:
                if process is not None and process.pid is not None:
                    try:
                        if result_received and process.is_alive():
                            process.join(timeout=1)

                        if process.is_alive():
                            try:
                                process.terminate()
                            except (ProcessLookupError, OSError):
                                pass

                        process.join(timeout=1)

                        if process.is_alive():
                            try:
                                process.kill()
                            except (ProcessLookupError, OSError):
                                pass
                            process.join()
                    finally:
                        if registered:
                            with self._condition:
                                self._processes.pop(id(process), None)

                        process.close()
            finally:
                try:
                    if sender is not None:
                        sender.close()
                except OSError:
                    pass
                finally:
                    try:
                        if reader_started and reader is not None:
                            reader.join()
                        elif receiver is not None:
                            receiver.close()
                    except OSError:
                        pass
                    finally:
                        self._release_slot()

    def _acquire_slot(self) -> None:
        """Admit one analytics job without blocking the event loop."""
        with self._condition:
            if self._closed:
                raise AnalyticsExecutionError("Analytics service is unavailable.")

            if self._active >= self._max_concurrent_jobs:
                raise AnalyticsBusyError

            self._active += 1

    def _release_slot(self) -> None:
        """Release one admitted analytics slot."""
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    async def shutdown(self) -> None:
        """Stop accepting work and wait for all worker processes to be reaped."""
        await asyncio.to_thread(self._shutdown_sync)

    def _shutdown_sync(self) -> None:
        """Terminate active workers and wait for supervisor cleanup."""
        with self._condition:
            self._closed = True

            for process in tuple(self._processes.values()):
                if process.is_alive():
                    try:
                        process.terminate()
                    except (ProcessLookupError, OSError):
                        pass

            while self._active:
                self._condition.wait()
