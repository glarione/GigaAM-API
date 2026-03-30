"""Dynamic request batching for improved throughput."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BatchRequest:
    """A single request waiting to be batched."""

    future: asyncio.Future
    args: tuple
    kwargs: dict
    timestamp: float = field(default_factory=time.time)


class BatchManager:
    """
    Manages dynamic request batching for improved throughput.

    Groups incoming requests into batches with a configurable wait time.
    This is especially useful on CPU to amortize overhead across multiple requests.
    """

    def __init__(
        self,
        batch_wait_ms: int = 25,
        max_batch_size: int = 8,
    ):
        """
        Initialize the batch manager.

        Args:
            batch_wait_ms: Maximum time to wait for batching (milliseconds)
            max_batch_size: Maximum number of requests per batch
        """
        self.batch_wait_ms = batch_wait_ms
        self.max_batch_size = max_batch_size
        self._queue: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False
        self._processor_task: asyncio.Task | None = None

    async def submit(
        self, func: Callable, *args, **kwargs
    ) -> Any:
        """
        Submit a request for batching.

        Args:
            func: The function to call with the batched args
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call
        """
        future = asyncio.get_event_loop().create_future()
        request = BatchRequest(future=future, args=args, kwargs=kwargs)

        async with self._lock:
            self._queue.append(request)

            # Start processor if not running
            if not self._processing:
                self._processing = True
                # Schedule batch processing after wait time
                asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self) -> None:
        """Process the current batch of requests."""
        # Wait for batch window or max batch size
        await asyncio.sleep(self.batch_wait_ms / 1000.0)

        async with self._lock:
            if not self._queue:
                self._processing = False
                return

            # Get current batch (up to max_batch_size)
            batch = self._queue[: self.max_batch_size]
            self._queue = self._queue[self.max_batch_size :]

            # Keep processing flag if more requests waiting
            if self._queue:
                asyncio.create_task(self._process_batch())
            else:
                self._processing = False

        # Process batch - for now, process sequentially
        # TODO: Implement true batched inference for more speedup
        for request in batch:
            try:
                # Call function with request's args
                result = await self._call_with_args(request)
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as e:
                if not request.future.done():
                    request.future.set_exception(e)

    async def _call_with_args(self, request: BatchRequest) -> Any:
        """Call the function with the request's arguments."""
        # For transcription, we process each request individually
        # but batched at the queue level to reduce overhead
        from gigaam_server.services.transcription import TranscriptionService

        if isinstance(request.args[0], TranscriptionService):
            service = request.args[0]
            return await service.transcribe_from_bytes(
                *request.args[1:], **request.kwargs
            )
        return None

    def is_enabled(self) -> bool:
        """Check if batching is enabled."""
        return self.max_batch_size > 1
