"""
Async batch processing for Gemini API requests.

Provides GeminiBatch for concurrent execution of multiple Gemini API calls
with dynamic request queuing and result streaming via async iterator.

Example:
    batch = GeminiBatch(max_concurrent=5)

    req = batch.create(contents="What is 2+2?")
    req.my_id = "calc1"
    batch.add(req)

    batch.close()

    async for req in batch:
        print(f"{req.my_id}: {req.response}")
"""

import asyncio
import time
from typing import Any, List, Optional, Union

from google import genai

from think.models import GEMINI_FLASH, _get_or_create_client, gemini_agenerate


class GeminiRequest:
    """
    Mutable request object for a single Gemini API call.

    Core attributes are passed to gemini_agenerate(). Callers can add
    arbitrary attributes for tracking (e.g., frame_id, stage, etc).

    After execution, these attributes are populated:
        - response: Optional[str] - Response text (None if error)
        - error: Optional[str] - Error message (None if success)
        - duration: float - Execution time in seconds
        - model_used: str - Model that was used
    """

    def __init__(
        self,
        contents: Union[str, List[Any]],
        model: str = GEMINI_FLASH,
        temperature: float = 0.3,
        max_output_tokens: int = 8192 * 2,
        system_instruction: Optional[str] = None,
        json_output: bool = False,
        thinking_budget: Optional[int] = None,
        cached_content: Optional[str] = None,
    ):
        self.contents = contents
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_instruction = system_instruction
        self.json_output = json_output
        self.thinking_budget = thinking_budget
        self.cached_content = cached_content

        # Populated after execution
        self.response: Optional[str] = None
        self.error: Optional[str] = None
        self.duration: float = 0.0
        self.model_used: str = model


class GeminiBatch:
    """
    Async batch processor for Gemini API requests.

    Manages concurrent execution with dynamic request queuing and result
    streaming via async iterator pattern.

    Example:
        batch = GeminiBatch(max_concurrent=5)

        # Add requests
        req1 = batch.create(contents="What is 2+2?")
        req1.task_id = "calc1"
        batch.add(req1)

        req2 = batch.create(contents="What is 3+3?")
        req2.task_id = "calc2"
        batch.add(req2)

        batch.close()  # Signal no more requests

        # Process results as they complete
        async for req in batch:
            print(f"{req.task_id}: {req.response}")
    """

    def __init__(self, max_concurrent: int = 5, client: Optional[genai.Client] = None):
        """
        Initialize batch processor.

        Parameters
        ----------
        max_concurrent : int
            Maximum number of concurrent API requests (default: 5)
        client : genai.Client, optional
            Shared client to reuse. If not provided, creates a new one.
        """
        self.max_concurrent = max_concurrent
        self.client = _get_or_create_client(client)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.pending_tasks: set = set()
        self.closed = False
        self._monitor_task: Optional[asyncio.Task] = None

    def create(
        self,
        contents: Union[str, List[Any]],
        model: str = GEMINI_FLASH,
        temperature: float = 0.3,
        max_output_tokens: int = 8192 * 2,
        system_instruction: Optional[str] = None,
        json_output: bool = False,
        thinking_budget: Optional[int] = None,
        cached_content: Optional[str] = None,
    ) -> GeminiRequest:
        """
        Create a new GeminiRequest.

        Convenience factory method. Caller can add arbitrary attributes
        to the returned request before calling add().

        Returns
        -------
        GeminiRequest
            New request object ready to be customized and added
        """
        return GeminiRequest(
            contents=contents,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            json_output=json_output,
            thinking_budget=thinking_budget,
            cached_content=cached_content,
        )

    def add(self, request: GeminiRequest) -> None:
        """
        Add request to batch for execution.

        Request will be executed concurrently (up to max_concurrent limit).
        Non-blocking - returns immediately.

        Parameters
        ----------
        request : GeminiRequest
            Request to execute
        """
        if self.closed:
            raise RuntimeError("Cannot add requests to closed batch")

        task = asyncio.create_task(self._execute_request(request))
        self.pending_tasks.add(task)

        # Start monitor task if not already running
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_completion())

    def close(self) -> None:
        """
        Signal that no more requests will be added.

        The async iterator will drain all pending results and then stop.
        Can be called multiple times (idempotent).
        """
        self.closed = True

        # Start monitor if not already running (handles empty batch case)
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_completion())

    async def _execute_request(self, request: GeminiRequest) -> None:
        """
        Execute a single request and put result in queue.

        Parameters
        ----------
        request : GeminiRequest
            Request to execute (will be modified in place)
        """
        start_time = time.time()

        try:
            async with self.semaphore:
                response = await gemini_agenerate(
                    contents=request.contents,
                    model=request.model,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    system_instruction=request.system_instruction,
                    json_output=request.json_output,
                    thinking_budget=request.thinking_budget,
                    cached_content=request.cached_content,
                    client=self.client,
                )
                request.response = response
                request.error = None
        except Exception as e:
            request.response = None
            request.error = str(e)

        request.duration = time.time() - start_time
        request.model_used = request.model

        # Put completed request in result queue
        await self.result_queue.put(request)

    async def _monitor_completion(self) -> None:
        """
        Monitor pending tasks and put sentinel when all complete and closed.

        This background task waits for all pending tasks to complete, then
        puts None in the result queue to signal the iterator to stop.
        """
        while True:
            await asyncio.sleep(0.1)  # Check every 100ms

            # Remove completed tasks
            self.pending_tasks = {t for t in self.pending_tasks if not t.done()}

            # If closed and no pending tasks, put sentinel and exit
            if self.closed and len(self.pending_tasks) == 0:
                await self.result_queue.put(None)  # Sentinel
                break

    def __aiter__(self):
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> GeminiRequest:
        """
        Get next completed request.

        Blocks until a request completes. Raises StopAsyncIteration when
        batch is closed and all requests have been processed.

        Returns
        -------
        GeminiRequest
            Completed request with response/error populated
        """
        result = await self.result_queue.get()

        if result is None:  # Sentinel value
            raise StopAsyncIteration

        return result


__all__ = ["GeminiRequest", "GeminiBatch"]
