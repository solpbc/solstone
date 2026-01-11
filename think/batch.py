# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Async batch processing for Gemini API requests.

Provides GeminiBatch for concurrent execution of multiple Gemini API calls
with dynamic request queuing and result streaming via async iterator.

Example:
    batch = GeminiBatch(max_concurrent=5)

    req = batch.create(contents="What is 2+2?")
    req.my_id = "calc1"
    batch.add(req)

    async for req in batch.drain_batch():
        print(f"{req.my_id}: {req.response}")
"""

import asyncio
import time
from typing import Any, List, Optional, Union

from google import genai

from muse.google import get_or_create_client
from think.models import GEMINI_FLASH, gemini_agenerate

# Note: This module is Google-specific for async batch processing.
# Callers should use resolve_provider() to get the model and pass it explicitly.
# The GEMINI_FLASH default is a fallback for cases where context-based
# resolution isn't needed.


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
        timeout_s: Optional[float] = None,
        context: Optional[str] = None,
    ):
        self.contents = contents
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_instruction = system_instruction
        self.json_output = json_output
        self.thinking_budget = thinking_budget
        self.cached_content = cached_content
        self.timeout_s = timeout_s
        self.context = context

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

        # Process results as they complete
        async for req in batch.drain_batch():
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
        self.client = get_or_create_client(client)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.pending_tasks: set = set()

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
        timeout_s: Optional[float] = None,
        context: Optional[str] = None,
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
            timeout_s=timeout_s,
            context=context,
        )

    def add(self, request: GeminiRequest) -> None:
        """
        Add request to batch for execution.

        Request will be executed concurrently (up to max_concurrent limit).
        Non-blocking - returns immediately. Can be called at any time, even
        during iteration or after draining.

        Parameters
        ----------
        request : GeminiRequest
            Request to execute
        """
        task = asyncio.create_task(self._execute_request(request))
        self.pending_tasks.add(task)

    def update(self, request: GeminiRequest, **kwargs) -> None:
        """
        Update request attributes and re-add to batch for execution.

        This is useful for retries or multi-stage processing where you want
        to reuse the same request object with different parameters.

        Parameters
        ----------
        request : GeminiRequest
            Request to update and re-execute
        **kwargs
            Any attributes to update on the request object

        Example
        -------
        >>> batch.update(
        ...     req,
        ...     model=GEMINI_FLASH,
        ...     contents="New prompt",
        ...     temperature=0.8,
        ...     custom_attr="foo"
        ... )
        """
        # Update any provided attributes
        for key, value in kwargs.items():
            setattr(request, key, value)

        # Clear previous execution results
        request.response = None
        request.error = None
        request.duration = 0.0

        # Re-add to batch
        self.add(request)

    def is_drained(self) -> bool:
        """
        Check if batch is fully drained.

        Returns True when there are no pending tasks and no results waiting
        in the queue.

        Returns
        -------
        bool
            True if batch is drained, False otherwise
        """
        # Clean up completed tasks
        self.pending_tasks = {t for t in self.pending_tasks if not t.done()}
        return len(self.pending_tasks) == 0 and self.result_queue.empty()

    async def wait_until_drained(self) -> None:
        """
        Wait until all pending work completes and queue is empty.

        Blocks until is_drained() returns True.
        """
        while not self.is_drained():
            await asyncio.sleep(0.1)

    async def _execute_request(self, request: GeminiRequest) -> None:
        """
        Execute a single request and put result in queue.

        Parameters
        ----------
        request : GeminiRequest
            Request to execute (will be modified in place)
        """
        try:
            async with self.semaphore:
                start_time = time.time()
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
                    timeout_s=request.timeout_s,
                    context=request.context,
                )
                request.duration = time.time() - start_time
                request.response = response
                request.error = None
        except Exception as e:
            request.duration = time.time() - start_time
            request.response = None
            request.error = str(e)

        request.model_used = request.model

        # Put completed request in result queue
        await self.result_queue.put(request)

    async def drain_batch(self):
        """
        Async generator that yields completed requests until batch is drained.

        Yields results from the queue while there's still pending work OR
        results waiting. Stops when both pending_tasks is empty AND queue
        is empty.

        This can be called multiple times - each call will drain whatever
        work is currently in the batch.

        Yields
        ------
        GeminiRequest
            Completed request with response/error populated

        Example
        -------
        >>> async for req in batch.drain_batch():
        ...     print(req.response)
        ...     if req.error:
        ...         batch.add(req)  # Retry on error
        """
        while True:
            # Check if we're truly drained
            self.pending_tasks = {t for t in self.pending_tasks if not t.done()}

            # If drained, stop iteration
            if len(self.pending_tasks) == 0 and self.result_queue.empty():
                break

            # Try to get a result (with timeout to avoid blocking forever)
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
                yield result
            except asyncio.TimeoutError:
                # No result ready yet, but might have pending work
                continue


__all__ = ["GeminiRequest", "GeminiBatch"]
