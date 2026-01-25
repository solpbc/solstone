# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Async batch processing for LLM API requests.

Provides Batch for concurrent execution of multiple LLM API calls
with dynamic request queuing and result streaming via async iterator.
Routes requests to providers based on context via the unified agenerate() API.

Example:
    batch = Batch(max_concurrent=5)

    req = batch.create(contents="What is 2+2?", context="myapp.calc")
    req.my_id = "calc1"
    batch.add(req)

    async for req in batch.drain_batch():
        print(f"{req.my_id}: {req.response}")

Provider-specific features:
    - client: Optional client for connection reuse (Google only, others use singletons)
    - cached_content: Content caching (Google only)
"""

import asyncio
import time
from typing import Any, List, Optional, Union

from .models import agenerate


class BatchRequest:
    """
    Mutable request object for a single LLM API call.

    Core attributes are passed to agenerate(). Callers can add
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
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 8192 * 2,
        system_instruction: Optional[str] = None,
        json_output: bool = False,
        thinking_budget: Optional[int] = None,
        cached_content: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ):
        self.contents = contents
        self.context = context
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_instruction = system_instruction
        self.json_output = json_output
        self.thinking_budget = thinking_budget
        self.cached_content = cached_content
        self.timeout_s = timeout_s

        # Populated after execution
        self.response: Optional[str] = None
        self.error: Optional[str] = None
        self.duration: float = 0.0
        self.model_used: str = model or ""


class Batch:
    """
    Async batch processor for LLM API requests.

    Manages concurrent execution with dynamic request queuing and result
    streaming via async iterator pattern. Routes to providers via agenerate().

    Example:
        batch = Batch(max_concurrent=5)

        # Add requests
        req1 = batch.create(contents="What is 2+2?", context="myapp.calc")
        req1.task_id = "calc1"
        batch.add(req1)

        req2 = batch.create(contents="What is 3+3?", context="myapp.calc")
        req2.task_id = "calc2"
        batch.add(req2)

        # Process results as they complete
        async for req in batch.drain_batch():
            print(f"{req.task_id}: {req.response}")
    """

    def __init__(self, max_concurrent: int = 5, client: Any = None):
        """
        Initialize batch processor.

        Parameters
        ----------
        max_concurrent : int
            Maximum number of concurrent API requests (default: 5)
        client : Any, optional
            Provider client for connection reuse. Passed through to backend.
            Google: genai.Client instance for connection pooling
            Other providers: Ignored (they use internal singletons)
        """
        self.max_concurrent = max_concurrent
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.pending_tasks: set = set()

    def create(
        self,
        contents: Union[str, List[Any]],
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 8192 * 2,
        system_instruction: Optional[str] = None,
        json_output: bool = False,
        thinking_budget: Optional[int] = None,
        cached_content: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> BatchRequest:
        """
        Create a new BatchRequest.

        Convenience factory method. Caller can add arbitrary attributes
        to the returned request before calling add().

        Parameters
        ----------
        contents : str or List
            The content to send to the model
        context : str
            Context string for provider routing (e.g., "observe.describe.frame")
        model : str, optional
            Model override. If not provided, resolved from context.

        Returns
        -------
        BatchRequest
            New request object ready to be customized and added
        """
        return BatchRequest(
            contents=contents,
            context=context,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            json_output=json_output,
            thinking_budget=thinking_budget,
            cached_content=cached_content,
            timeout_s=timeout_s,
        )

    def add(self, request: BatchRequest) -> None:
        """
        Add request to batch for execution.

        Request will be executed concurrently (up to max_concurrent limit).
        Non-blocking - returns immediately. Can be called at any time, even
        during iteration or after draining.

        Parameters
        ----------
        request : BatchRequest
            Request to execute
        """
        task = asyncio.create_task(self._execute_request(request))
        self.pending_tasks.add(task)

    def update(self, request: BatchRequest, **kwargs) -> None:
        """
        Update request attributes and re-add to batch for execution.

        This is useful for retries or multi-stage processing where you want
        to reuse the same request object with different parameters.

        Parameters
        ----------
        request : BatchRequest
            Request to update and re-execute
        **kwargs
            Any attributes to update on the request object

        Example
        -------
        >>> batch.update(
        ...     req,
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

    async def _execute_request(self, request: BatchRequest) -> None:
        """
        Execute a single request and put result in queue.

        Parameters
        ----------
        request : BatchRequest
            Request to execute (will be modified in place)
        """
        start_time = time.time()
        try:
            async with self.semaphore:
                # Build kwargs for provider-specific options
                kwargs: dict = {}
                if self.client is not None:
                    kwargs["client"] = self.client
                if request.cached_content is not None:
                    kwargs["cached_content"] = request.cached_content
                if request.model is not None:
                    kwargs["model"] = request.model

                response = await agenerate(
                    contents=request.contents,
                    context=request.context,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    system_instruction=request.system_instruction,
                    json_output=request.json_output,
                    thinking_budget=request.thinking_budget,
                    timeout_s=request.timeout_s,
                    **kwargs,
                )
                request.duration = time.time() - start_time
                request.response = response
                request.error = None

                # Track which model was actually used
                if request.model:
                    request.model_used = request.model
                else:
                    # Model was resolved from context - we don't have easy access
                    # to what was resolved, so leave as empty string
                    pass
        except Exception as e:
            request.duration = time.time() - start_time
            request.response = None
            request.error = str(e)

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
        BatchRequest
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


__all__ = ["BatchRequest", "Batch"]
