"""Tests for the GeminiBatch async batch processor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from think.batch import GeminiBatch, GeminiRequest
from think.models import GEMINI_FLASH, GEMINI_LITE


def test_gemini_request_creation():
    """Test GeminiRequest can be created with default and custom params."""
    # Default params
    req = GeminiRequest(contents="Test prompt")
    assert req.contents == "Test prompt"
    assert req.model == GEMINI_FLASH
    assert req.temperature == 0.3
    assert req.response is None
    assert req.error is None

    # Custom params
    req2 = GeminiRequest(
        contents=["Part 1", "Part 2"],
        model=GEMINI_LITE,
        temperature=0.7,
        json_output=True,
    )
    assert req2.contents == ["Part 1", "Part 2"]
    assert req2.model == GEMINI_LITE
    assert req2.temperature == 0.7
    assert req2.json_output is True


def test_gemini_request_custom_attributes():
    """Test that arbitrary attributes can be added to GeminiRequest."""
    req = GeminiRequest(contents="Test")
    req.frame_id = 123
    req.stage = "initial"
    req.custom_data = {"foo": "bar"}

    assert req.frame_id == 123
    assert req.stage == "initial"
    assert req.custom_data == {"foo": "bar"}


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_basic(mock_get_client, mock_agenerate):
    """Test basic batch execution with single request."""
    # Setup mocks
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_agenerate.return_value = "Response 1"

    # Create batch and add request
    batch = GeminiBatch(max_concurrent=5)
    req = batch.create(contents="What is 2+2?")
    req.task_id = "calc1"
    batch.add(req)

    # Iterate and verify
    results = []
    async for completed_req in batch.drain_batch():
        results.append(completed_req)

    assert len(results) == 1
    assert results[0].task_id == "calc1"
    assert results[0].response == "Response 1"
    assert results[0].error is None
    assert results[0].duration > 0
    assert results[0].model_used == GEMINI_FLASH

    # Verify API was called correctly
    mock_agenerate.assert_called_once()
    call_kwargs = mock_agenerate.call_args[1]
    assert call_kwargs["contents"] == "What is 2+2?"
    assert call_kwargs["model"] == GEMINI_FLASH


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_multiple_requests(mock_get_client, mock_agenerate):
    """Test batch with multiple requests."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Mock returns different responses
    mock_agenerate.side_effect = ["Response 1", "Response 2", "Response 3"]

    batch = GeminiBatch(max_concurrent=2)

    # Add multiple requests
    req1 = batch.create(contents="Prompt 1")
    req1.id = 1
    batch.add(req1)

    req2 = batch.create(contents="Prompt 2")
    req2.id = 2
    batch.add(req2)

    req3 = batch.create(contents="Prompt 3")
    req3.id = 3
    batch.add(req3)

    # Collect results
    results = []
    async for req in batch.drain_batch():
        results.append(req)

    # Should have all 3 results
    assert len(results) == 3

    # Results may come in any order (concurrent execution)
    result_ids = {r.id for r in results}
    assert result_ids == {1, 2, 3}

    # All should have responses
    for r in results:
        assert r.response is not None
        assert r.error is None


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_error_handling(mock_get_client, mock_agenerate):
    """Test that errors are captured in request.error."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Mock raises exception
    mock_agenerate.side_effect = ValueError("API error")

    batch = GeminiBatch(max_concurrent=5)
    req = batch.create(contents="Bad prompt")
    req.id = "error_test"
    batch.add(req)

    # Get result
    results = []
    async for r in batch.drain_batch():
        results.append(r)

    assert len(results) == 1
    assert results[0].id == "error_test"
    assert results[0].response is None
    assert results[0].error == "API error"
    assert results[0].duration > 0


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_dynamic_adding(mock_get_client, mock_agenerate):
    """Test adding requests dynamically during iteration."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_agenerate.return_value = "Response"

    batch = GeminiBatch(max_concurrent=5)

    # Add initial request
    req1 = batch.create(contents="Initial")
    req1.stage = "initial"
    batch.add(req1)

    # Process and add more during iteration
    results = []
    added_followup = False

    async for req in batch.drain_batch():
        results.append(req)

        # After first result, add a follow-up
        if req.stage == "initial" and not added_followup:
            req2 = batch.create(contents="Follow-up")
            req2.stage = "followup"
            batch.add(req2)
            added_followup = True

    # Should have both results
    assert len(results) == 2
    stages = {r.stage for r in results}
    assert stages == {"initial", "followup"}


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_retry_pattern(mock_get_client, mock_agenerate):
    """Test retry pattern - add failed request back with different model."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # First call fails, second succeeds
    call_count = 0

    async def mock_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Transient error")
        return "Success on retry"

    mock_agenerate.side_effect = mock_response

    batch = GeminiBatch(max_concurrent=5)

    # Add initial request
    req1 = batch.create(contents="Test", model=GEMINI_FLASH)
    req1.attempt = 1
    batch.add(req1)

    results = []
    async for req in batch.drain_batch():
        results.append(req)

        # If error, retry with different model
        if req.error and req.attempt == 1:
            retry = batch.create(req.contents, model=GEMINI_LITE)
            retry.attempt = 2
            batch.add(retry)

    # Should have both attempts
    assert len(results) == 2
    assert results[0].error is not None
    assert results[0].attempt == 1
    assert results[1].response == "Success on retry"
    assert results[1].attempt == 2


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_factory_method(mock_get_client, mock_agenerate):
    """Test that batch.create() factory method works correctly."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_agenerate.return_value = "Response"

    batch = GeminiBatch()

    # Use factory method
    req = batch.create(
        contents="Test",
        model=GEMINI_LITE,
        temperature=0.8,
        json_output=True,
    )

    assert isinstance(req, GeminiRequest)
    assert req.contents == "Test"
    assert req.model == GEMINI_LITE
    assert req.temperature == 0.8
    assert req.json_output is True


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_can_add_after_draining(mock_get_client, mock_agenerate):
    """Test that adding after draining works (reusable batch)."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_agenerate.side_effect = ["Response 1", "Response 2"]

    batch = GeminiBatch()

    # First batch
    req1 = batch.create(contents="First")
    req1.id = 1
    batch.add(req1)

    results = []
    async for req in batch.drain_batch():
        results.append(req)

    assert len(results) == 1
    assert results[0].id == 1

    # Add more work after draining
    req2 = batch.create(contents="Second")
    req2.id = 2
    batch.add(req2)

    async for req in batch.drain_batch():
        results.append(req)

    # Should have both results
    assert len(results) == 2
    assert {r.id for r in results} == {1, 2}


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_empty_batch(mock_get_client, mock_agenerate):
    """Test that empty batch (no requests) completes immediately."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    batch = GeminiBatch()

    results = []
    async for req in batch.drain_batch():
        results.append(req)

    assert len(results) == 0


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_concurrency_limit(mock_get_client, mock_agenerate):
    """Test that semaphore limits concurrent requests."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Track concurrent calls
    concurrent_calls = 0
    max_concurrent_seen = 0
    lock = asyncio.Lock()

    async def mock_with_tracking(*args, **kwargs):
        nonlocal concurrent_calls, max_concurrent_seen
        async with lock:
            concurrent_calls += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_calls)

        await asyncio.sleep(0.1)  # Simulate API call

        async with lock:
            concurrent_calls -= 1

        return "Response"

    mock_agenerate.side_effect = mock_with_tracking

    # Create batch with max_concurrent=2
    batch = GeminiBatch(max_concurrent=2)

    # Add 5 requests
    for i in range(5):
        req = batch.create(contents=f"Request {i}")
        batch.add(req)

    results = []
    async for req in batch.drain_batch():
        results.append(req)

    assert len(results) == 5
    # Should never exceed max_concurrent=2
    assert max_concurrent_seen <= 2


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_update_method(mock_get_client, mock_agenerate):
    """Test batch.update() method for modifying and re-adding requests."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Track which model was used in each call
    call_models = []

    async def mock_track_model(*args, **kwargs):
        call_models.append(kwargs.get("model", "unknown"))
        return f"Response from {kwargs.get('model', 'unknown')}"

    mock_agenerate.side_effect = mock_track_model

    batch = GeminiBatch(max_concurrent=5)

    # Create initial request
    req = batch.create(contents="Initial prompt", model=GEMINI_FLASH)
    req.stage = "initial"
    batch.add(req)

    results = []
    result_count = 0
    async for completed_req in batch.drain_batch():
        result_count += 1
        # Capture the response at this point
        results.append((result_count, completed_req.response, completed_req.stage))

        # After first result, update and re-add with different model
        if result_count == 1:
            batch.update(
                completed_req,
                contents="Updated prompt",
                model=GEMINI_LITE,
                stage="updated",  # Update custom attribute too
                custom_field="test_value",  # Add new custom attribute
            )

    # Should have both results
    assert len(results) == 2
    assert results[0][2] == "initial"  # First result was initial stage
    assert results[1][2] == "updated"  # Second result was updated stage

    # Verify models used
    assert call_models == [GEMINI_FLASH, GEMINI_LITE]

    # Verify correct responses at each stage
    assert results[0][1] == f"Response from {GEMINI_FLASH}"
    assert results[1][1] == f"Response from {GEMINI_LITE}"

    # Verify custom attribute was set
    assert req.custom_field == "test_value"


def test_gemini_request_with_timeout():
    """Test GeminiRequest can be created with timeout_s parameter."""
    req = GeminiRequest(contents="Test prompt", timeout_s=30)
    assert req.timeout_s == 30

    req2 = GeminiRequest(contents="Test prompt", timeout_s=60.5)
    assert req2.timeout_s == 60.5

    # Default is None
    req3 = GeminiRequest(contents="Test prompt")
    assert req3.timeout_s is None


@pytest.mark.asyncio
@patch("think.batch.gemini_agenerate", new_callable=AsyncMock)
@patch("think.batch.get_or_create_client")
async def test_gemini_batch_timeout_passthrough(mock_get_client, mock_agenerate):
    """Test that timeout_s is passed through to gemini_agenerate."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_agenerate.return_value = "Response"

    batch = GeminiBatch(max_concurrent=5)

    # Create request with timeout_s
    req = batch.create(contents="Test prompt", timeout_s=45)
    batch.add(req)

    results = []
    async for completed_req in batch.drain_batch():
        results.append(completed_req)

    assert len(results) == 1

    # Verify timeout_s was passed to gemini_agenerate
    mock_agenerate.assert_called_once()
    call_kwargs = mock_agenerate.call_args[1]
    assert call_kwargs["timeout_s"] == 45
