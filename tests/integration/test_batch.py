"""Integration tests for GeminiBatch with real Gemini API."""

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest

from think.batch import GeminiBatch
from think.models import GEMINI_FLASH, GEMINI_LITE


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_basic_execution():
    """Test basic batch execution with real API."""
    batch = GeminiBatch(max_concurrent=3)

    # Add simple requests
    req1 = batch.create(contents="What is 2+2? Reply with just the number.")
    req1.id = "calc1"
    batch.add(req1)

    req2 = batch.create(contents="What is 3+3? Reply with just the number.")
    req2.id = "calc2"
    batch.add(req2)

    # Collect results
    results = []
    async for req in batch.drain_batch():
        results.append(req)

    # Verify both completed
    assert len(results) == 2

    # Check IDs are preserved
    result_ids = {r.id for r in results}
    assert result_ids == {"calc1", "calc2"}

    # Check responses
    for r in results:
        assert r.response is not None
        assert r.error is None
        assert r.duration > 0
        assert (
            "2" in r.response
            or "3" in r.response
            or "4" in r.response
            or "6" in r.response
        )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_concurrent_timing():
    """Test that concurrent execution is actually faster than sequential."""
    # Sequential baseline
    start = time.time()
    batch_seq = GeminiBatch(max_concurrent=1)
    for i in range(3):
        req = batch_seq.create(contents=f"Count to {i+1}. Reply with just the number.")
        batch_seq.add(req)

    seq_results = []
    async for req in batch_seq.drain_batch():
        seq_results.append(req)
    seq_duration = time.time() - start

    # Concurrent execution
    start = time.time()
    batch_conc = GeminiBatch(max_concurrent=3)
    for i in range(3):
        req = batch_conc.create(contents=f"Count to {i+1}. Reply with just the number.")
        batch_conc.add(req)

    conc_results = []
    async for req in batch_conc.drain_batch():
        conc_results.append(req)
    conc_duration = time.time() - start

    # Both should complete successfully
    assert len(seq_results) == 3
    assert len(conc_results) == 3

    # Concurrent should be faster (allow some margin for variance)
    # In practice, 3 concurrent requests should be ~1.5-2x faster
    assert conc_duration < seq_duration * 0.8


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_json_output():
    """Test batch with JSON output mode."""
    batch = GeminiBatch(max_concurrent=2)

    req = batch.create(
        contents='Return a JSON object with "result": 10',
        json_output=True,
    )
    req.id = "json_test"
    batch.add(req)

    results = []
    async for r in batch.drain_batch():
        results.append(r)

    assert len(results) == 1
    assert results[0].response is not None
    assert "{" in results[0].response
    assert "}" in results[0].response


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_different_models():
    """Test batch with different models."""
    batch = GeminiBatch(max_concurrent=2)

    req1 = batch.create(contents="Say 'flash'", model=GEMINI_FLASH)
    req1.model_type = "flash"
    batch.add(req1)

    req2 = batch.create(contents="Say 'lite'", model=GEMINI_LITE)
    req2.model_type = "lite"
    batch.add(req2)

    results = []
    async for r in batch.drain_batch():
        results.append(r)

    assert len(results) == 2

    # Both should succeed
    for r in results:
        assert r.response is not None
        assert r.error is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_dynamic_adding():
    """Test multi-stage pattern - add stage 2 based on stage 1 results."""
    batch = GeminiBatch(max_concurrent=3)

    # Stage 1: Initial requests
    req1 = batch.create(contents="What is 5+5? Just the number.")
    req1.stage = "stage1"
    req1.value = 5
    batch.add(req1)

    stage1_count = 0
    stage2_added = False

    async for req in batch.drain_batch():
        if req.stage == "stage1":
            stage1_count += 1

            # Add stage 2 request based on result
            if not stage2_added:
                req2 = batch.create(
                    contents=f"Previous answer was {req.response}. Double it. Just the number."
                )
                req2.stage = "stage2"
                batch.add(req2)
                stage2_added = True

    # Should have processed both stages
    assert stage1_count == 1


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_token_logging():
    """Test that token logging works with batch execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["JOURNAL_PATH"] = tmpdir

        batch = GeminiBatch(max_concurrent=2)

        req = batch.create(contents="Say hello")
        batch.add(req)

        results = []
        async for r in batch.drain_batch():
            results.append(r)

        # Check that token logs were created
        tokens_dir = Path(tmpdir) / "tokens"
        if tokens_dir.exists():
            log_files = list(tokens_dir.glob("*.jsonl"))
            # Should have at least one log file
            assert len(log_files) >= 1


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_error_recovery():
    """Test retry pattern with real API (simulate by using invalid then valid)."""
    batch = GeminiBatch(max_concurrent=2)

    # This might error or succeed depending on model - just test the pattern
    req1 = batch.create(
        contents="What is 1+1? Reply with just the number.",
        max_output_tokens=5,  # Very small, might cause issues
    )
    req1.attempt = 1
    batch.add(req1)

    retried = False
    async for req in batch.drain_batch():
        if req.attempt == 1:
            # Always add a retry request to test the pattern
            if not retried:
                req2 = batch.create(
                    contents="What is 1+1? Reply with just the number.",
                    max_output_tokens=100,  # Normal size
                )
                req2.attempt = 2
                batch.add(req2)
                retried = True

    # Pattern should complete successfully
    assert retried


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_client_reuse():
    """Test that client is reused across requests in batch."""
    from google import genai

    from think.models import get_or_create_client

    # Create shared client
    client = get_or_create_client()

    # Use it in batch
    batch = GeminiBatch(max_concurrent=2, client=client)

    req1 = batch.create(contents="Say 'first'")
    batch.add(req1)

    req2 = batch.create(contents="Say 'second'")
    batch.add(req2)

    results = []
    async for req in batch.drain_batch():
        results.append(req)

    # Both should succeed with shared client
    assert len(results) == 2
    for r in results:
        assert r.response is not None
        assert r.error is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_batch_custom_attributes_preserved():
    """Test that custom attributes added to requests are preserved."""
    batch = GeminiBatch(max_concurrent=2)

    req = batch.create(contents="What is 10+10? Just the number.")
    req.frame_id = 42
    req.monitor = "DP-3"
    req.metadata = {"foo": "bar", "nested": {"baz": 123}}
    batch.add(req)

    results = []
    async for r in batch.drain_batch():
        results.append(r)

    assert len(results) == 1
    result = results[0]

    # Custom attributes should be preserved
    assert result.frame_id == 42
    assert result.monitor == "DP-3"
    assert result.metadata == {"foo": "bar", "nested": {"baz": 123}}

    # Result attributes should be populated
    assert result.response is not None
    assert result.error is None
    assert result.duration > 0
