# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration test for gemini_generate function with real API calls."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv
from google.genai import types

from think.models import GEMINI_FLASH, GEMINI_LITE, gemini_agenerate, gemini_generate


def get_fixtures_env():
    """Load the fixtures/.env file and return the environment."""
    fixtures_env = Path(__file__).parent.parent.parent / "fixtures" / ".env"
    if not fixtures_env.exists():
        return None, None, None

    # Load the env file
    load_dotenv(fixtures_env, override=True)

    api_key = os.getenv("GOOGLE_API_KEY")
    journal_path = os.getenv("JOURNAL_PATH")

    return fixtures_env, api_key, journal_path


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_basic_text():
    """Test basic text generation with gemini_generate."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    # Set the API key in environment for gemini_generate to find
    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test basic text generation
    response = gemini_generate(
        "What is 2+2? Reply with just the number.",
        model=GEMINI_FLASH,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "4" in response or "four" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_with_system_instruction():
    """Test gemini_generate with system instruction."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test with system instruction
    response = gemini_generate(
        "Tell me about Python",
        model=GEMINI_FLASH,
        system_instruction="You are a helpful assistant. Keep responses under 50 words.",
        temperature=0.3,
        max_output_tokens=500,  # Increased to avoid hitting limit
    )

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    # Should mention Python
    assert "python" in response.lower()
    # Should be relatively short due to system instruction
    assert len(response.split()) < 100


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_json_output():
    """Test gemini_generate with JSON output mode."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test JSON output mode
    # Note: Gemini 3 uses implicit thinking which consumes part of max_output_tokens
    response = gemini_generate(
        "Create a JSON object with fields 'name' (value: 'test') and 'number' (value: 42)",
        model=GEMINI_FLASH,
        json_output=True,
        temperature=0.1,
        max_output_tokens=500,
    )

    assert response is not None
    assert isinstance(response, str)

    # Should be valid JSON
    try:
        data = json.loads(response)
        assert isinstance(data, dict)
        assert data.get("name") == "test"
        assert data.get("number") == 42
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON: {response}")


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_multipart_content():
    """Test gemini_generate with multipart content list."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test with multiple parts in content
    contents = [
        "Here are two math problems:",
        "1. What is 3+5?",
        "2. What is 10-2?",
        "Please solve both and give just the numbers separated by a comma.",
    ]

    response = gemini_generate(
        contents,
        model=GEMINI_FLASH,
        temperature=0.1,
        max_output_tokens=200,  # Increased for multipart content
    )

    assert response is not None
    assert isinstance(response, str)
    # Should contain both answers
    assert "8" in response
    # Could be "8, 8" or "8 and 8" etc.


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_with_conversation():
    """Test gemini_generate with conversation history using Content objects."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Create a conversation with Content objects
    contents = [
        types.Content(role="user", parts=[types.Part(text="My favorite number is 7.")]),
        types.Content(
            role="model",
            parts=[types.Part(text="I'll remember that your favorite number is 7.")],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text="What is my favorite number plus 3? Just give the number."
                )
            ],
        ),
    ]

    response = gemini_generate(
        contents,
        model=GEMINI_FLASH,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    # Should calculate 7 + 3 = 10
    assert "10" in response or "ten" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_token_logging():
    """Test that gemini_generate logs token usage correctly."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Use a temporary journal path for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["JOURNAL_PATH"] = tmpdir

        # Make a call that should log tokens
        response = gemini_generate(
            "What is the capital of France? One word answer.",
            model=GEMINI_FLASH,
            temperature=0.1,
            max_output_tokens=100,
        )

        assert response is not None
        assert "paris" in response.lower()

        # Check that token log was created
        tokens_dir = Path(tmpdir) / "tokens"
        assert tokens_dir.exists(), "tokens directory should be created"

        log_files = list(tokens_dir.glob("*.jsonl"))
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"

        # Verify log content (JSONL format - read last line)
        with open(log_files[0]) as f:
            lines = f.readlines()
            assert len(lines) >= 1, "Expected at least one log entry"
            log_data = json.loads(lines[-1])

        # Model name may have "models/" prefix
        assert log_data["model"] in [GEMINI_FLASH, f"models/{GEMINI_FLASH}"]
        assert "usage" in log_data
        assert "input_tokens" in log_data["usage"]
        assert "output_tokens" in log_data["usage"]
        assert "total_tokens" in log_data["usage"]
        assert log_data["usage"]["input_tokens"] > 0
        assert log_data["usage"]["output_tokens"] > 0


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_with_lite_model():
    """Test gemini_generate with the Lite model."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test with Lite model (faster, cheaper)
    response = gemini_generate(
        "Say 'hello' in Spanish. One word only.",
        model=GEMINI_LITE,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "hola" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_with_thinking_budget():
    """Test gemini_generate with thinking budget parameter."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test with thinking budget (may not affect non-thinking models)
    response = gemini_generate(
        "What is the square root of 144? Just the number.",
        model=GEMINI_FLASH,
        thinking_budget=1000,
        temperature=0.1,
        max_output_tokens=500,  # Further increased as thinking models may need more
    )

    assert response is not None
    assert isinstance(response, str)
    assert "12" in response or "twelve" in response.lower()


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_max_tokens_error():
    """Test that hitting max_tokens limit produces a clear error."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Intentionally use a very low max_output_tokens to trigger the error
    with pytest.raises(ValueError) as exc_info:
        gemini_generate(
            "Write a detailed essay about the history of computing.",
            model=GEMINI_FLASH,
            temperature=0.3,
            max_output_tokens=10,  # Too low to produce any output
        )

    # Check that the error message is clear and helpful
    error_msg = str(exc_info.value)
    assert "token limit" in error_msg
    assert "10 total" in error_msg  # Should mention the total limit
    assert "Try increasing max_output_tokens" in error_msg


@pytest.mark.integration
@pytest.mark.requires_api
def test_gemini_generate_client_reuse():
    """Test that gemini_generate can reuse an existing client."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from google import genai

    # Create a client
    client = genai.Client(api_key=api_key)

    # Use the same client for multiple calls
    response1 = gemini_generate(
        "What is 1+1? Number only.",
        model=GEMINI_FLASH,
        client=client,
        temperature=0.1,
        max_output_tokens=100,
    )

    response2 = gemini_generate(
        "What is 2+2? Number only.",
        model=GEMINI_FLASH,
        client=client,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response1 is not None
    assert response2 is not None
    assert "2" in response1
    assert "4" in response2


# Async integration tests


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_agenerate_basic_text():
    """Test basic async text generation with gemini_agenerate."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test basic async text generation
    response = await gemini_agenerate(
        "What is 5+5? Reply with just the number.",
        model=GEMINI_FLASH,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response is not None
    assert isinstance(response, str)
    assert "10" in response or "ten" in response.lower()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_agenerate_json_output():
    """Test gemini_agenerate with JSON output mode."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    # Test async JSON output mode
    # Note: Gemini 3 uses implicit thinking which consumes part of max_output_tokens
    response = await gemini_agenerate(
        "Create a JSON object with fields 'status' (value: 'success') and 'value' (value: 100)",
        model=GEMINI_FLASH,
        json_output=True,
        temperature=0.1,
        max_output_tokens=500,
    )

    assert response is not None
    assert isinstance(response, str)

    # Should be valid JSON
    try:
        data = json.loads(response)
        assert isinstance(data, dict)
        assert data.get("status") == "success"
        assert data.get("value") == 100
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON: {response}")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_agenerate_client_reuse():
    """Test that gemini_agenerate can reuse an existing client."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    from google import genai

    # Create a client
    client = genai.Client(api_key=api_key)

    # Use the same client for multiple async calls
    response1 = await gemini_agenerate(
        "What is 3+3? Number only.",
        model=GEMINI_FLASH,
        client=client,
        temperature=0.1,
        max_output_tokens=100,
    )

    response2 = await gemini_agenerate(
        "What is 5+5? Number only.",
        model=GEMINI_FLASH,
        client=client,
        temperature=0.1,
        max_output_tokens=100,
    )

    assert response1 is not None
    assert response2 is not None
    assert "6" in response1
    assert "10" in response2


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_agenerate_token_logging():
    """Test that gemini_agenerate logs token usage correctly."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Use a temporary journal path for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["JOURNAL_PATH"] = tmpdir

        # Make an async call that should log tokens
        response = await gemini_agenerate(
            "What is the capital of Spain? One word answer.",
            model=GEMINI_FLASH,
            temperature=0.1,
            max_output_tokens=100,
        )

        assert response is not None
        assert "madrid" in response.lower()

        # Check that token log was created
        tokens_dir = Path(tmpdir) / "tokens"
        assert tokens_dir.exists(), "tokens directory should be created"

        log_files = list(tokens_dir.glob("*.jsonl"))
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"

        # Verify log content (JSONL format - read last line)
        with open(log_files[0]) as f:
            lines = f.readlines()
            assert len(lines) >= 1, "Expected at least one log entry"
            log_data = json.loads(lines[-1])

        # Model name may have "models/" prefix
        assert log_data["model"] in [GEMINI_FLASH, f"models/{GEMINI_FLASH}"]
        assert "usage" in log_data
        assert "input_tokens" in log_data["usage"]
        assert "output_tokens" in log_data["usage"]
        assert "total_tokens" in log_data["usage"]
        assert log_data["usage"]["input_tokens"] > 0
        assert log_data["usage"]["output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_api
async def test_gemini_agenerate_concurrent_calls():
    """Test that gemini_agenerate can handle concurrent async calls."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    os.environ["GOOGLE_API_KEY"] = api_key
    if journal_path:
        os.environ["JOURNAL_PATH"] = journal_path

    import asyncio

    # Make multiple concurrent async calls
    tasks = [
        gemini_agenerate(
            "What is 1+1? Number only.",
            model=GEMINI_FLASH,
            temperature=0.1,
            max_output_tokens=100,
        ),
        gemini_agenerate(
            "What is 2+2? Number only.",
            model=GEMINI_FLASH,
            temperature=0.1,
            max_output_tokens=100,
        ),
        gemini_agenerate(
            "What is 3+3? Number only.",
            model=GEMINI_FLASH,
            temperature=0.1,
            max_output_tokens=100,
        ),
    ]

    responses = await asyncio.gather(*tasks)

    assert len(responses) == 3
    assert all(r is not None for r in responses)
    assert all(isinstance(r, str) for r in responses)
    # Check expected answers are in responses
    assert "2" in responses[0]
    assert "4" in responses[1]
    assert "6" in responses[2]
