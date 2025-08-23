"""Tests for the gemini_generate wrapper function."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from think.models import GEMINI_FLASH, GEMINI_LITE, gemini_generate


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
@patch("think.models.genai.Client")
def test_gemini_generate_basic(mock_client_class):
    """Test basic gemini_generate functionality."""
    # Setup mock
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=100,
        candidates_token_count=50,
        cached_content_token_count=0,
        thoughts_token_count=0,
        total_token_count=150,
    )
    mock_client.models.generate_content.return_value = mock_response
    
    # Call function
    text, usage = gemini_generate("Test prompt")
    
    # Verify
    assert text == "Test response"
    assert usage.prompt_token_count == 100
    assert usage.candidates_token_count == 50
    
    # Check that client was called correctly
    mock_client.models.generate_content.assert_called_once()
    call_args = mock_client.models.generate_content.call_args
    assert call_args[1]["model"] == GEMINI_FLASH
    assert call_args[1]["contents"] == ["Test prompt"]


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
@patch("think.models.genai.Client")
def test_gemini_generate_with_options(mock_client_class):
    """Test gemini_generate with various options."""
    # Setup mock
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = '{"result": "success"}'
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response
    
    # Call with options
    text, usage = gemini_generate(
        ["Part 1", "Part 2"],
        model=GEMINI_LITE,
        temperature=0.5,
        max_output_tokens=1024,
        system_instruction="Be helpful",
        json_output=True,
        thinking_budget=2048,
    )
    
    # Verify
    assert text == '{"result": "success"}'
    assert usage is None
    
    # Check config
    call_args = mock_client.models.generate_content.call_args
    assert call_args[1]["model"] == GEMINI_LITE
    assert call_args[1]["contents"] == ["Part 1", "Part 2"]
    
    config = call_args[1]["config"]
    # Check that the config object has the expected attributes
    assert hasattr(config, "__dict__") or hasattr(config, "_config")


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
@patch("think.models.genai.Client")
def test_gemini_generate_token_logging(mock_client_class):
    """Test that token logging works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set JOURNAL_PATH
        os.environ["JOURNAL_PATH"] = tmpdir
        
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=25,
            thoughts_token_count=10,
            total_token_count=185,
        )
        mock_client.models.generate_content.return_value = mock_response
        
        # Call function (logging is always enabled now)
        text, usage = gemini_generate(
            "Test prompt",
            model=GEMINI_FLASH,
        )
        
        # Check that log file was created
        tokens_dir = Path(tmpdir) / "tokens"
        assert tokens_dir.exists()
        
        # Find the log file
        log_files = list(tokens_dir.glob("*.json"))
        assert len(log_files) == 1
        
        # Read and verify log content
        with open(log_files[0]) as f:
            log_data = json.load(f)
        
        assert log_data["model"] == GEMINI_FLASH
        assert log_data["context"] is not None  # Should auto-detect
        assert log_data["usage"]["prompt_tokens"] == 100
        assert log_data["usage"]["candidates_tokens"] == 50
        assert log_data["usage"]["cached_tokens"] == 25
        assert log_data["usage"]["thoughts_tokens"] == 10
        assert log_data["usage"]["total_tokens"] == 185


@patch.dict(os.environ, {}, clear=True)
@patch("think.models.load_dotenv")
def test_gemini_generate_no_api_key(mock_load_dotenv):
    """Test that gemini_generate raises error when no API key."""
    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        gemini_generate("Test prompt")


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
@patch("think.models.genai.Client")
def test_gemini_generate_string_normalization(mock_client_class):
    """Test that string contents are normalized to list."""
    # Setup mock
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response
    
    # Test with string
    gemini_generate("Single string")
    
    # Verify it was converted to list
    call_args = mock_client.models.generate_content.call_args
    assert call_args[1]["contents"] == ["Single string"]
    
    # Test with list (should remain unchanged)
    gemini_generate(["Already", "a", "list"])
    
    call_args = mock_client.models.generate_content.call_args
    assert call_args[1]["contents"] == ["Already", "a", "list"]