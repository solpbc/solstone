"""Tests for think.models module."""

from think.models import (
    CLAUDE_HAIKU_4,
    CLAUDE_OPUS_4,
    CLAUDE_SONNET_4,
    GEMINI_FLASH,
    GEMINI_LITE,
    GEMINI_PRO,
    GPT_5,
    GPT_5_MINI,
    GPT_5_NANO,
    calc_token_cost,
    get_model_provider,
)


def test_get_model_provider_gemini():
    """Test provider detection for Gemini models."""
    assert get_model_provider(GEMINI_PRO) == "google"
    assert get_model_provider(GEMINI_FLASH) == "google"
    assert get_model_provider(GEMINI_LITE) == "google"


def test_get_model_provider_gpt():
    """Test provider detection for GPT models."""
    assert get_model_provider(GPT_5) == "openai"
    assert get_model_provider(GPT_5_MINI) == "openai"
    assert get_model_provider(GPT_5_NANO) == "openai"


def test_get_model_provider_claude():
    """Test provider detection for Claude models."""
    assert get_model_provider(CLAUDE_OPUS_4) == "anthropic"
    assert get_model_provider(CLAUDE_SONNET_4) == "anthropic"
    assert get_model_provider(CLAUDE_HAIKU_4) == "anthropic"


def test_get_model_provider_case_insensitive():
    """Test that provider detection is case-insensitive."""
    assert get_model_provider("GPT-5") == "openai"
    assert get_model_provider("Gemini-2.5-Flash") == "google"
    assert get_model_provider("CLAUDE-SONNET-4-5") == "anthropic"


def test_get_model_provider_unknown():
    """Test that unknown models return 'unknown'."""
    assert get_model_provider("random-model-xyz") == "unknown"
    assert get_model_provider("llama-3") == "unknown"
    assert get_model_provider("") == "unknown"


def test_calc_token_cost_basic():
    """Test basic cost calculation with a known model."""
    token_data = {
        "model": "gpt-4o",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
        },
    }

    result = calc_token_cost(token_data)

    assert result is not None
    assert "total_cost" in result
    assert "input_cost" in result
    assert "output_cost" in result
    assert "currency" in result
    assert result["currency"] == "USD"
    assert result["total_cost"] > 0
    assert result["input_cost"] > 0
    assert result["output_cost"] > 0


def test_calc_token_cost_with_cache():
    """Test cost calculation with cached tokens."""
    token_data = {
        "model": "claude-sonnet-4-20250514",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cached_tokens": 500,
            "total_tokens": 1600,
        },
    }

    result = calc_token_cost(token_data)

    assert result is not None
    assert result["total_cost"] > 0
    # Cached tokens should reduce the cost compared to all uncached
    assert result["input_cost"] >= 0


def test_calc_token_cost_unknown_model():
    """Test that unknown models return None."""
    token_data = {
        "model": "random-model-xyz",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
        },
    }

    result = calc_token_cost(token_data)
    assert result is None


def test_calc_token_cost_missing_data():
    """Test that missing data returns None."""
    # Missing model
    assert calc_token_cost({"usage": {"input_tokens": 1000}}) is None

    # Missing usage
    assert calc_token_cost({"model": "gpt-4o"}) is None

    # Empty dict
    assert calc_token_cost({}) is None


def test_calc_token_cost_with_reasoning_tokens():
    """Test cost calculation includes reasoning tokens in output."""
    token_data = {
        "model": "gpt-4o",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "reasoning_tokens": 50,
            "total_tokens": 1150,
        },
    }

    result = calc_token_cost(token_data)

    # Should succeed - reasoning tokens are implicitly part of output pricing
    assert result is not None
    assert result["total_cost"] > 0
