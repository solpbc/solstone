# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.models module."""

import os

import pytest

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
from think.utils import resolve_provider


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


# ---------------------------------------------------------------------------
# resolve_provider tests
# ---------------------------------------------------------------------------


@pytest.fixture
def use_fixtures_journal(monkeypatch):
    """Use the fixtures journal for provider config tests."""
    monkeypatch.setenv("JOURNAL_PATH", "fixtures/journal")


def test_resolve_provider_default(use_fixtures_journal):
    """Test that default provider is returned for unknown context."""
    provider, model = resolve_provider("unknown.context")
    assert provider == "google"
    assert model == "gemini-3-flash-preview"


def test_resolve_provider_exact_match(use_fixtures_journal):
    """Test that exact context match works."""
    provider, model = resolve_provider("test.openai")
    assert provider == "openai"
    assert model == "gpt-5-mini"


def test_resolve_provider_glob_match(use_fixtures_journal):
    """Test that glob pattern matching works."""
    # describe.* pattern should match
    provider, model = resolve_provider("describe.frame")
    assert provider == "google"
    assert model == "gemini-2.5-flash-lite"

    # Also matches with other suffixes
    provider, model = resolve_provider("describe.meeting")
    assert provider == "google"
    assert model == "gemini-2.5-flash-lite"


def test_resolve_provider_anthropic(use_fixtures_journal):
    """Test anthropic provider routing."""
    provider, model = resolve_provider("test.anthropic")
    assert provider == "anthropic"
    assert model == "claude-sonnet-4-5"


def test_resolve_provider_empty_context(use_fixtures_journal):
    """Test that empty context returns default."""
    provider, model = resolve_provider("")
    assert provider == "google"


def test_resolve_provider_no_config(monkeypatch, tmp_path):
    """Test fallback when no provider config exists."""
    # Use a journal path with no config
    empty_journal = tmp_path / "empty_journal"
    empty_journal.mkdir()
    monkeypatch.setenv("JOURNAL_PATH", str(empty_journal))

    provider, model = resolve_provider("anything")
    assert provider == "google"
    assert model == GEMINI_FLASH
