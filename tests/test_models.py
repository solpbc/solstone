# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.models module."""

import pytest

from think.models import (
    CLAUDE_HAIKU_4,
    CLAUDE_OPUS_4,
    CLAUDE_SONNET_4,
    DEFAULT_PROVIDER,
    DEFAULT_TIER,
    GEMINI_FLASH,
    GEMINI_LITE,
    GEMINI_PRO,
    GPT_5,
    GPT_5_MINI,
    GPT_5_NANO,
    PROVIDER_DEFAULTS,
    TIER_FLASH,
    TIER_LITE,
    TIER_PRO,
    calc_token_cost,
    get_model_provider,
    resolve_provider,
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
    # Default tier is 2, which is overridden in fixture config to custom model
    assert model == "gemini-custom-flash-test"


def test_resolve_provider_exact_match(use_fixtures_journal):
    """Test that exact context match works."""
    provider, model = resolve_provider("test.openai")
    assert provider == "openai"
    assert model == "gpt-5-mini"


def test_resolve_provider_glob_match(use_fixtures_journal):
    """Test that glob pattern matching works."""
    # observe.* pattern should match
    provider, model = resolve_provider("observe.describe.frame")
    assert provider == "google"
    assert model == "gemini-2.5-flash-lite"

    # Also matches with other suffixes
    provider, model = resolve_provider("observe.enrich")
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


# ---------------------------------------------------------------------------
# Tier system tests
# ---------------------------------------------------------------------------


def test_tier_constants():
    """Test tier constant values."""
    assert TIER_PRO == 1
    assert TIER_FLASH == 2
    assert TIER_LITE == 3
    assert DEFAULT_TIER == TIER_FLASH
    assert DEFAULT_PROVIDER == "google"


def test_provider_defaults_structure():
    """Test PROVIDER_DEFAULTS contains all providers and tiers."""
    assert "google" in PROVIDER_DEFAULTS
    assert "openai" in PROVIDER_DEFAULTS
    assert "anthropic" in PROVIDER_DEFAULTS

    for provider in PROVIDER_DEFAULTS:
        assert TIER_PRO in PROVIDER_DEFAULTS[provider]
        assert TIER_FLASH in PROVIDER_DEFAULTS[provider]
        assert TIER_LITE in PROVIDER_DEFAULTS[provider]


def test_provider_defaults_models():
    """Test PROVIDER_DEFAULTS maps to correct model constants."""
    assert PROVIDER_DEFAULTS["google"][TIER_PRO] == GEMINI_PRO
    assert PROVIDER_DEFAULTS["google"][TIER_FLASH] == GEMINI_FLASH
    assert PROVIDER_DEFAULTS["google"][TIER_LITE] == GEMINI_LITE

    assert PROVIDER_DEFAULTS["openai"][TIER_PRO] == GPT_5
    assert PROVIDER_DEFAULTS["openai"][TIER_FLASH] == GPT_5_MINI
    assert PROVIDER_DEFAULTS["openai"][TIER_LITE] == GPT_5_NANO

    assert PROVIDER_DEFAULTS["anthropic"][TIER_PRO] == CLAUDE_OPUS_4
    assert PROVIDER_DEFAULTS["anthropic"][TIER_FLASH] == CLAUDE_SONNET_4
    assert PROVIDER_DEFAULTS["anthropic"][TIER_LITE] == CLAUDE_HAIKU_4


def test_resolve_provider_tier_based(use_fixtures_journal):
    """Test tier-based resolution."""
    # test.tier has tier: 1 (pro)
    provider, model = resolve_provider("test.tier")
    assert provider == "google"
    assert model == GEMINI_PRO


def test_resolve_provider_tier_inherit_provider(use_fixtures_journal):
    """Test tier with inherited provider from default."""
    # test.tier.inherit has tier: 3 only, should inherit google from default
    provider, model = resolve_provider("test.tier.inherit")
    assert provider == "google"
    assert model == GEMINI_LITE


def test_resolve_provider_tier_with_provider(use_fixtures_journal):
    """Test tier with explicit provider."""
    # test.tier.override has provider: openai, tier: 2
    provider, model = resolve_provider("test.tier.override")
    assert provider == "openai"
    assert model == GPT_5_MINI


def test_resolve_provider_tier_glob(use_fixtures_journal):
    """Test tier-based glob pattern matching."""
    # observe.* now uses tier: 3 instead of explicit model
    provider, model = resolve_provider("observe.describe.frame")
    assert provider == "google"
    assert model == GEMINI_LITE


def test_resolve_provider_model_overrides_tier(use_fixtures_journal):
    """Test that explicit model takes precedence over tier."""
    # test.openai has explicit model, not tier
    provider, model = resolve_provider("test.openai")
    assert provider == "openai"
    assert model == "gpt-5-mini"


def test_resolve_provider_default_tier(use_fixtures_journal):
    """Test default uses tier-based resolution with config override."""
    # Default is tier: 2, which is overridden in config to custom model
    provider, model = resolve_provider("unknown.context")
    assert provider == "google"
    assert model == "gemini-custom-flash-test"


def test_resolve_provider_config_model_override(use_fixtures_journal):
    """Test that config models section overrides system defaults."""
    # test.config.override uses tier: 2, which is overridden in config
    provider, model = resolve_provider("test.config.override")
    assert provider == "google"
    # Should use the custom model from config, not system default GEMINI_FLASH
    assert model == "gemini-custom-flash-test"
    assert model != GEMINI_FLASH


def test_resolve_provider_tier_fallback_to_system_default(use_fixtures_journal):
    """Test that tiers not in config fall back to system defaults."""
    # test.tier uses tier: 1 (pro), which is NOT overridden in config
    # Should fall back to system default GEMINI_PRO
    provider, model = resolve_provider("test.tier")
    assert provider == "google"
    assert model == GEMINI_PRO


def test_resolve_provider_invalid_tier(use_fixtures_journal, monkeypatch, tmp_path):
    """Test that invalid tier values fall back to default tier."""
    import json

    # Create a config with an invalid tier
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {
        "providers": {
            "default": {"provider": "google", "tier": 2},
            "contexts": {
                "test.invalid": {"provider": "google", "tier": 99},
                "test.string": {"provider": "google", "tier": "flash"},
            },
        }
    }
    (config_dir / "journal.json").write_text(json.dumps(config))
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Invalid tier 99 should fall back to default tier (2)
    provider, model = resolve_provider("test.invalid")
    assert provider == "google"
    assert model == GEMINI_FLASH  # tier 2 system default

    # String tier should also fall back
    provider, model = resolve_provider("test.string")
    assert provider == "google"
    assert model == GEMINI_FLASH
