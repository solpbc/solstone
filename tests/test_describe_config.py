# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/describe.py category discovery and configuration."""

from observe import describe as describe_module


def test_categories_discovered():
    """Test that categories are discovered on import."""
    CATEGORIES = describe_module.CATEGORIES

    # Should have discovered at least some categories
    assert isinstance(CATEGORIES, dict)
    assert len(CATEGORIES) > 0


def test_categories_have_required_fields():
    """Test that all categories have required metadata."""
    CATEGORIES = describe_module.CATEGORIES

    for category, metadata in CATEGORIES.items():
        # Every category must have description
        assert "description" in metadata, f"Category {category} missing 'description'"
        assert isinstance(metadata["description"], str)
        assert len(metadata["description"]) > 0

        # Every category should have output field (defaulted if not set)
        assert "output" in metadata, f"Category {category} missing 'output'"
        assert metadata["output"] in ("json", "markdown")

        # Every category should have context field (for provider resolution)
        assert "context" in metadata, f"Category {category} missing 'context'"
        assert metadata["context"].startswith("observe.describe.")


def test_extractable_categories_have_prompts():
    """Test that extractable categories have valid prompts loaded."""
    CATEGORIES = describe_module.CATEGORIES

    extractable_count = 0
    for category, metadata in CATEGORIES.items():
        if "prompt" in metadata:
            extractable_count += 1
            assert isinstance(metadata["prompt"], str)
            assert len(metadata["prompt"]) > 0, f"Category {category} has empty prompt"

    # Sanity check: we should have at least some extractable categories
    assert extractable_count > 0, "No extractable categories found"


def test_categorization_prompt_built():
    """Test that categorization prompt is built correctly."""
    prompt = describe_module.CATEGORIZATION_PROMPT

    # Should contain all category descriptions
    for category, metadata in describe_module.CATEGORIES.items():
        assert f"- {category}:" in prompt
        assert metadata["description"] in prompt

    # Should have the template structure
    assert "primary" in prompt
    assert "secondary" in prompt
    assert "overlap" in prompt
    assert "Categories (choose one):" in prompt


def test_categorization_prompt_alphabetical():
    """Test that categories in prompt are alphabetically ordered."""
    prompt = describe_module.CATEGORIZATION_PROMPT

    # Extract category lines from prompt
    lines = prompt.split("\n")
    category_lines = [l for l in lines if l.startswith("- ") and ":" in l]

    # Extract category names
    categories = [l.split(":")[0].replace("- ", "") for l in category_lines]

    # Should be sorted
    assert categories == sorted(categories)
