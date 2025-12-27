"""Tests for observe/describe.py category discovery and configuration."""

from observe import describe as describe_module


def test_categories_discovered():
    """Test that categories are discovered on import."""
    CATEGORIES = describe_module.CATEGORIES

    # Should have discovered all 9 categories
    assert len(CATEGORIES) == 9

    # All expected categories should be present
    expected = [
        "terminal",
        "code",
        "messaging",
        "meeting",
        "browsing",
        "reading",
        "media",
        "gaming",
        "productivity",
    ]
    for cat in expected:
        assert cat in CATEGORIES, f"Expected category {cat} not found"


def test_categories_have_required_fields():
    """Test that all categories have required metadata."""
    CATEGORIES = describe_module.CATEGORIES

    for category, metadata in CATEGORIES.items():
        # Every category must have description
        assert "description" in metadata, f"Category {category} missing 'description'"
        assert isinstance(metadata["description"], str)
        assert len(metadata["description"]) > 0

        # Every category should have followup field (defaulted if not set)
        assert "followup" in metadata, f"Category {category} missing 'followup'"
        assert isinstance(metadata["followup"], bool)

        # Every category should have output field (defaulted if not set)
        assert "output" in metadata, f"Category {category} missing 'output'"
        assert metadata["output"] in ("json", "markdown")

        # Every category should have model field (mapped from iq)
        assert "model" in metadata, f"Category {category} missing 'model'"


def test_followup_categories_have_prompts():
    """Test that categories with followup=true have prompt loaded."""
    CATEGORIES = describe_module.CATEGORIES

    for category, metadata in CATEGORIES.items():
        if metadata["followup"]:
            assert (
                "prompt" in metadata
            ), f"Category {category} has followup=true but no prompt"
            assert isinstance(metadata["prompt"], str)
            assert len(metadata["prompt"]) > 0


def test_non_followup_categories():
    """Test that non-followup categories don't have prompts."""
    CATEGORIES = describe_module.CATEGORIES

    non_followup = ["terminal", "code", "media", "gaming"]
    for category in non_followup:
        assert category in CATEGORIES
        assert CATEGORIES[category]["followup"] is False
        assert "prompt" not in CATEGORIES[category]


def test_meeting_category_config():
    """Test that meeting category has correct configuration."""
    CATEGORIES = describe_module.CATEGORIES

    assert "meeting" in CATEGORIES
    meeting = CATEGORIES["meeting"]
    assert meeting["followup"] is True
    assert meeting["output"] == "json"
    assert meeting["iq"] == "flash"  # Meeting needs flash for complex JSON output


def test_text_categories_config():
    """Test that text-based categories have correct configuration."""
    CATEGORIES = describe_module.CATEGORIES

    text_categories = ["messaging", "browsing", "reading", "productivity"]
    for category in text_categories:
        assert category in CATEGORIES
        cat_meta = CATEGORIES[category]
        assert cat_meta["followup"] is True
        assert cat_meta["output"] == "markdown"

    # Messaging uses flash for better text extraction
    assert CATEGORIES["messaging"]["iq"] == "flash"
    # Others default to lite
    for category in ["browsing", "reading", "productivity"]:
        assert CATEGORIES[category].get("iq", "lite") == "lite"


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
