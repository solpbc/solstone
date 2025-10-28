"""Tests for observe/describe.py config loading."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_config_loads_successfully():
    """Test that config loads successfully on import."""
    from observe.describe import CONFIG

    assert "text_extraction_categories" in CONFIG
    assert isinstance(CONFIG["text_extraction_categories"], list)
    assert len(CONFIG["text_extraction_categories"]) > 0


def test_config_has_expected_categories():
    """Test that config contains text extraction categories."""
    from observe.describe import CONFIG

    # Verify structure, not specific values (config can change)
    assert "text_extraction_categories" in CONFIG
    assert isinstance(CONFIG["text_extraction_categories"], list)
    # Should have at least one category configured
    assert len(CONFIG["text_extraction_categories"]) > 0
    # All entries should be non-empty strings
    for category in CONFIG["text_extraction_categories"]:
        assert isinstance(category, str)
        assert len(category) > 0


def test_config_loading_with_missing_file(tmp_path):
    """Test that config loading fails gracefully when file is missing."""
    from observe.describe import _load_config

    with patch("observe.describe.Path") as mock_path:
        # Mock the config path to point to non-existent file
        mock_config_path = tmp_path / "nonexistent.json"
        mock_path.return_value.parent = tmp_path
        mock_path.return_value.__truediv__.return_value = mock_config_path

        with pytest.raises(SystemExit) as exc_info:
            _load_config()
        assert exc_info.value.code == 1


def test_config_loading_with_invalid_json(tmp_path):
    """Test that config loading fails gracefully with invalid JSON."""
    from observe.describe import _load_config

    # Create invalid JSON file
    invalid_json_path = tmp_path / "describe.json"
    invalid_json_path.write_text("{ invalid json }")

    with patch("observe.describe.Path") as mock_path:
        mock_path.return_value.parent = tmp_path
        mock_path.return_value.__truediv__.return_value = invalid_json_path

        with pytest.raises(SystemExit) as exc_info:
            _load_config()
        assert exc_info.value.code == 1


def test_config_loading_with_valid_json(tmp_path):
    """Test that config loading succeeds with valid JSON."""
    from observe.describe import _load_config

    # Create valid JSON file
    valid_json_path = tmp_path / "describe.json"
    config_data = {"text_extraction_categories": ["code", "messaging", "reading"]}
    valid_json_path.write_text(json.dumps(config_data))

    with patch("observe.describe.Path") as mock_path:
        mock_path.return_value.parent = tmp_path
        mock_path.return_value.__truediv__.return_value = valid_json_path

        config = _load_config()
        assert config == config_data
