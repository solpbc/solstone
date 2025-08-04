"""Tests for matter view functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_matter_detail_route_validation():
    """Test that matter_detail route validates domain and matter existence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set up test journal structure
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matters_path = domain_path / "matters"
        matters_path.mkdir(parents=True)

        # Create domain.json
        domain_data = {
            "title": "Test Domain",
            "description": "A test domain",
            "color": "#007bff",
            "emoji": "🧪",
        }
        (domain_path / "domain.json").write_text(json.dumps(domain_data))

        # Create matter
        matter_data = {
            "title": "Test Matter",
            "description": "A test matter",
            "status": "active",
            "priority": "high",
            "tags": ["test"],
            "created": "2025-01-01T12:00:00Z",
        }
        (matters_path / "20250101120000.json").write_text(json.dumps(matter_data))

        # Mock the environment
        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            # Import here to avoid audio dependency issues
            import sys

            if "dream.utils" in sys.modules:
                del sys.modules["dream.utils"]
            if "dream.views.domains" in sys.modules:
                del sys.modules["dream.views.domains"]

            # Test get_domains function
            from think.utils import get_domains

            domains = get_domains()
            assert "test-domain" in domains
            assert domains["test-domain"]["title"] == "Test Domain"

            # Test get_matters function
            from think.utils import get_matters

            matters = get_matters("test-domain")
            assert "20250101120000" in matters
            assert matters["20250101120000"]["title"] == "Test Matter"


def test_matter_detail_template_exists():
    """Test that matter_detail.html template exists and has expected structure."""
    template_path = Path("dream/templates/matter_detail.html")
    assert template_path.exists(), "matter_detail.html template should exist"

    content = template_path.read_text()
    # Check for key template elements
    assert "matter-header" in content
    assert "Objectives" in content
    assert "Logs" in content
    assert "Attachments" in content
    assert "matter_data" in content


def test_domain_detail_template_has_matter_links():
    """Test that domain_detail.html template includes matter navigation."""
    template_path = Path("dream/templates/domain_detail.html")
    assert template_path.exists(), "domain_detail.html template should exist"

    content = template_path.read_text()
    # Check that matter click navigation is implemented
    assert "window.location.href" in content
    assert "/domains/" in content
    assert "/matters/" in content
