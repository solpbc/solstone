"""Tests for matter view functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest


def test_matter_detail_route_validation():
    """Test that matter_detail route validates domain and matter existence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set up test journal structure
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "20250101120000"
        matter_path.mkdir(parents=True)

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
        (matter_path / "matter.json").write_text(json.dumps(matter_data))

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


def test_get_matter_comprehensive():
    """Test get_matter function returns complete matter data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set up comprehensive test structure
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "20250101120000"
        matter_path.mkdir(parents=True)

        # Create domain.json
        domain_data = {"title": "Test Domain", "description": "A test domain"}
        (domain_path / "domain.json").write_text(json.dumps(domain_data))

        # Create matter metadata
        matter_data = {
            "title": "Test Matter",
            "description": "A comprehensive test matter",
            "status": "active",
            "priority": "high",
            "tags": ["test"],
        }
        (matter_path / "matter.json").write_text(json.dumps(matter_data))

        # Create matter activity log
        matter_log = [
            {
                "timestamp": "2025-01-01T12:00:00Z",
                "type": "created",
                "description": "Matter created",
            },
            {
                "timestamp": "2025-01-01T12:30:00Z",
                "type": "progress",
                "description": "Started work",
            },
        ]
        (matter_path / "matter.jsonl").write_text(
            "\n".join(json.dumps(entry) for entry in matter_log)
        )

        # Create objectives
        objectives_dir = matter_path / "objectives"
        obj1_dir = objectives_dir / "20250101140000"
        obj1_dir.mkdir(parents=True)

        obj1_data = {
            "title": "First Objective",
            "description": "Test objective 1",
            "status": "in_progress",
            "priority": "high",
        }
        (obj1_dir / "20250101140000.json").write_text(json.dumps(obj1_data))

        obj1_log = [
            {
                "timestamp": "2025-01-01T14:00:00Z",
                "type": "created",
                "description": "Objective created",
            },
            {
                "timestamp": "2025-01-01T14:30:00Z",
                "type": "progress",
                "description": "Work started",
            },
        ]
        (obj1_dir / "20250101140000.jsonl").write_text(
            "\n".join(json.dumps(entry) for entry in obj1_log)
        )

        # Create attachments
        attachments_dir = matter_path / "attachments"
        attachments_dir.mkdir()

        # Create a test file and its metadata
        (attachments_dir / "test.txt").write_text("Test content")
        attachment_meta = {
            "title": "Test File",
            "description": "A test attachment",
            "size": 12,
            "mime_type": "text/plain",
        }
        (attachments_dir / "test.json").write_text(json.dumps(attachment_meta))

        # Test the function
        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            from think.utils import get_matter

            result = get_matter("test-domain", "20250101120000")

            # Verify matter metadata
            assert result["metadata"]["title"] == "Test Matter"
            assert result["metadata"]["status"] == "active"

            # Verify matter activity log
            assert len(result["activity_log"]) == 2
            assert result["activity_log"][0]["type"] == "created"
            assert result["activity_log"][1]["type"] == "progress"

            # Verify objectives
            assert len(result["objectives"]) == 1
            assert "20250101140000" in result["objectives"]
            obj1 = result["objectives"]["20250101140000"]
            assert obj1["metadata"]["title"] == "First Objective"
            assert obj1["metadata"]["status"] == "in_progress"
            assert len(obj1["activity_log"]) == 2

            # Verify attachments
            assert len(result["attachments"]) == 1
            assert "test" in result["attachments"]
            att = result["attachments"]["test"]
            assert att["title"] == "Test File"
            assert att["mime_type"] == "text/plain"


def test_get_matter_missing_files():
    """Test get_matter handles missing optional files gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "20250101120000"
        matter_path.mkdir(parents=True)

        # Create only minimal required structure (just the directory)
        # No matter.json, matter.jsonl, objectives, or attachments

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            from think.utils import get_matter

            result = get_matter("test-domain", "20250101120000")

            # Should return empty structures, not fail
            assert result["metadata"] == {}
            assert result["activity_log"] == []
            assert result["objectives"] == {}
            assert result["attachments"] == {}


def test_get_matter_error_handling():
    """Test get_matter error handling for non-existent matters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            from think.utils import get_matter

            # Test non-existent domain
            with pytest.raises(
                FileNotFoundError,
                match="Matter .* not found in domain nonexistent-domain",
            ):
                get_matter("nonexistent-domain", "20250101120000")

            # Test non-existent matter in existing domain
            domain_path = journal_path / "domains" / "test-domain"
            domain_path.mkdir(parents=True)
            (domain_path / "domain.json").write_text('{"title": "Test"}')

            with pytest.raises(
                FileNotFoundError, match="Matter .* not found in domain test-domain"
            ):
                get_matter("test-domain", "99999999999999")


def test_get_matter_attachment_validation():
    """Test get_matter only includes attachments with corresponding files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "20250101120000"
        attachments_dir = matter_path / "attachments"
        attachments_dir.mkdir(parents=True)

        # Create attachment file and metadata
        (attachments_dir / "valid.txt").write_text("Valid content")
        (attachments_dir / "valid.json").write_text(
            '{"title": "Valid", "description": "Has file"}'
        )

        # Create orphaned metadata (no corresponding file)
        (attachments_dir / "orphan.json").write_text(
            '{"title": "Orphan", "description": "No file"}'
        )

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            from think.utils import get_matter

            result = get_matter("test-domain", "20250101120000")

            # Should only include attachment with corresponding file
            assert len(result["attachments"]) == 1
            assert "valid" in result["attachments"]
            assert "orphan" not in result["attachments"]
