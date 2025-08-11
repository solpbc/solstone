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
        matter_path = domain_path / "matter_1"
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
            assert "matter_1" in matters
            assert matters["matter_1"]["title"] == "Test Matter"


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
        matter_path = domain_path / "matter_1"
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
        (matter_path / "activity_log.jsonl").write_text(
            "\n".join(json.dumps(entry) for entry in matter_log)
        )

        # Create objectives (new format: objective_<name> directories)
        obj1_dir = matter_path / "objective_ui_implementation"
        obj1_dir.mkdir(parents=True)

        (obj1_dir / "OBJECTIVE.md").write_text(
            "# UI Implementation\n\nCreate a comprehensive matter detail view."
        )
        (obj1_dir / "OUTCOME.md").write_text(
            "# Completed Successfully\n\nThe UI has been implemented with all features."
        )

        obj2_dir = matter_path / "objective_api_integration"
        obj2_dir.mkdir(parents=True)

        (obj2_dir / "OBJECTIVE.md").write_text(
            "# API Integration\n\nIntegrate backend API support."
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

            result = get_matter("test-domain", "matter_1")

            # Verify matter metadata
            assert result["metadata"]["title"] == "Test Matter"
            assert result["metadata"]["status"] == "active"

            # Verify matter activity log
            assert len(result["activity_log"]) == 2
            assert result["activity_log"][0]["type"] == "created"
            assert result["activity_log"][1]["type"] == "progress"

            # Verify objectives (new structure)
            assert len(result["objectives"]) == 2

            # Test completed objective
            assert "ui_implementation" in result["objectives"]
            obj1 = result["objectives"]["ui_implementation"]
            assert obj1["name"] == "ui_implementation"
            assert "UI Implementation" in obj1["objective"]
            assert obj1["outcome"] is not None
            assert "Completed Successfully" in obj1["outcome"]
            assert obj1["created"] is not None
            assert obj1["modified"] is not None

            # Test in-progress objective
            assert "api_integration" in result["objectives"]
            obj2 = result["objectives"]["api_integration"]
            assert obj2["name"] == "api_integration"
            assert "API Integration" in obj2["objective"]
            assert obj2["outcome"] is None

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
        matter_path = domain_path / "matter_1"
        matter_path.mkdir(parents=True)

        # Create only minimal required structure (just the directory)
        # No matter.json, activity_log.jsonl, objectives, or attachments

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            from think.utils import get_matter

            result = get_matter("test-domain", "matter_1")

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
                get_matter("nonexistent-domain", "matter_1")

            # Test non-existent matter in existing domain
            domain_path = journal_path / "domains" / "test-domain"
            domain_path.mkdir(parents=True)
            (domain_path / "domain.json").write_text('{"title": "Test"}')

            with pytest.raises(
                FileNotFoundError, match="Matter .* not found in domain test-domain"
            ):
                get_matter("test-domain", "matter_999")


def test_get_matter_attachment_validation():
    """Test get_matter only includes attachments with corresponding files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "matter_1"
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

            result = get_matter("test-domain", "matter_1")

            # Should only include attachment with corresponding file
            assert len(result["attachments"]) == 1
            assert "valid" in result["attachments"]
            assert "orphan" not in result["attachments"]


def test_get_matter_new_objective_structure():
    """Test get_matter handles the new objective_<name> directory structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "matter_1"
        matter_path.mkdir(parents=True)

        # Create domain.json
        (domain_path / "domain.json").write_text('{"title": "Test Domain"}')

        # Create objectives in new format: objective_<name> directories
        # Completed objective with both OBJECTIVE.md and OUTCOME.md
        completed_obj_dir = matter_path / "objective_ui_design"
        completed_obj_dir.mkdir(parents=True)
        (completed_obj_dir / "OBJECTIVE.md").write_text(
            "# UI Design\n\nDesign the user interface for the application.\n\n"
            "This includes wireframes, mockups, and design specifications."
        )
        (completed_obj_dir / "OUTCOME.md").write_text(
            "# Design Completed\n\nAll UI designs have been finalized and approved.\n\n"
            "The designs are ready for implementation."
        )

        # In-progress objective with only OBJECTIVE.md
        in_progress_obj_dir = matter_path / "objective_database_schema"
        in_progress_obj_dir.mkdir(parents=True)
        (in_progress_obj_dir / "OBJECTIVE.md").write_text(
            "# Database Schema\n\nDesign and implement the database schema.\n\n"
            "Include tables, relationships, and indexes."
        )

        # Objective with invalid name (should be ignored)
        invalid_obj_dir = matter_path / "not_an_objective"
        invalid_obj_dir.mkdir(parents=True)
        (invalid_obj_dir / "OBJECTIVE.md").write_text("This should be ignored")

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            from think.utils import get_matter

            result = get_matter("test-domain", "matter_1")

            # Should find 2 objectives (not the invalid one)
            assert len(result["objectives"]) == 2

            # Test completed objective
            assert "ui_design" in result["objectives"]
            ui_obj = result["objectives"]["ui_design"]
            assert ui_obj["name"] == "ui_design"
            assert "UI Design" in ui_obj["objective"]
            assert "Design the user interface" in ui_obj["objective"]
            assert ui_obj["outcome"] is not None
            assert "Design Completed" in ui_obj["outcome"]
            assert "All UI designs have been finalized" in ui_obj["outcome"]
            assert isinstance(ui_obj["created"], (int, float))
            assert isinstance(ui_obj["modified"], (int, float))

            # Test in-progress objective
            assert "database_schema" in result["objectives"]
            db_obj = result["objectives"]["database_schema"]
            assert db_obj["name"] == "database_schema"
            assert "Database Schema" in db_obj["objective"]
            assert "Design and implement the database" in db_obj["objective"]
            assert db_obj["outcome"] is None
            assert isinstance(db_obj["created"], (int, float))
            assert isinstance(db_obj["modified"], (int, float))

            # Invalid objective should not be included
            assert "not_an_objective" not in result["objectives"]


def test_get_matter_objective_edge_cases():
    """Test get_matter handles edge cases in objective structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "matter_1"
        matter_path.mkdir(parents=True)

        # Create domain.json
        (domain_path / "domain.json").write_text('{"title": "Test Domain"}')

        # Objective with missing OBJECTIVE.md file
        missing_obj_dir = matter_path / "objective_missing_file"
        missing_obj_dir.mkdir(parents=True)
        (missing_obj_dir / "OUTCOME.md").write_text("Has outcome but no objective")

        # Objective with empty OBJECTIVE.md
        empty_obj_dir = matter_path / "objective_empty_file"
        empty_obj_dir.mkdir(parents=True)
        (empty_obj_dir / "OBJECTIVE.md").write_text("")

        # Objective with only whitespace in files
        whitespace_obj_dir = matter_path / "objective_whitespace"
        whitespace_obj_dir.mkdir(parents=True)
        (whitespace_obj_dir / "OBJECTIVE.md").write_text("   \n\n   ")
        (whitespace_obj_dir / "OUTCOME.md").write_text("   \n\n   ")

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            from think.utils import get_matter

            result = get_matter("test-domain", "matter_1")

            # Should find all 3 objectives even with edge cases
            assert len(result["objectives"]) == 3

            # Test missing OBJECTIVE.md
            missing_obj = result["objectives"]["missing_file"]
            assert missing_obj["objective"] == ""  # Should be empty string, not None
            assert "Has outcome but no objective" in missing_obj["outcome"]

            # Test empty OBJECTIVE.md
            empty_obj = result["objectives"]["empty_file"]
            assert empty_obj["objective"] == ""

            # Test whitespace-only files (should be stripped)
            whitespace_obj = result["objectives"]["whitespace"]
            assert whitespace_obj["objective"] == ""
            assert whitespace_obj["outcome"] == ""


def test_domains_template_modal_improvements():
    """Test that domains.html template has the improved modal structure."""
    template_path = Path("dream/templates/domains.html")
    assert template_path.exists(), "domains.html template should exist"

    content = template_path.read_text()

    # Check that asterisks (*) are removed from field labels
    assert (
        "Domain Name*" not in content
    ), "Asterisks should be removed from field labels"
    assert "Title*" not in content, "Asterisks should be removed from field labels"

    # Check that field labels are updated
    assert (
        '<label for="domainName">Handle</label>' in content
    ), "Domain Name should be changed to Handle"
    assert (
        '<label for="domainTitle">Name</label>' in content
    ), "Title should be changed to Name"

    # Check that color input group is present
    assert 'class="color-input-group"' in content, "Color input group should be present"
    assert 'type="color"' in content, "Color picker should be present"
    assert 'id="domainColorHex"' in content, "Hex input field should be present"

    # Check for compact modal styling
    assert "max-width: 420px" in content, "Modal should be more compact"
    assert "margin-bottom: 1.2em" in content, "Form groups should have reduced spacing"

    # Check for color synchronization JavaScript
    assert "updateColorHex" in content, "Color sync functions should be present"
    assert "updateColorPicker" in content, "Color sync functions should be present"
    assert (
        "colorPicker.addEventListener" in content
    ), "Color event listeners should be present"


def test_create_objective_api():
    """Test the create_objective API endpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "matter_1"
        matter_path.mkdir(parents=True)

        # Create domain.json
        domain_data = {"title": "Test Domain", "description": "Test domain"}
        (domain_path / "domain.json").write_text(json.dumps(domain_data))

        # Create matter.json
        matter_data = {"title": "Test Matter", "description": "Test matter"}
        (matter_path / "matter.json").write_text(json.dumps(matter_data))

        # Create activity_log.jsonl
        (matter_path / "activity_log.jsonl").write_text("")

        # Test the API endpoint
        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            import json as flask_json

            from flask import Flask

            from dream.views.domains import create_objective

            app = Flask(__name__)

            with app.test_request_context(
                "/api/domains/test-domain/matters/matter_1/objectives",
                method="POST",
                data=flask_json.dumps(
                    {
                        "name": "test_implementation",
                        "objective": "# Test Implementation\n\nThis is a test objective.\n\n## Goals\n- [ ] Goal 1\n- [ ] Goal 2",
                    }
                ),
                content_type="application/json",
            ):
                result = create_objective("test-domain", "matter_1")

                # Check that the response is successful
                assert hasattr(result, "status_code")
                response_data = result.get_json()
                assert response_data["success"] is True
                assert response_data["objective_name"] == "test_implementation"
                assert response_data["objective_dir"] == "objective_test_implementation"

                # Verify the objective directory was created
                objective_dir = matter_path / "objective_test_implementation"
                assert objective_dir.exists()
                assert objective_dir.is_dir()

                # Verify OBJECTIVE.md was created with correct content
                objective_file = objective_dir / "OBJECTIVE.md"
                assert objective_file.exists()
                content = objective_file.read_text()
                assert "# Test Implementation" in content
                assert "## Goals" in content
                assert "Goal 1" in content

                # Verify activity log was updated
                activity_log = matter_path / "activity_log.jsonl"
                log_content = activity_log.read_text().strip()
                assert log_content != ""
                log_entries = [
                    flask_json.loads(line)
                    for line in log_content.split("\n")
                    if line.strip()
                ]
                assert len(log_entries) == 1
                assert log_entries[0]["type"] == "created"
                assert "test_implementation" in log_entries[0]["description"]


def test_create_objective_validation():
    """Test create_objective API validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "matter_1"
        matter_path.mkdir(parents=True)

        # Create required files
        (domain_path / "domain.json").write_text('{"title": "Test"}')
        (matter_path / "matter.json").write_text('{"title": "Test"}')
        (matter_path / "activity_log.jsonl").write_text("")

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            import json as flask_json

            from flask import Flask

            from dream.views.domains import create_objective

            app = Flask(__name__)

            # Test missing name
            with app.test_request_context(
                "/api/domains/test-domain/matters/matter_1/objectives",
                method="POST",
                data=flask_json.dumps({"objective": "Test content"}),
                content_type="application/json",
            ):
                result = create_objective("test-domain", "matter_1")
                if isinstance(result, tuple):
                    response_data, status_code = result
                    response_data = response_data.get_json()
                    assert status_code == 400
                else:
                    response_data = result.get_json()
                    assert result.status_code == 400
                assert "Objective name is required" in response_data["error"]

            # Test missing objective content
            with app.test_request_context(
                "/api/domains/test-domain/matters/matter_1/objectives",
                method="POST",
                data=flask_json.dumps({"name": "test_obj"}),
                content_type="application/json",
            ):
                result = create_objective("test-domain", "matter_1")
                if isinstance(result, tuple):
                    response_data, status_code = result
                    response_data = response_data.get_json()
                    assert status_code == 400
                else:
                    response_data = result.get_json()
                    assert result.status_code == 400
                assert "Objective content is required" in response_data["error"]

            # Test invalid name
            with app.test_request_context(
                "/api/domains/test-domain/matters/matter_1/objectives",
                method="POST",
                data=flask_json.dumps(
                    {"name": "invalid name with spaces!", "objective": "Test content"}
                ),
                content_type="application/json",
            ):
                result = create_objective("test-domain", "matter_1")
                if isinstance(result, tuple):
                    response_data, status_code = result
                    response_data = response_data.get_json()
                    assert status_code == 400
                else:
                    response_data = result.get_json()
                    assert result.status_code == 400
                assert "alphanumeric" in response_data["error"]


def test_create_objective_duplicate_handling():
    """Test create_objective handles duplicate objectives."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "journal"
        domain_path = journal_path / "domains" / "test-domain"
        matter_path = domain_path / "matter_1"
        matter_path.mkdir(parents=True)

        # Create required files
        (domain_path / "domain.json").write_text('{"title": "Test"}')
        (matter_path / "matter.json").write_text('{"title": "Test"}')
        (matter_path / "activity_log.jsonl").write_text("")

        # Create existing objective
        existing_obj_dir = matter_path / "objective_existing"
        existing_obj_dir.mkdir()
        (existing_obj_dir / "OBJECTIVE.md").write_text("Existing objective")

        with patch.dict(os.environ, {"JOURNAL_PATH": str(journal_path)}):
            import json as flask_json

            from flask import Flask

            from dream.views.domains import create_objective

            app = Flask(__name__)

            # Try to create duplicate objective
            with app.test_request_context(
                "/api/domains/test-domain/matters/matter_1/objectives",
                method="POST",
                data=flask_json.dumps(
                    {"name": "existing", "objective": "Duplicate content"}
                ),
                content_type="application/json",
            ):
                result = create_objective("test-domain", "matter_1")
                if isinstance(result, tuple):
                    response_data, status_code = result
                    response_data = response_data.get_json()
                    assert status_code == 409
                else:
                    response_data = result.get_json()
                    assert result.status_code == 409
                assert "already exists" in response_data["error"]


def test_matter_detail_template_has_add_objective():
    """Test that matter_detail.html template includes add objective functionality."""
    template_path = Path("dream/templates/matter_detail.html")
    assert template_path.exists(), "matter_detail.html template should exist"

    content = template_path.read_text()

    # Check for add objective button
    assert "Add Objective" in content
    assert "showAddObjectiveModal" in content

    # Check for modal elements
    assert "addObjectiveModal" in content
    assert "objectiveName" in content
    assert "objectiveContent" in content

    # Check for form handling
    assert "addObjectiveForm" in content
    assert "create_objective" not in content  # Should use API endpoint

    # Check for JavaScript API call
    assert "/objectives" in content
    assert "fetch(" in content
