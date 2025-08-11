from unittest.mock import patch

import pytest
from flask import Flask

from dream.views.entities import bp


@pytest.fixture
def app():
    """Create a test Flask app with mocked dependencies."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-key"

    # Register the entities blueprint
    app.register_blueprint(bp)

    yield app


@pytest.fixture
def authenticated_client(app):
    """Create a test client with authentication bypassed."""
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["logged_in"] = True
        yield client


@patch("dream.views.entities.render_template")
def test_entities_page(mock_render, authenticated_client):
    """Test that the entities page loads."""
    mock_render.return_value = "Mocked entities page with Entity Review"

    response = authenticated_client.get("/entities")
    assert response.status_code == 200
    assert b"Entity Review" in response.data

    # Verify the template was called correctly
    mock_render.assert_called_once_with("entities.html", active="entities")


@patch("dream.views.entities.search_entities")
def test_entities_types_api(mock_search, authenticated_client):
    """Test the entity types API endpoint."""
    # Mock search_entities to return counts for each type
    mock_search.side_effect = [
        (5, []),  # Person count
        (3, []),  # Company count
        (2, []),  # Project count
        (1, []),  # Tool count
    ]

    response = authenticated_client.get("/entities/api/types")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert data["Person"] == 5
    assert data["Company"] == 3
    assert data["Project"] == 2
    assert data["Tool"] == 1


@patch("dream.views.entities.search_entities")
def test_entities_list_api(mock_search, authenticated_client):
    """Test the entity list API endpoint."""
    mock_search.side_effect = [
        (
            2,
            [
                {
                    "metadata": {"name": "Alice", "top": True, "days": 5},
                    "text": "Software engineer",
                },
                {
                    "metadata": {"name": "Bob", "top": False, "days": 3},
                    "text": "Product manager",
                },
            ],
        ),
        (0, []),
    ]

    response = authenticated_client.get("/entities/api/list?type=Person")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["name"] == "Alice"
    assert data[0]["top"] is True
    assert data[1]["name"] == "Bob"


@patch("dream.views.entities.reload_entities")
@patch("dream.views.entities.update_top_entry")
@patch("dream.views.entities.search_entities")
@patch("dream.views.entities.state")
def test_create_entity_valid(
    mock_state, mock_search, mock_update, mock_reload, authenticated_client
):
    """Test creating a valid new entity."""
    mock_state.journal_root = "/fake/path"
    mock_search.return_value = (0, [])  # No existing entity

    response = authenticated_client.post(
        "/entities/api/create",
        json={"type": "Person", "name": "Bob Smith", "description": "Product manager"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True

    # Verify functions were called
    mock_update.assert_called_once_with(
        "/fake/path", "Person", "Bob Smith", "Product manager"
    )
    mock_reload.assert_called_once()


@patch("dream.views.entities.reload_entities")
@patch("dream.views.entities.update_top_entry")
@patch("dream.views.entities.search_entities")
@patch("dream.views.entities.state")
def test_create_entity_without_description(
    mock_state, mock_search, mock_update, mock_reload, authenticated_client
):
    """Test creating an entity without description."""
    mock_state.journal_root = "/fake/path"
    mock_search.return_value = (0, [])  # No existing entity

    response = authenticated_client.post(
        "/entities/api/create",
        json={"type": "Tool", "name": "New Tool"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True

    # Should use name as default description
    mock_update.assert_called_once_with("/fake/path", "Tool", "New Tool", "New Tool")


def test_create_entity_missing_name(authenticated_client):
    """Test creating an entity without name fails."""
    response = authenticated_client.post(
        "/entities/api/create",
        json={"type": "Person", "description": "Some description"},
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "name are required" in data["error"]


def test_create_entity_missing_type(authenticated_client):
    """Test creating an entity without type fails."""
    response = authenticated_client.post(
        "/entities/api/create",
        json={"name": "Test Name", "description": "Some description"},
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "type and name are required" in data["error"]


def test_create_entity_invalid_type(authenticated_client):
    """Test creating an entity with invalid type fails."""
    response = authenticated_client.post(
        "/entities/api/create",
        json={
            "type": "InvalidType",
            "name": "Test Name",
            "description": "Some description",
        },
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Invalid entity type" in data["error"]


@patch("dream.views.entities.search_entities")
@patch("dream.views.entities.state")
def test_create_entity_duplicate(mock_state, mock_search, authenticated_client):
    """Test creating a duplicate entity fails."""
    mock_state.journal_root = "/fake/path"
    mock_search.return_value = (
        1,
        [{"metadata": {"top": True}}],  # Existing top-level entity
    )

    response = authenticated_client.post(
        "/entities/api/create",
        json={
            "type": "Person",
            "name": "Existing Person",
            "description": "Duplicate person",
        },
        content_type="application/json",
    )
    assert response.status_code == 409
    data = response.get_json()
    assert data["success"] is False
    assert "already exists" in data["error"]
