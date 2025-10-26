from unittest.mock import MagicMock, patch

import muse.mcp_tools as mcp_tools


def test_todo_list_success_returns_numbered_markdown():
    mock_checklist = MagicMock()
    mock_checklist.numbered.return_value = "1: - [ ] Investigate"

    with patch.object(
        mcp_tools.todo.TodoChecklist,
        "load",
        return_value=mock_checklist,
    ) as load_mock:
        result = mcp_tools.todo_list("20240101", "test")

    load_mock.assert_called_once_with("20240101", "test")
    assert result == {
        "day": "20240101",
        "domain": "test",
        "markdown": "1: - [ ] Investigate",
    }
    mock_checklist.numbered.assert_called_once_with()


def test_search_news_success():
    """Test successful news search with results."""
    mock_results = [
        {
            "text": "AI breakthrough announced",
            "metadata": {
                "domain": "tech",
                "day": "20250118",
                "path": "domains/tech/news/20250118.md",
            },
        },
        {
            "text": "Quarterly earnings report",
            "metadata": {
                "domain": "finance",
                "day": "20250117",
                "path": "domains/finance/news/20250117.md",
            },
        },
    ]

    with patch("muse.mcp_tools.search_news_impl") as mock_search:
        mock_search.return_value = (2, mock_results)
        result = mcp_tools.search_news("breakthrough", limit=5, offset=0)

    mock_search.assert_called_once_with("breakthrough", 5, 0)
    assert result["total"] == 2
    assert result["limit"] == 5
    assert result["offset"] == 0
    assert len(result["results"]) == 2
    assert result["results"][0] == {
        "domain": "tech",
        "day": "20250118",
        "text": "AI breakthrough announced",
        "path": "domains/tech/news/20250118.md",
    }


def test_search_news_with_filters():
    """Test news search with domain and day filters."""
    mock_results = [
        {
            "text": "Product launch news",
            "metadata": {
                "domain": "work",
                "day": "20250118",
                "path": "domains/work/news/20250118.md",
            },
        },
    ]

    with patch("muse.mcp_tools.search_news_impl") as mock_search:
        mock_search.return_value = (1, mock_results)
        result = mcp_tools.search_news(
            "product", limit=10, offset=0, domain="work", day="20250118"
        )

    mock_search.assert_called_once_with("product", 10, 0, domain="work", day="20250118")
    assert result["total"] == 1
    assert result["results"][0]["domain"] == "work"
    assert result["results"][0]["day"] == "20250118"


def test_search_news_empty_results():
    """Test news search with no results."""
    with patch("muse.mcp_tools.search_news_impl") as mock_search:
        mock_search.return_value = (0, [])
        result = mcp_tools.search_news("nonexistent", limit=5, offset=0)

    assert result["total"] == 0
    assert result["results"] == []
    assert result["limit"] == 5
    assert result["offset"] == 0


def test_search_news_error_handling():
    """Test news search error handling."""
    with patch("muse.mcp_tools.search_news_impl") as mock_search:
        mock_search.side_effect = RuntimeError("Database error")
        result = mcp_tools.search_news("test", limit=5, offset=0)

    assert "error" in result
    assert "Failed to search news" in result["error"]
    assert "suggestion" in result


def test_get_news_content_exists(tmp_path):
    """Test get_news_content resource when news file exists."""
    # Setup test environment
    journal_path = tmp_path / "journal"
    domain_path = journal_path / "domains" / "tech"
    news_dir = domain_path / "news"
    news_dir.mkdir(parents=True)

    news_file = news_dir / "20250118.md"
    news_content = (
        "# 2025-01-18 Tech News\n\n## AI Breakthrough\n\nMajor advancement in AI..."
    )
    news_file.write_text(news_content)

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal_path)}):
        # Access the underlying function via the fn attribute
        resource = mcp_tools.get_news_content.fn("tech", "20250118")

    assert str(resource.uri) == "journal://news/tech/20250118"
    assert resource.name == "News: tech/20250118"
    assert resource.mime_type == "text/markdown"
    assert resource.text == news_content


def test_get_news_content_missing_file(tmp_path):
    """Test get_news_content resource when news file doesn't exist."""
    journal_path = tmp_path / "journal"
    domain_path = journal_path / "domains" / "tech"
    domain_path.mkdir(parents=True)

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal_path)}):
        # Access the underlying function via the fn attribute
        resource = mcp_tools.get_news_content.fn("tech", "20250118")

    assert str(resource.uri) == "journal://news/tech/20250118"
    assert resource.text == "No news recorded for 20250118 in domain 'tech'."


def test_get_news_content_missing_domain(tmp_path):
    """Test get_news_content resource when domain doesn't exist."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal_path)}):
        # Access the underlying function via the fn attribute
        resource = mcp_tools.get_news_content.fn("nonexistent", "20250118")

    assert str(resource.uri) == "journal://news/nonexistent/20250118"
    assert resource.text == "Domain 'nonexistent' not found."


def test_entity_add_aka_success():
    """Test successfully adding an aka to an entity."""
    mock_entities = [
        {
            "type": "Tool",
            "name": "PostgreSQL",
            "description": "Database system",
            "aka": ["Postgres"],
        },
        {"type": "Person", "name": "Alice", "description": "Engineer"},
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.save_entities") as mock_save,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    mock_validate.assert_called_once_with("Tool")
    mock_load.assert_called_once_with("work", day=None)
    mock_save.assert_called_once()

    # Verify the entity was updated
    saved_entities = mock_save.call_args[0][1]
    postgres = next(e for e in saved_entities if e["name"] == "PostgreSQL")
    assert "PG" in postgres["aka"]
    assert "Postgres" in postgres["aka"]

    # Verify response
    assert result["domain"] == "work"
    assert "Added alias 'PG'" in result["message"]
    assert result["entity"]["aka"] == ["Postgres", "PG"]


def test_entity_add_aka_duplicate():
    """Test adding an aka that already exists (dedup)."""
    mock_entities = [
        {
            "type": "Tool",
            "name": "PostgreSQL",
            "description": "Database system",
            "aka": ["Postgres", "PG"],
        },
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.save_entities") as mock_save,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    # Should not call save since aka already exists
    mock_save.assert_not_called()

    # Verify response
    assert result["domain"] == "work"
    assert "already exists" in result["message"]
    assert result["entity"]["aka"] == ["Postgres", "PG"]


def test_entity_add_aka_initialize_aka_list():
    """Test adding aka to entity that has no aka field yet."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Engineer",
        },
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.save_entities") as mock_save,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka("personal", "Person", "Alice Johnson", "Ali")

    mock_save.assert_called_once()

    # Verify the aka list was initialized
    saved_entities = mock_save.call_args[0][1]
    alice = next(e for e in saved_entities if e["name"] == "Alice Johnson")
    assert alice["aka"] == ["Ali"]

    # Verify response
    assert result["domain"] == "personal"
    assert "Added alias 'Ali'" in result["message"]
    assert result["entity"]["aka"] == ["Ali"]


def test_entity_add_aka_invalid_type():
    """Test adding aka with invalid entity type."""
    with patch("muse.mcp_tools.is_valid_entity_type") as mock_validate:
        mock_validate.return_value = False
        result = mcp_tools.entity_add_aka("work", "XY", "PostgreSQL", "PG")

    assert "error" in result
    assert "Invalid entity type 'XY'" in result["error"]
    assert "suggestion" in result


def test_entity_add_aka_entity_not_found():
    """Test adding aka to non-existent entity."""
    mock_entities = [
        {"type": "Person", "name": "Alice", "description": "Engineer"},
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    assert "error" in result
    assert "not found in attached entities" in result["error"]
    assert "suggestion" in result


def test_entity_add_aka_runtime_error():
    """Test entity_add_aka when JOURNAL_PATH not set."""
    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.side_effect = RuntimeError("JOURNAL_PATH not set")
        result = mcp_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    assert "error" in result
    assert "JOURNAL_PATH not set" in result["error"]
    assert "suggestion" in result


def test_entity_add_aka_skip_first_word():
    """Test that adding first word of entity name as aka is silently skipped."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Jeremie Miller",
            "description": "Software engineer",
        },
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.save_entities") as mock_save,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka(
            "personal", "Person", "Jeremie Miller", "Jeremie"
        )

    # Should not call save since first word is skipped
    mock_save.assert_not_called()

    # Verify response indicates skip
    assert result["domain"] == "personal"
    assert "first word" in result["message"]
    assert "skipped" in result["message"]


def test_entity_add_aka_skip_first_word_case_insensitive():
    """Test first word skip is case-insensitive."""
    mock_entities = [
        {
            "type": "Organization",
            "name": "Anthropic PBC",
            "description": "AI research company",
        },
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.save_entities") as mock_save,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka(
            "work", "Organization", "Anthropic PBC", "anthropic"
        )

    mock_save.assert_not_called()
    assert "first word" in result["message"]
    assert "skipped" in result["message"]


def test_entity_add_aka_skip_first_word_with_parens():
    """Test first word detection strips parentheses from entity name."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson (AJ)",
            "description": "Project manager",
        },
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.save_entities") as mock_save,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka(
            "personal", "Person", "Alice Johnson (AJ)", "Alice"
        )

    # Should skip since "Alice" is the first word (ignoring parens)
    mock_save.assert_not_called()
    assert "first word" in result["message"]


def test_entity_add_aka_not_first_word():
    """Test that non-first-word aliases are still added."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Jeremie Miller",
            "description": "Software engineer",
        },
    ]

    with (
        patch("muse.mcp_tools.load_entities") as mock_load,
        patch("muse.mcp_tools.save_entities") as mock_save,
        patch("muse.mcp_tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = mcp_tools.entity_add_aka("personal", "Person", "Jeremie Miller", "Jer")

    # Should save since "Jer" is not the first word "Jeremie"
    mock_save.assert_called_once()

    # Verify the aka was added
    saved_entities = mock_save.call_args[0][1]
    jeremie = next(e for e in saved_entities if e["name"] == "Jeremie Miller")
    assert "Jer" in jeremie["aka"]

    assert "Added alias 'Jer'" in result["message"]
