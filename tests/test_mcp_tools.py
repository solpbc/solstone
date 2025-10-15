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
    assert result == {"day": "20240101", "domain": "test", "markdown": "1: - [ ] Investigate"}
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
