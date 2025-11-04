"""Tests for news indexer functionality."""

from pathlib import Path
from unittest.mock import patch

from think.indexer import find_news_files, scan_news, search_news
from think.indexer.core import get_index


def _write_news_file(
    path: Path, title: str, *, source: str, time: str, body: str
) -> None:
    """Helper to create a news markdown file."""
    content = (
        f"# {title}\n\n"
        f"## {title} Headline\n"
        f"**Source:** {source} | **Time:** {time}\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def test_find_news_files(tmp_path):
    """Test that find_news_files discovers all news markdown files."""
    journal = tmp_path / "journal"

    # Create news files for two facets
    facet1 = journal / "facets" / "tech-news"
    facet1_news = facet1 / "news"
    facet1_news.mkdir(parents=True)

    facet2 = journal / "facets" / "world-events"
    facet2_news = facet2 / "news"
    facet2_news.mkdir(parents=True)

    # Create news files
    (facet1_news / "20240101.md").write_text("Tech news 1")
    (facet1_news / "20240102.md").write_text("Tech news 2")
    (facet2_news / "20240103.md").write_text("World news 1")

    # Create non-news files that should be ignored
    (facet1_news / "summary.txt").write_text("Not a news file")
    (facet1_news / "draft.md").write_text("Not in YYYYMMDD format")

    files = find_news_files(str(journal))

    assert len(files) == 3
    assert "facets/tech-news/news/20240101.md" in files
    assert "facets/tech-news/news/20240102.md" in files
    assert "facets/world-events/news/20240103.md" in files
    assert "facets/tech-news/news/summary.txt" not in files
    assert "facets/tech-news/news/draft.md" not in files


def test_scan_news_indexes_content(tmp_path):
    """Test that scan_news properly indexes news content."""
    journal = tmp_path / "journal"
    facet = journal / "facets" / "test-facet"
    news_dir = facet / "news"
    news_dir.mkdir(parents=True)

    # Create news files
    _write_news_file(
        news_dir / "20240101.md",
        "Breaking News",
        source="news.com",
        time="10:00",
        body="This is an important development in technology.",
    )

    _write_news_file(
        news_dir / "20240102.md",
        "Latest Update",
        source="update.org",
        time="14:30",
        body="Another significant event has occurred today.",
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal)}):
        # Scan the news files
        changed = scan_news(str(journal))
        assert changed

        # Verify index was created and contains data
        conn, _ = get_index(index="news", journal=str(journal))

        # Check file tracking
        cursor = conn.execute("SELECT COUNT(*) FROM files")
        assert cursor.fetchone()[0] == 2

        # Check indexed content
        cursor = conn.execute("SELECT COUNT(*) FROM news_text")
        assert cursor.fetchone()[0] == 2

        # Verify content and metadata
        cursor = conn.execute("SELECT content, facet, day FROM news_text ORDER BY day")
        rows = cursor.fetchall()

        assert "Breaking News" in rows[0][0]
        assert rows[0][1] == "test-facet"
        assert rows[0][2] == "20240101"

        assert "Latest Update" in rows[1][0]
        assert rows[1][1] == "test-facet"
        assert rows[1][2] == "20240102"

        conn.close()


def test_scan_news_incremental(tmp_path):
    """Test that scan_news only re-indexes modified files."""
    import time

    journal = tmp_path / "journal"
    facet = journal / "facets" / "test-facet"
    news_dir = facet / "news"
    news_dir.mkdir(parents=True)

    news_file = news_dir / "20240101.md"
    _write_news_file(
        news_file,
        "Initial News",
        source="source.com",
        time="09:00",
        body="Initial content.",
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal)}):
        # First scan
        changed = scan_news(str(journal))
        assert changed

        # Second scan without changes
        changed = scan_news(str(journal))
        assert not changed

        # Wait to ensure mtime changes (filesystem timestamp resolution)
        time.sleep(1)

        # Modify the file
        _write_news_file(
            news_file,
            "Updated News",
            source="source.com",
            time="09:00",
            body="Updated content with new information.",
        )

        # Third scan should detect change
        changed = scan_news(str(journal))
        assert changed

        # Verify updated content
        conn, _ = get_index(index="news", journal=str(journal))
        cursor = conn.execute("SELECT content FROM news_text")
        content = cursor.fetchone()[0]
        assert "Updated News" in content
        assert "Updated content" in content
        conn.close()


def test_search_news_basic(tmp_path):
    """Test basic news search functionality."""
    journal = tmp_path / "journal"
    facet = journal / "facets" / "tech-facet"
    news_dir = facet / "news"
    news_dir.mkdir(parents=True)

    _write_news_file(
        news_dir / "20240101.md",
        "AI Development",
        source="tech.com",
        time="10:00",
        body="Revolutionary artificial intelligence breakthrough announced.",
    )

    _write_news_file(
        news_dir / "20240102.md",
        "Climate Report",
        source="science.org",
        time="11:00",
        body="New climate change data reveals concerning trends.",
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal)}):
        scan_news(str(journal))

        # Search for AI content
        total, results = search_news("artificial intelligence", limit=10)
        assert total == 1
        assert len(results) == 1
        assert results[0]["metadata"]["facet"] == "tech-facet"
        assert results[0]["metadata"]["day"] == "20240101"

        # Search for climate content
        total, results = search_news("climate", limit=10)
        assert total == 1
        assert len(results) == 1
        assert results[0]["metadata"]["day"] == "20240102"


def test_search_news_with_filters(tmp_path):
    """Test news search with facet and day filters."""
    journal = tmp_path / "journal"

    # Create news for multiple facets
    facet1 = journal / "facets" / "tech"
    facet1_news = facet1 / "news"
    facet1_news.mkdir(parents=True)

    facet2 = journal / "facets" / "finance"
    facet2_news = facet2 / "news"
    facet2_news.mkdir(parents=True)

    # Tech facet news
    _write_news_file(
        facet1_news / "20240101.md",
        "Tech News 1",
        source="tech.com",
        time="10:00",
        body="Technology market analysis and trends.",
    )

    _write_news_file(
        facet1_news / "20240102.md",
        "Tech News 2",
        source="tech.com",
        time="10:00",
        body="Latest technology market updates.",
    )

    # Finance facet news
    _write_news_file(
        facet2_news / "20240101.md",
        "Finance News",
        source="finance.com",
        time="11:00",
        body="Stock market analysis for today.",
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal)}):
        scan_news(str(journal))

        # Search all facets for "market"
        total, results = search_news("market", limit=10)
        assert total == 3

        # Filter by tech facet
        total, results = search_news("market", limit=10, facet="tech")
        assert total == 2
        assert all(r["metadata"]["facet"] == "tech" for r in results)

        # Filter by finance facet
        total, results = search_news("market", limit=10, facet="finance")
        assert total == 1
        assert results[0]["metadata"]["facet"] == "finance"

        # Filter by specific day
        total, results = search_news("market", limit=10, day="20240101")
        assert total == 2
        assert all(r["metadata"]["day"] == "20240101" for r in results)

        # Filter by facet and day
        total, results = search_news("market", limit=10, facet="tech", day="20240102")
        assert total == 1
        assert results[0]["metadata"]["facet"] == "tech"
        assert results[0]["metadata"]["day"] == "20240102"


def test_search_news_pagination(tmp_path):
    """Test news search pagination."""
    journal = tmp_path / "journal"
    facet = journal / "facets" / "news-facet"
    news_dir = facet / "news"
    news_dir.mkdir(parents=True)

    # Create multiple news files with similar content
    for i in range(1, 6):
        _write_news_file(
            news_dir / f"2024010{i}.md",
            f"News Item {i}",
            source="source.com",
            time="10:00",
            body=f"Important development number {i} in the ongoing story.",
        )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal)}):
        scan_news(str(journal))

        # Search with limit
        total, page1 = search_news("development", limit=2, offset=0)
        assert total == 5
        assert len(page1) == 2

        # Get next page
        total, page2 = search_news("development", limit=2, offset=2)
        assert total == 5
        assert len(page2) == 2

        # Get last page
        total, page3 = search_news("development", limit=2, offset=4)
        assert total == 5
        assert len(page3) == 1

        # Verify no duplicate results
        all_ids = [r["id"] for r in page1 + page2 + page3]
        assert len(all_ids) == len(set(all_ids))


def test_news_file_removal(tmp_path):
    """Test that deleted news files are removed from index."""
    journal = tmp_path / "journal"
    facet = journal / "facets" / "test-facet"
    news_dir = facet / "news"
    news_dir.mkdir(parents=True)

    news_file = news_dir / "20240101.md"
    _write_news_file(
        news_file,
        "Temporary News",
        source="temp.com",
        time="10:00",
        body="This news will be deleted.",
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal)}):
        # Initial scan
        scan_news(str(journal))

        # Verify file is indexed
        total, results = search_news("deleted", limit=10)
        assert total == 1

        # Delete the file
        news_file.unlink()

        # Rescan
        changed = scan_news(str(journal))
        assert changed

        # Verify file is removed from index
        total, results = search_news("deleted", limit=10)
        assert total == 0
