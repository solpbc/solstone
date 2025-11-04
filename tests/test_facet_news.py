"""Tests for facet news utilities."""

import json
from pathlib import Path
from unittest.mock import patch


def _write_news_file(
    path: Path, title: str, *, source: str, time: str, body: str
) -> None:
    content = (
        f"# {title}\n\n"
        f"## {title} Headline\n"
        f"**Source:** {source} | **Time:** {time}\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def test_get_facet_news_orders_and_paginates(tmp_path):
    """get_facet_news should return newest news first and support pagination."""

    journal_path = tmp_path / "journal"
    facet_path = journal_path / "facets" / "test-facet"
    news_dir = facet_path / "news"
    news_dir.mkdir(parents=True)

    # Minimal facet metadata required by get_facet_news parent lookups
    (facet_path / "facet.json").write_text(
        json.dumps({"title": "Test"}), encoding="utf-8"
    )

    # Create three news files for better testing
    newest_news = news_dir / "20240103.md"
    latest_news = news_dir / "20240102.md"
    older_news = news_dir / "20240101.md"

    _write_news_file(
        newest_news,
        "2024-01-03 News",
        source="site.com",
        time="12:00",
        body="Newest insight summary for the facet.",
    )

    _write_news_file(
        latest_news,
        "2024-01-02 News",
        source="example.com",
        time="10:00",
        body="Latest insight summary for the facet.",
    )

    _write_news_file(
        older_news,
        "2024-01-01 News",
        source="another.com",
        time="08:30",
        body="Older summary entry for the facet.",
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal_path)}):
        from think.facets import get_facet_news

        first_page = get_facet_news("test-facet")

        assert first_page["days"], "First page should include at least one news day"
        assert first_page["days"][0]["date"] == "20240103"
        assert first_page["days"][0]["raw_content"], "News should have raw content"
        assert "Newest insight summary" in first_page["days"][0]["raw_content"]

        # Should signal more pages are available
        assert first_page["has_more"], "Expected additional pages"
        assert first_page["next_cursor"] == "20240103"

        second_page = get_facet_news("test-facet", cursor=first_page["next_cursor"])

        assert second_page["days"], "Second page should include older news"
        assert second_page["days"][0]["date"] == "20240102"
        assert second_page["has_more"]

        # Test specific day retrieval
        specific_day = get_facet_news("test-facet", day="20240102")
        assert len(specific_day["days"]) == 1
        assert specific_day["days"][0]["date"] == "20240102"
        assert "example.com" in specific_day["days"][0]["raw_content"]
        assert specific_day["next_cursor"] is None
        assert specific_day["has_more"] is False

        # Test non-existent day
        no_day = get_facet_news("test-facet", day="20240199")
        assert len(no_day["days"]) == 0
        assert no_day["next_cursor"] is None
        assert no_day["has_more"] is False
