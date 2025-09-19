"""Tests for domain news utilities."""

import json
from pathlib import Path
from unittest.mock import patch


def _write_news_file(path: Path, title: str, *, source: str, time: str, body: str) -> None:
    content = (
        f"# {title}\n\n"
        f"## {title} Headline\n"
        f"**Source:** {source} | **Time:** {time}\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def test_get_domain_news_orders_and_paginates(tmp_path):
    """get_domain_news should return newest news first and support pagination."""

    journal_path = tmp_path / "journal"
    domain_path = journal_path / "domains" / "test-domain"
    news_dir = domain_path / "news"
    news_dir.mkdir(parents=True)

    # Minimal domain metadata required by get_domain_news parent lookups
    (domain_path / "domain.json").write_text(json.dumps({"title": "Test"}), encoding="utf-8")

    latest_news = news_dir / "20240102.md"
    older_news = news_dir / "20240101.md"

    _write_news_file(
        latest_news,
        "2024-01-02 News",
        source="example.com",
        time="10:00",
        body="Latest insight summary for the domain."
    )

    _write_news_file(
        older_news,
        "2024-01-01 News",
        source="another.com",
        time="08:30",
        body="Older summary entry for the domain."
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(journal_path)}):
        from think.domains import get_domain_news

        first_page = get_domain_news("test-domain")

        assert first_page["days"], "First page should include at least one news day"
        assert first_page["days"][0]["date"] == "20240102"
        assert first_page["days"][0]["entries"], "News entries should be parsed"

        entry = first_page["days"][0]["entries"][0]
        assert entry["source"] == "example.com"
        assert entry["time"] == "10:00"
        assert "Latest insight summary" in entry["paragraphs"][0]

        # Should signal more pages are available
        assert first_page["has_more"], "Expected additional pages"
        assert first_page["next_cursor"] == "20240102"

        second_page = get_domain_news("test-domain", cursor=first_page["next_cursor"])

        assert second_page["days"], "Second page should include older news"
        assert second_page["days"][0]["date"] == "20240101"
        assert not second_page["has_more"]
