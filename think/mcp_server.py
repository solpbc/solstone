import argparse
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from think.indexer import search_occurrences as search_occurrences_impl
from think.indexer import search_topics as search_topics_impl
from think.utils import setup_cli


def search_topic(query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
    """Full-text search over topic summaries."""

    journal = os.getenv("JOURNAL_PATH", "")
    total, results = search_topics_impl(journal, query, limit, offset)
    items = []
    for r in results:
        meta = r.get("metadata", {})
        topic = meta.get("topic", "")
        if topic.endswith(".md"):
            topic = topic[:-3]
        items.append(
            {"day": meta.get("day", ""), "filename": topic, "text": r.get("text", "")}
        )
    return {"total": total, "results": items}


def search_occurrence(query: str) -> str:
    """Search structured occurrences by keyword."""

    journal = os.getenv("JOURNAL_PATH", "")
    results = search_occurrences_impl(journal, query, 5)
    lines = []
    for r in results:
        meta = r.get("metadata", {})
        lines.append(f"{meta.get('day')} {meta.get('type')}: {r['text']}")
    return "\n".join(lines)


def read_markdown(date: str, filename: str) -> str:
    """Return journal markdown contents."""

    journal = os.getenv("JOURNAL_PATH", "journal")
    md_path = Path(journal) / date / f"{filename}.md"
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown not found: {md_path}")
    return md_path.read_text(encoding="utf-8")


def create_server(journal: str) -> FastMCP:
    mcp = FastMCP(name="sunstone")
    os.environ["JOURNAL_PATH"] = journal
    mcp.tool(search_topic, title="Search topics")
    mcp.tool(search_occurrence, title="Search occurrences")
    mcp.tool(read_markdown, title="Read markdown")
    return mcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run using STDIO transport instead of HTTP",
    )
    args = setup_cli(parser)

    journal = os.getenv("JOURNAL_PATH") or parser.error("JOURNAL_PATH not set")
    server = create_server(journal)
    if args.stdio:
        server.run()
    else:
        server.run(
            "streamable-http",
            host="0.0.0.0",
            port=args.port,
        )


if __name__ == "__main__":
    main()
