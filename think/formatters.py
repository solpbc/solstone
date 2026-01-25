# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Formatters framework for JSONL and Markdown files.

This module provides a registry-based system for converting structured files
to markdown chunks. Each formatter is a plain function that lives near its
source domain code.

Supported file types:
    - JSONL (.jsonl): Parsed as JSON lines, passed as list[dict] to formatter
    - Markdown (.md): Read as text, passed as str to formatter

Output contract: All formatters return tuple[list[dict], dict] where:
    - list[dict]: Chunks, each with:
        - markdown: str (formatted markdown for this chunk)
        - timestamp: int (optional - unix timestamp in milliseconds for ordering)
        - source: dict (optional - original entry from JSONL for enriched streams)
    - dict: Metadata about the formatting with optional keys:
        - header: str - Optional header markdown (metadata summary, context, etc.)
        - error: str - Optional error/warning message (e.g., skipped entries)
        - indexer: dict - Indexing metadata with keys:
            - topic: str - Content type (e.g., "audio", "screen", "event")
            JSONL formatters must provide topic. Markdown topic is path-derived.
            Day and facet are extracted from path by extract_path_metadata().

JSONL formatters receive list[dict] entries and are responsible for:
    - Extracting metadata from entries (typically first line)
    - Building header from metadata if applicable
    - Formatting content entries into chunks
    - Providing indexer.topic in the meta dict

Markdown formatters receive str text and perform semantic chunking.
"""

import argparse
import fnmatch
import json
import os
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from think.utils import day_dirs, get_journal, segment_key

# Date pattern for path parsing
_DATE_RE = re.compile(r"^\d{8}$")


def extract_path_metadata(rel_path: str) -> dict[str, str]:
    """Extract indexing metadata from a journal-relative path.

    Extracts day and facet from path structure. For markdown files, also
    derives topic from path. For JSONL files, topic should be provided
    by the formatter via meta["indexer"]["topic"].

    Args:
        rel_path: Journal-relative path (e.g., "20240101/insights/flow.md")

    Returns:
        Dict with keys: day, facet, topic
        - day: YYYYMMDD string or empty
        - facet: Facet name or empty
        - topic: Derived topic for .md files, empty for .jsonl
    """
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1]
    basename = os.path.splitext(filename)[0]
    is_markdown = filename.endswith(".md")

    day = ""
    facet = ""
    topic = ""

    # Extract day from YYYYMMDD directory prefix
    if parts[0] and _DATE_RE.match(parts[0]):
        day = parts[0]

    # Extract facet from facets/{facet}/... paths
    if parts[0] == "facets" and len(parts) >= 3:
        facet = parts[1]
        # Day from YYYYMMDD filename (events/entities/todos/news)
        if len(parts) >= 4 and _DATE_RE.match(basename):
            day = basename

    # Extract day from imports/YYYYMMDD_HHMMSS/...
    if parts[0] == "imports" and len(parts) >= 2:
        import_id = parts[1]
        day = import_id.split("_")[0] if "_" in import_id else import_id[:8]

    # Extract day from config/actions/YYYYMMDD.jsonl (journal-level logs)
    if parts[0] == "config" and len(parts) >= 3 and parts[1] == "actions":
        if _DATE_RE.match(basename):
            day = basename

    # Derive topic for markdown files only
    if is_markdown:
        if parts[0] == "facets" and len(parts) >= 4 and parts[2] == "news":
            topic = "news"
        elif parts[0] == "imports":
            topic = "import"
        elif parts[0] == "apps" and len(parts) >= 4:
            topic = f"{parts[1]}:{basename}"
        else:
            # Daily insights, segment markdown: use basename
            topic = basename

    return {"day": day, "facet": facet, "topic": topic}


# Registry mapping glob patterns to (module_path, function_name)
# Patterns are matched against journal-relative paths
# Order matters: first match wins, so place specific patterns before general ones
# Note: agents/*_active.jsonl is excluded - only completed agents are formatted
FORMATTERS: dict[str, tuple[str, str]] = {
    # JSONL formatters
    "agents/*.jsonl": ("think.cortex", "format_agent"),
    "config/actions/*.jsonl": ("think.facets", "format_logs"),
    "facets/*/entities/*.jsonl": ("think.entities.formatting", "format_entities"),
    "facets/*/events/*.jsonl": ("think.events", "format_events"),
    "facets/*/todos/*.jsonl": ("apps.todos.todo", "format_todos"),
    "facets/*/logs/*.jsonl": ("think.facets", "format_logs"),
    "*/*_screen.jsonl": ("observe.screen", "format_screen"),
    "*/screen.jsonl": ("observe.screen", "format_screen"),
    "*/*_audio.jsonl": ("observe.hear", "format_audio"),
    "*/audio.jsonl": ("observe.hear", "format_audio"),
    # Markdown formatter (semantic chunking)
    "**/*.md": ("think.insights", "format_insight"),
}


def get_formatter(file_path: str) -> Callable | None:
    """Return formatter function for a file path.

    Matches against registered glob patterns. For absolute paths not under
    JOURNAL_PATH, tries progressively shorter path suffixes to find a match.

    Args:
        file_path: Path to match (journal-relative or absolute)

    Returns:
        Formatter function or None if no pattern matches
    """
    # Try the path as-is first
    for pattern, (module_path, func_name) in FORMATTERS.items():
        if fnmatch.fnmatch(file_path, pattern):
            module = import_module(module_path)
            return getattr(module, func_name)

    # For absolute paths, try progressively shorter suffixes
    # e.g., /home/user/data/agents/123.jsonl -> agents/123.jsonl
    if os.path.isabs(file_path):
        parts = Path(file_path).parts
        for i in range(1, len(parts)):
            suffix = str(Path(*parts[i:]))
            for pattern, (module_path, func_name) in FORMATTERS.items():
                if fnmatch.fnmatch(suffix, pattern):
                    module = import_module(module_path)
                    return getattr(module, func_name)

    return None


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Load entries from a JSONL file.

    Args:
        file_path: Absolute path to JSONL file

    Returns:
        List of parsed JSON objects (one per line)
    """
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_markdown(file_path: str | Path) -> str:
    """Load text from a markdown file.

    Args:
        file_path: Absolute path to markdown file

    Returns:
        File contents as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def find_formattable_files(journal: str) -> dict[str, str]:
    """Find all indexable files in the journal.

    Scans the journal directory for files that have formatters and should
    be included in the journal index. Excludes agents/*.jsonl.

    Locations scanned:
    - Daily insights: YYYYMMDD/insights/*.md
    - Segment content: YYYYMMDD/HHMMSS*/*.md, *.jsonl
    - Facet content: facets/*/events/*.jsonl, entities/, todos/, news/, logs/
    - Import summaries: imports/*/summary.md
    - App insights: apps/*/insights/*.md

    Args:
        journal: Path to journal root directory

    Returns:
        Mapping of journal-relative paths to absolute paths
    """
    files: dict[str, str] = {}
    journal_path = Path(journal)

    # Scan day directories for insights and segment content
    for day, day_abs in day_dirs().items():
        day_path = Path(day_abs)

        # Daily insights: YYYYMMDD/insights/*.md
        insights_dir = day_path / "insights"
        if insights_dir.is_dir():
            for md_file in insights_dir.glob("*.md"):
                rel = f"{day}/insights/{md_file.name}"
                files[rel] = str(md_file)

        # Segment content: YYYYMMDD/HHMMSS_LEN/*
        for entry in day_path.iterdir():
            if not entry.is_dir():
                continue
            seg_key = segment_key(entry.name)
            if not seg_key:
                continue

            # Segment insight markdown files (*.md)
            for md_file in entry.glob("*.md"):
                rel = f"{day}/{entry.name}/{md_file.name}"
                files[rel] = str(md_file)

            # Segment JSONL: audio.jsonl, screen.jsonl, *_audio.jsonl, *_screen.jsonl
            for jsonl_file in entry.glob("*.jsonl"):
                name = jsonl_file.name
                if (
                    name == "audio.jsonl"
                    or name == "screen.jsonl"
                    or name.endswith("_audio.jsonl")
                    or name.endswith("_screen.jsonl")
                ):
                    rel = f"{day}/{entry.name}/{name}"
                    files[rel] = str(jsonl_file)

    # Facet content: facets/*/...
    facets_dir = journal_path / "facets"
    if facets_dir.is_dir():
        for facet_dir in facets_dir.iterdir():
            if not facet_dir.is_dir():
                continue
            facet_name = facet_dir.name

            # Events: facets/*/events/*.jsonl
            events_dir = facet_dir / "events"
            if events_dir.is_dir():
                for jsonl_file in events_dir.glob("*.jsonl"):
                    rel = f"facets/{facet_name}/events/{jsonl_file.name}"
                    files[rel] = str(jsonl_file)

            # Entities detected: facets/*/entities/*.jsonl
            entities_dir = facet_dir / "entities"
            if entities_dir.is_dir():
                for jsonl_file in entities_dir.glob("*.jsonl"):
                    rel = f"facets/{facet_name}/entities/{jsonl_file.name}"
                    files[rel] = str(jsonl_file)

            # Todos: facets/*/todos/*.jsonl
            todos_dir = facet_dir / "todos"
            if todos_dir.is_dir():
                for jsonl_file in todos_dir.glob("*.jsonl"):
                    rel = f"facets/{facet_name}/todos/{jsonl_file.name}"
                    files[rel] = str(jsonl_file)

            # News: facets/*/news/*.md
            news_dir = facet_dir / "news"
            if news_dir.is_dir():
                for md_file in news_dir.glob("*.md"):
                    rel = f"facets/{facet_name}/news/{md_file.name}"
                    files[rel] = str(md_file)

            # Action logs: facets/*/logs/*.jsonl
            logs_dir = facet_dir / "logs"
            if logs_dir.is_dir():
                for jsonl_file in logs_dir.glob("*.jsonl"):
                    rel = f"facets/{facet_name}/logs/{jsonl_file.name}"
                    files[rel] = str(jsonl_file)

    # Import summaries: imports/*/summary.md
    imports_dir = journal_path / "imports"
    if imports_dir.is_dir():
        for import_dir in imports_dir.iterdir():
            if not import_dir.is_dir():
                continue
            summary_file = import_dir / "summary.md"
            if summary_file.is_file():
                rel = f"imports/{import_dir.name}/summary.md"
                files[rel] = str(summary_file)

    # App insights: apps/*/insights/*.md
    apps_dir = journal_path / "apps"
    if apps_dir.is_dir():
        for app_dir in apps_dir.iterdir():
            if not app_dir.is_dir():
                continue
            app_insights_dir = app_dir / "insights"
            if app_insights_dir.is_dir():
                for md_file in app_insights_dir.glob("*.md"):
                    rel = f"apps/{app_dir.name}/insights/{md_file.name}"
                    files[rel] = str(md_file)

    return files


def format_file(
    file_path: str | Path,
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load file, detect formatter, return formatted chunks and metadata.

    Supports both JSONL and Markdown files. File type is detected by extension.

    Args:
        file_path: Absolute path to file (.jsonl or .md)
        context: Optional context dict passed to formatter

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with "markdown" key (and optional "timestamp")
            - meta: Dict with optional "header" and "error" keys

    Raises:
        ValueError: If no formatter found for file path
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get journal-relative path for pattern matching
    journal_path = Path(get_journal()).resolve()

    if file_path.is_relative_to(journal_path):
        rel_path = str(file_path.relative_to(journal_path))
    else:
        # Fall back to just the filename parts for matching
        rel_path = str(file_path)

    formatter = get_formatter(rel_path)
    if formatter is None:
        raise ValueError(f"No formatter found for: {rel_path}")

    # Load file based on extension
    if file_path.suffix == ".md":
        content = load_markdown(file_path)
    else:
        content = load_jsonl(file_path)

    # Build context with file path info
    ctx = context or {}
    ctx.setdefault("file_path", file_path)

    return formatter(content, ctx)


def _format_chunk_summary(chunks: list[dict], raw_chunks: list[dict] | None) -> None:
    """Print human-readable chunk summary (for markdown files with raw chunks)."""
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        # Use raw chunk data if available, otherwise extract from markdown
        if raw_chunks and i < len(raw_chunks):
            c = raw_chunks[i]
            chunk_type = c.get("type", "unknown")
            header_path = c.get("header_path", [])
            intro = c.get("intro")
            preview = c.get("preview", "")
        else:
            chunk_type = "chunk"
            header_path = []
            intro = None
            preview = chunk.get("markdown", "")[:70]

        path = " > ".join(f"H{h['level']}:{h['text']}" for h in header_path)
        print(f"#{i:3d} [{chunk_type:13s}]")
        if path:
            print(f"      path: {path}")
        if intro:
            print(f"      intro: \"{intro[:60]}{'...' if len(intro) > 60 else ''}\"")
        print(f"      {preview[:70]}{'...' if len(preview) > 70 else ''}")
        print()


def main() -> None:
    """CLI entry point for sol formatter."""
    from think.utils import setup_cli

    parser = argparse.ArgumentParser(
        description="Convert JSONL or Markdown files to formatted chunks"
    )
    parser.add_argument("file", help="Path to JSONL or Markdown file")
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "markdown", "summary"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Show only the chunk at this index",
    )
    parser.add_argument(
        "--join",
        action="store_true",
        help="Output concatenated markdown (shorthand for --format=markdown)",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="JSON string of context to pass to formatter",
    )
    args = setup_cli(parser)

    # --join is shorthand for --format=markdown
    if args.join:
        args.format = "markdown"

    try:
        context = json.loads(args.context) if args.context else None
    except json.JSONDecodeError as e:
        print(f"Error parsing context JSON: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        chunks, meta = format_file(args.file, context)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # For summary format on markdown files, get raw chunks with metadata
    raw_chunks = None
    if args.format == "summary" and args.file.endswith(".md"):
        from think.insights import chunk_markdown

        text = load_markdown(args.file)
        raw_chunks = chunk_markdown(text)

    # Filter to single chunk if requested
    if args.index is not None:
        if 0 <= args.index < len(chunks):
            chunks = [chunks[args.index]]
            if raw_chunks:
                raw_chunks = [raw_chunks[args.index]]
        else:
            print(
                f"Error: Index {args.index} out of range (0-{len(chunks) - 1})",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.format == "markdown":
        # Output concatenated markdown with header first
        parts = []
        if meta.get("header"):
            parts.append(meta["header"])
        parts.extend(chunk["markdown"] for chunk in chunks)
        print("\n".join(parts))
    elif args.format == "summary":
        _format_chunk_summary(chunks, raw_chunks)
    else:
        # Output JSON object with metadata and chunks
        print(json.dumps({"meta": meta, "chunks": chunks}, indent=2))


if __name__ == "__main__":
    main()
