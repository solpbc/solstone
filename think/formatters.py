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
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from think.utils import DATE_RE, get_journal


def extract_path_metadata(rel_path: str) -> dict[str, str]:
    """Extract indexing metadata from a journal-relative path.

    Extracts day and facet from path structure. For markdown files, also
    derives topic from path. For JSONL files, topic should be provided
    by the formatter via meta["indexer"]["topic"].

    Args:
        rel_path: Journal-relative path (e.g., "20240101/agents/flow.md")

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
    if parts[0] and DATE_RE.fullmatch(parts[0]):
        day = parts[0]

    # Extract facet from agents/{facet}/... paths
    try:
        agents_idx = parts.index("agents")
        if agents_idx + 2 < len(parts):
            facet = parts[agents_idx + 1]
    except ValueError:
        pass

    # Extract facet from facets/{facet}/... paths
    if parts[0] == "facets" and len(parts) >= 3:
        facet = parts[1]
        # Day from YYYYMMDD filename (events/entities/todos/news)
        if len(parts) >= 4 and DATE_RE.fullmatch(basename):
            day = basename

    # Extract day from imports/YYYYMMDD_HHMMSS/...
    if parts[0] == "imports" and len(parts) >= 2:
        import_id = parts[1]
        day = import_id.split("_")[0] if "_" in import_id else import_id[:8]

    # Extract day from config/actions/YYYYMMDD.jsonl (journal-level logs)
    if parts[0] == "config" and len(parts) >= 3 and parts[1] == "actions":
        if DATE_RE.fullmatch(basename):
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
            # Daily agent outputs, segment markdown: use basename
            topic = basename

    return {"day": day, "facet": facet, "topic": topic}


# Registry mapping glob patterns to (module_path, function_name, indexed).
# Patterns are matched against journal-relative paths and must be specific
# enough to use as Path.glob() arguments from the journal root.  The indexed
# flag controls whether find_formattable_files() collects matching files for
# the search index.  Adding a new journal content location requires a new
# entry here — see docs/JOURNAL.md "Search Index" for details.
#
# Order matters: first match wins, so place specific patterns before general ones.
FORMATTERS: dict[str, tuple[str, str, bool]] = {
    # JSONL formatters (indexed)
    "config/actions/*.jsonl": ("think.facets", "format_logs", True),
    "facets/*/entities/*.jsonl": ("think.entities.formatting", "format_entities", True),
    "facets/*/events/*.jsonl": ("think.events", "format_events", True),
    "facets/*/todos/*.jsonl": ("apps.todos.todo", "format_todos", True),
    "facets/*/logs/*.jsonl": ("think.facets", "format_logs", True),
    # Raw transcripts — formattable but not indexed (agent outputs are more useful)
    "*/*/audio.jsonl": ("observe.hear", "format_audio", False),
    "*/*/*_audio.jsonl": ("observe.hear", "format_audio", False),
    "*/*/screen.jsonl": ("observe.screen", "format_screen", False),
    "*/*/*_screen.jsonl": ("observe.screen", "format_screen", False),
    # Markdown — specific journal paths (all indexed)
    "*/agents/*.md": ("think.markdown", "format_markdown", True),
    "*/agents/*/*.md": ("think.markdown", "format_markdown", True),
    "*/*/agents/*.md": ("think.markdown", "format_markdown", True),
    "*/*/agents/*/*.md": ("think.markdown", "format_markdown", True),
    "facets/*/news/*.md": ("think.markdown", "format_markdown", True),
    "imports/*/summary.md": ("think.markdown", "format_markdown", True),
    "apps/*/agents/*.md": ("think.markdown", "format_markdown", True),
}


def get_formatter(file_path: str) -> Callable | None:
    """Return formatter function for a journal-relative file path.

    Matches against registered glob patterns (regardless of indexed flag).

    Args:
        file_path: Journal-relative path (e.g., "20240101/agents/flow.md")

    Returns:
        Formatter function or None if no pattern matches
    """
    for pattern, (module_path, func_name, _indexed) in FORMATTERS.items():
        if fnmatch.fnmatch(file_path, pattern):
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

    Globs each indexed FORMATTERS pattern from the journal root to discover
    files.  The registry is the single source of truth for what gets indexed.

    Args:
        journal: Path to journal root directory

    Returns:
        Mapping of journal-relative paths to absolute paths
    """
    files: dict[str, str] = {}
    journal_path = Path(journal)

    for pattern, (_mod, _func, indexed) in FORMATTERS.items():
        if not indexed:
            continue
        for match in journal_path.glob(pattern):
            if match.is_file():
                rel = str(match.relative_to(journal_path))
                files[rel] = str(match)

    return files


def format_file(
    file_path: str | Path,
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load file, detect formatter, return formatted chunks and metadata.

    File must be under JOURNAL_PATH. Supports JSONL and Markdown files.

    Args:
        file_path: Absolute or journal-relative path to file
        context: Optional context dict passed to formatter

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with "markdown" key (and optional "timestamp")
            - meta: Dict with optional "header" and "error" keys

    Raises:
        ValueError: If file is outside journal or no formatter found
        FileNotFoundError: If file doesn't exist
    """
    journal_path = Path(get_journal()).resolve()
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Require file to be under journal
    if not file_path.is_relative_to(journal_path):
        raise ValueError(f"File is outside journal directory: {file_path}")

    rel_path = str(file_path.relative_to(journal_path))

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
        from think.markdown import chunk_markdown

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
