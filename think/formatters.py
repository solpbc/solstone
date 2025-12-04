"""JSONL to Markdown formatters framework.

This module provides a registry-based system for converting JSONL files
to structured markdown chunks. Each formatter is a plain function that
lives near its source domain code.

Output contract: All formatters return tuple[list[dict], dict] where:
    - list[dict]: Timestamp-ordered chunks, each with:
        - timestamp: int (unix timestamp or segment offset)
        - markdown: str (formatted markdown for this chunk)
    - dict: Metadata about the formatting with optional keys:
        - header: str - Optional header markdown (metadata summary, context, etc.)
        - error: str - Optional error/warning message (e.g., skipped entries)

Formatters receive the raw JSONL entries and are responsible for:
    - Extracting metadata from entries (typically first line)
    - Building header from metadata if applicable
    - Formatting content entries into chunks
"""

import argparse
import fnmatch
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# Registry mapping glob patterns to (module_path, function_name)
# Patterns are matched against journal-relative paths
# Order matters: first match wins, so place specific patterns before general ones
# Note: agents/*_active.jsonl is excluded - only completed agents are formatted
FORMATTERS: dict[str, tuple[str, str]] = {
    "agents/*.jsonl": ("muse.cortex", "format_agent"),
    "facets/*/entities/*.jsonl": ("think.entities", "format_entities"),
    "facets/*/entities.jsonl": ("think.entities", "format_entities"),
    "facets/*/todos/*.jsonl": ("apps.todos.todo", "format_todos"),
    "*/screen.jsonl": ("observe.reduce", "format_screen"),
    "*/*_audio.jsonl": ("observe.hear", "format_audio"),
    "*/audio.jsonl": ("observe.hear", "format_audio"),
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


def format_file(
    file_path: str | Path,
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load JSONL, detect formatter, return formatted chunks and metadata.

    Args:
        file_path: Absolute path to JSONL file
        context: Optional context dict passed to formatter

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of {"timestamp": int, "markdown": str} dicts
            - meta: Dict with optional "header" and "error" keys

    Raises:
        ValueError: If no formatter found for file path
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get journal-relative path for pattern matching
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH", "")
    journal_path = Path(journal).resolve() if journal else None

    if journal_path and file_path.is_relative_to(journal_path):
        rel_path = str(file_path.relative_to(journal_path))
    else:
        # Fall back to just the filename parts for matching
        rel_path = str(file_path)

    formatter = get_formatter(rel_path)
    if formatter is None:
        raise ValueError(f"No formatter found for: {rel_path}")

    entries = load_jsonl(file_path)

    # Build context with file path info
    ctx = context or {}
    ctx.setdefault("file_path", file_path)

    return formatter(entries, ctx)


def main() -> None:
    """CLI entry point for think-formatter."""
    from think.utils import setup_cli

    parser = argparse.ArgumentParser(
        description="Convert JSONL files to formatted markdown"
    )
    parser.add_argument("file", help="Path to JSONL file")
    parser.add_argument(
        "--join",
        action="store_true",
        help="Output concatenated markdown instead of JSON chunks",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="JSON string of context to pass to formatter",
    )
    args = setup_cli(parser)

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

    if args.join:
        # Output concatenated markdown with header first
        parts = []
        if meta.get("header"):
            parts.append(meta["header"])
        parts.extend(chunk["markdown"] for chunk in chunks)
        print("\n".join(parts))
    else:
        # Output JSON object with metadata and chunks
        print(json.dumps({"meta": meta, "chunks": chunks}, indent=2))


if __name__ == "__main__":
    main()
