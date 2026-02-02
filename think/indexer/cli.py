# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI functionality for the indexer."""

import argparse
from typing import Any

from think.utils import get_journal, journal_log, setup_cli

from .journal import (
    index_file,
    reset_journal_index,
    scan_journal,
    search_counts,
    search_journal,
)


def _format_count_column(
    items: list[tuple[str, int]], total: int, top_n: int
) -> list[str]:
    """Format a column of count items with overflow indicator."""
    lines = [f"{name} ({count})" for name, count in items[:top_n]]
    if total > top_n:
        lines.append(f"... +{total - top_n} more")
    return lines


def _display_counts(counts: dict[str, Any], top_n: int = 5) -> None:
    """Display aggregated counts in a compact table format."""
    total = counts["total"]
    facets = counts["facets"]  # Counter
    topics = counts["topics"]  # Counter
    days = counts["days"]  # Counter

    print(f"Total: {total:,} chunks\n")

    # Build columns
    facet_col = _format_count_column(facets.most_common(top_n), len(facets), top_n)
    topic_col = _format_count_column(topics.most_common(top_n), len(topics), top_n)
    day_col = _format_count_column(
        sorted(days.items(), reverse=True)[:top_n], len(days), top_n
    )

    # Header and rows
    print(f"{'Facet':<20} {'Topic':<20} {'Day':<20}")
    print("-" * 60)

    from itertools import zip_longest

    for f, t, d in zip_longest(facet_col, topic_col, day_col, fillvalue=""):
        print(f"{f:<20} {t:<20} {d:<20}")

    print()


def _display_search_results(
    results: list[dict[str, Any]], total: int, offset: int
) -> None:
    """Display search results in a consistent format."""
    if total == 0 or not results:
        print("No results found")
        return

    # Show pagination context
    start = offset + 1
    end = offset + len(results)
    print(f"Showing {start}-{end} of {total} results\n")

    for idx, r in enumerate(results, start):
        meta = r.get("metadata", {})
        text = r.get("text", "").replace("\n", " ")
        snippet = text[:100] + "..." if len(text) > 100 else text
        label = meta.get("topic") or meta.get("time") or ""
        facet = meta.get("facet")
        facet_str = f" ({facet})" if facet else ""
        print(f"{idx}. {meta.get('day')} {label}{facet_str}: {snippet}")


def main() -> None:
    """Main CLI entry point for the indexer."""
    parser = argparse.ArgumentParser(
        description="Index journal content (insights, transcripts, events, entities, todos)"
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan and update index (light mode: today + facets/imports/apps, excludes historical days)",
    )
    parser.add_argument(
        "--rescan-full",
        action="store_true",
        help="Full rescan including all historical day directories",
    )
    parser.add_argument(
        "--rescan-file",
        metavar="PATH",
        help="Index a specific file (absolute or journal-relative path)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the index before rescan",
    )
    parser.add_argument(
        "--day",
        help="Filter search results by exact YYYYMMDD day",
    )
    parser.add_argument(
        "--day-from",
        help="Filter search results by date range start (YYYYMMDD, inclusive)",
    )
    parser.add_argument(
        "--day-to",
        help="Filter search results by date range end (YYYYMMDD, inclusive)",
    )
    parser.add_argument(
        "--facet",
        help="Filter search results by facet name",
    )
    parser.add_argument(
        "--topic",
        help="Filter search results by topic (e.g., 'flow', 'audio', 'event')",
    )
    parser.add_argument(
        "-q",
        "--query",
        nargs="?",
        const="",
        help="Run query (interactive mode if no query provided)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of results to skip for pagination (default: 0)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of items to show per count column (default: 5)",
    )

    args = setup_cli(parser)
    journal = get_journal()

    if (
        not args.rescan
        and not args.rescan_full
        and not args.rescan_file
        and not args.reset
        and args.query is None
    ):
        parser.print_help()
        return

    if args.reset:
        reset_journal_index(journal)

    if args.rescan_file:
        # Single file indexing (incompatible with --rescan/--rescan-full)
        if args.rescan or args.rescan_full:
            parser.error("--rescan-file cannot be used with --rescan or --rescan-full")
        try:
            index_file(journal, args.rescan_file, verbose=args.verbose)
            journal_log(f"indexer file indexed: {args.rescan_file}")
        except (ValueError, FileNotFoundError) as e:
            parser.error(str(e))
    elif args.rescan or args.rescan_full:
        changed = scan_journal(journal, verbose=args.verbose, full=args.rescan_full)
        if changed:
            journal_log("indexer journal rescan ok")

    if args.query is not None:
        query_kwargs: dict[str, Any] = {}
        if args.day:
            query_kwargs["day"] = args.day
        if getattr(args, "day_from", None):
            query_kwargs["day_from"] = args.day_from
        if getattr(args, "day_to", None):
            query_kwargs["day_to"] = args.day_to
        if args.facet:
            query_kwargs["facet"] = args.facet
        if args.topic:
            query_kwargs["topic"] = args.topic

        if args.query:
            # Single query mode - show counts then results
            counts = search_counts(args.query, **query_kwargs)
            _display_counts(counts, args.top)
            total, results = search_journal(
                args.query, args.limit, args.offset, **query_kwargs
            )
            _display_search_results(results, total, args.offset)
        else:
            # Interactive mode
            while True:
                try:
                    query = input("search> ").strip()
                except EOFError:
                    break
                if not query:
                    break
                counts = search_counts(query, **query_kwargs)
                _display_counts(counts, args.top)
                total, results = search_journal(
                    query, args.limit, args.offset, **query_kwargs
                )
                _display_search_results(results, total, args.offset)
