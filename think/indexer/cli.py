"""CLI functionality for the indexer."""

import argparse
from typing import Any

from think.utils import journal_log, setup_cli

from .journal import (
    reset_journal_index,
    scan_journal,
    search_journal,
)


def _display_search_results(results: list[dict[str, Any]]) -> None:
    """Display search results in a consistent format."""
    for idx, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        snippet = (
            r["text"][:100] + "..."
            if len(r.get("text", "")) > 100
            else r.get("text", "")
        )
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
        help="Scan journal and update the index",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the index before rescan",
    )
    parser.add_argument(
        "--day",
        help="Filter search results by YYYYMMDD day",
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

    args = setup_cli(parser)

    import os

    journal = os.getenv("JOURNAL_PATH")

    if not args.rescan and not args.reset and args.query is None:
        parser.print_help()
        return

    if args.reset:
        reset_journal_index(journal)

    if args.rescan:
        changed = scan_journal(journal, verbose=args.verbose)
        if changed:
            journal_log("indexer journal rescan ok")

    if args.query is not None:
        query_kwargs: dict[str, Any] = {}
        if args.day:
            query_kwargs["day"] = args.day
        if args.facet:
            query_kwargs["facet"] = args.facet
        if args.topic:
            query_kwargs["topic"] = args.topic

        if args.query:
            # Single query mode
            _total, results = search_journal(args.query, 10, **query_kwargs)
            _display_search_results(results)
        else:
            # Interactive mode
            while True:
                try:
                    query = input("search> ").strip()
                except EOFError:
                    break
                if not query:
                    break
                _total, results = search_journal(query, 10, **query_kwargs)
                _display_search_results(results)
