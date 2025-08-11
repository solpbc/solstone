"""CLI functionality for the indexer."""

import argparse
import os
from typing import Dict, List

from think.utils import journal_log, setup_cli

from .core import reset_index
from .entities import scan_entities, search_entities
from .events import scan_events, search_events
from .summaries import scan_summaries, search_summaries
from .transcripts import scan_transcripts, search_transcripts


def _display_search_results(results: List[Dict[str, str]]) -> None:
    """Display search results in a consistent format."""
    for idx, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        snippet = r["text"]
        label = meta.get("topic") or meta.get("time") or ""
        print(f"{idx}. {meta.get('day')} {label}: {snippet}")


def main() -> None:
    """Main CLI entry point for the indexer."""
    parser = argparse.ArgumentParser(
        description="Index summary markdown and event files"
    )
    parser.add_argument(
        "--index",
        choices=["summaries", "events", "transcripts", "entities"],
        help="Which index to operate on",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan journal and update the index before searching",
    )
    parser.add_argument(
        "--rescan-all",
        action="store_true",
        help="Scan journal and update all indexes",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the selected index before optional rescan",
    )
    parser.add_argument(
        "--day",
        help="Limit transcript query to a specific YYYYMMDD day",
    )
    parser.add_argument(
        "-q",
        "--query",
        nargs="?",
        const="",
        help="Run query (interactive mode if no query provided)",
    )

    args = setup_cli(parser)

    # Require either --rescan, --rescan-all, or -q
    if not args.rescan and not args.rescan_all and args.query is None:
        parser.print_help()
        return

    # Validate --index is required unless using --rescan-all
    if not args.rescan_all and not args.index:
        parser.error("--index is required unless using --rescan-all")

    journal = os.getenv("JOURNAL_PATH")

    if args.reset:
        reset_index(
            journal, args.index, day=args.day if args.index == "transcripts" else None
        )

    if args.rescan_all:
        # Rescan all indexes
        indexes = ["summaries", "events", "transcripts", "entities"]
        for index_name in indexes:
            if index_name == "transcripts":
                changed = scan_transcripts(journal, verbose=args.verbose)
                if changed:
                    journal_log(f"indexer {index_name} rescan ok")
            elif index_name == "events":
                changed = scan_events(journal, verbose=args.verbose)
                if changed:
                    journal_log(f"indexer {index_name} rescan ok")
            elif index_name == "summaries":
                changed = scan_summaries(journal, verbose=args.verbose)
                if changed:
                    journal_log(f"indexer {index_name} rescan ok")
            elif index_name == "entities":
                changed = scan_entities(journal, verbose=args.verbose)
                if changed:
                    journal_log(f"indexer {index_name} rescan ok")

    if args.rescan:
        if args.index == "transcripts":
            changed = scan_transcripts(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer transcripts rescan ok")
        elif args.index == "events":
            changed = scan_events(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer events rescan ok")
        elif args.index == "summaries":
            changed = scan_summaries(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer summaries rescan ok")
        elif args.index == "entities":
            changed = scan_entities(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer entities rescan ok")

    # Handle query argument
    if args.query is not None:
        if not args.index:
            parser.error("--index is required when using --query")

        if args.index == "transcripts":
            search_func = search_transcripts
            query_kwargs = {"day": args.day}
        elif args.index == "events":
            search_func = search_events
            query_kwargs = {}
        elif args.index == "entities":
            search_func = search_entities
            query_kwargs = {"day": args.day}
        else:
            search_func = search_summaries
            query_kwargs = {}
        if args.query:
            # Single query mode - run query and exit
            _total, results = search_func(args.query, 5, **query_kwargs)
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
                _total, results = search_func(query, 5, **query_kwargs)
                _display_search_results(results)
