import argparse
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from .indexer import (
    search_occurrences,
    search_ponders,
)


def create_server(journal: str) -> FastMCP:
    mcp = FastMCP(name="sunstone")

    @mcp.tool(title="Search ponders")
    def search_ponder(query: str, n_results: int = 5):
        return search_ponders(journal, query, n_results)

    @mcp.tool(title="Search occurrences")
    def search_occurrence(query: str, n_results: int = 5):
        return search_occurrences(journal, query, n_results)

    return mcp


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run using STDIO transport instead of HTTP",
    )
    args = parser.parse_args()

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
