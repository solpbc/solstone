import argparse
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth import OAuthProvider, RSAKeyPair

from .indexer import (
    load_cache,
    save_cache,
    scan_occurrences,
    scan_ponders,
    search_occurrences,
    search_ponders,
)


def create_server(journal: str) -> FastMCP:
    cache = load_cache(journal)
    if scan_ponders(journal, cache) | scan_occurrences(journal, cache):
        save_cache(journal, cache)

    key_pair = RSAKeyPair.generate()

    auth = OAuthProvider(
        issuer_url="https://sunstone.example.com",
        key_pair=key_pair,
        scopes_supported=["search", "search_occurrence"],
        allow_dynamic_client_registration=True,
    )

    mcp = FastMCP(name="sunstone", auth=auth)

    @mcp.tool(title="Search ponders", scope="search")
    def search_ponder(query: str, n_results: int = 5):
        return search_ponders(journal, query, n_results)

    @mcp.tool(title="Search occurrences", scope="search_occurrence")
    def search_occurrence(query: str, n_results: int = 5):
        return search_occurrences(journal, query, n_results)

    return mcp


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    journal = os.getenv("JOURNAL_PATH") or parser.error("JOURNAL_PATH not set")
    create_server(journal).run(
        "streamable-http",
        host="0.0.0.0",
        port=args.port,
    )


if __name__ == "__main__":
    main()
