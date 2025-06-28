import argparse

from mcp.server import FastMCP

from .indexer import load_cache, save_cache, scan_ponders, search_ponders


def create_server(journal: str) -> FastMCP:
    cache = load_cache(journal)
    changed = scan_ponders(journal, cache)
    if changed:
        save_cache(journal, cache)

    server = FastMCP(name="sunstone")

    @server.tool(name="search", title="Search ponders", description="Search journal ponders")
    def search(query: str, n_results: int = 5):
        return search_ponders(journal, query, n_results)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP server for journal search")
    parser.add_argument("journal", help="Path to the journal directory")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol",
    )
    args = parser.parse_args()

    server = create_server(args.journal)
    server.run(args.transport)


if __name__ == "__main__":
    main()
