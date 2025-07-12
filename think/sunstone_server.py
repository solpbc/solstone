import argparse
import os

from think.mcp_server import create_server
from think.utils import setup_cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", help="Path to the journal directory")
    args = setup_cli(ap)

    journal_path = args.journal or os.getenv("JOURNAL_PATH")
    if not journal_path:
        raise ValueError(
            "JOURNAL_PATH environment variable must be set or --journal argument provided"
        )

    create_server(journal_path).run()
