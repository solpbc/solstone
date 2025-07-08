import argparse
import os

from dotenv import load_dotenv

from think.mcp_server import create_server

if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", help="Path to the journal directory")
    opts = ap.parse_args()

    journal_path = opts.journal or os.getenv("JOURNAL_PATH")
    if not journal_path:
        raise ValueError("JOURNAL_PATH environment variable must be set or --journal argument provided")

    create_server(journal_path).run()
