import argparse

from dotenv import load_dotenv

from think.mcp_server import create_server

if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True, help="Path to the .jrnl database")
    opts = ap.parse_args()

    create_server(opts.journal).run()
