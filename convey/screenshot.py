#!/usr/bin/env python3
"""Screenshot utility for Convey web views."""

from __future__ import annotations

import argparse
import os
import sys

from playwright.sync_api import sync_playwright

from think.utils import setup_cli


def screenshot(
    route: str,
    output_path: str = "logs/screenshot.png",
    host: str = "localhost",
    port: int = 8000,
    width: int = 1440,
    height: int = 900,
) -> None:
    """
    Capture screenshot of a Convey view.

    Args:
        route: The route to screenshot (e.g., "/", "/facets")
        output_path: Where to save the screenshot (default: logs/screenshot.png)
        host: Server host (default: localhost)
        port: Server port (default: 8000)
        width: Viewport width (default: 1440)
        height: Viewport height (default: 900)
    """
    url = f"http://{host}:{port}{route}"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": width, "height": height})

        page = context.new_page()
        page.goto(url)

        # Wait for page to be fully loaded
        page.wait_for_load_state("networkidle")

        page.screenshot(path=output_path, full_page=True)
        browser.close()

    print(f"Screenshot saved to {output_path}")


def main() -> None:
    """CLI entry point for screenshot utility."""
    parser = argparse.ArgumentParser(
        description="Capture screenshots of Convey web views"
    )
    parser.add_argument("route", help="Route to screenshot (e.g., /, /facets)")
    parser.add_argument(
        "-o",
        "--output",
        default="logs/screenshot.png",
        help="Output path (default: logs/screenshot.png)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--width", type=int, default=1440, help="Viewport width")
    parser.add_argument("--height", type=int, default=900, help="Viewport height")

    args = setup_cli(parser)

    screenshot(
        route=args.route,
        output_path=args.output,
        port=args.port,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
