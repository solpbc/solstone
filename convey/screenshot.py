#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Screenshot utility for Convey web views."""

from __future__ import annotations

import argparse
import os
import sys

from playwright.sync_api import sync_playwright

from think.utils import read_service_port, setup_cli


class _HelpOnErrorParser(argparse.ArgumentParser):
    """ArgumentParser that shows full help on any error."""

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        sys.stderr.write(f"\nerror: {message}\n")
        sys.exit(2)


def screenshot(
    route: str,
    output_path: str = "logs/screenshot.png",
    host: str = "localhost",
    port: int = 8000,
    width: int = 1440,
    height: int = 900,
    script: str | None = None,
    facet: str | None = None,
    delay: int = 0,
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
        script: Optional JavaScript to execute before taking screenshot
        facet: Optional facet to select (use "all" for all-facet mode)
        delay: Milliseconds to wait after page load before screenshot (default: 0)
    """
    # Ensure route has leading slash
    if not route.startswith("/"):
        route = "/" + route

    url = f"http://{host}:{port}{route}"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": width, "height": height})

        # Set facet selection cookie if specified
        if facet:
            if facet.lower() == "all":
                # Clear cookie for all-facet mode (empty value)
                cookie_value = ""
            else:
                cookie_value = facet
            context.add_cookies(
                [
                    {
                        "name": "selectedFacet",
                        "value": cookie_value,
                        "domain": host,
                        "path": "/",
                    }
                ]
            )

        page = context.new_page()
        page.goto(url)

        # Wait for page to be fully loaded
        page.wait_for_load_state("networkidle")

        # Wait additional delay if specified (for JS to process hash, animations, etc.)
        if delay > 0:
            page.wait_for_timeout(delay)

        # Execute custom JavaScript if provided
        if script:
            page.evaluate(script)
            # Small delay for script effects to render
            page.wait_for_timeout(100)

        page.screenshot(path=output_path, full_page=True)
        browser.close()

    print(f"Screenshot saved to {output_path}")


def _get_available_facets() -> list[str]:
    """Return list of available facet names."""
    try:
        from think.facets import get_facets

        return sorted(get_facets().keys())
    except Exception:
        return []


def _validate_facet(facet: str, available: list[str]) -> None:
    """Warn if facet doesn't exist (non-fatal)."""
    if facet.lower() == "all":
        return

    if facet not in available:
        print(f"Warning: facet '{facet}' not found in journal")


def main() -> None:
    """CLI entry point for screenshot utility."""
    # Get available facets for help text and validation
    available_facets = _get_available_facets()
    facet_choices = ", ".join(available_facets) if available_facets else "none found"
    facet_help = f"Facet to select: {facet_choices}, or 'all' for all-facet mode"

    parser = _HelpOnErrorParser(description="Capture screenshots of Convey web views")
    parser.add_argument("route", help="Route to screenshot (e.g., /, /facets)")
    parser.add_argument(
        "-o",
        "--output",
        default="logs/screenshot.png",
        help="Output path (default: logs/screenshot.png)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: read from Convey port file)",
    )
    parser.add_argument("--width", type=int, default=1440, help="Viewport width")
    parser.add_argument("--height", type=int, default=900, help="Viewport height")
    parser.add_argument(
        "--script",
        help="JavaScript to execute before taking screenshot",
    )
    parser.add_argument(
        "--facet",
        help=facet_help,
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=None,
        help="Delay in ms after page load (default: 500 if route has #fragment, else 0)",
    )

    args = setup_cli(parser)

    # Determine port: CLI arg takes precedence, then port file, then error
    if args.port is not None:
        port = args.port
    else:
        port = read_service_port("convey")
        if port is None:
            print(
                "Error: Convey port not found. Is Convey running? "
                "Use --port to specify manually.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Auto-set delay for fragment routes if not explicitly specified
    if args.delay is None:
        args.delay = 500 if "#" in args.route else 0

    # Validate facet if specified
    if args.facet:
        _validate_facet(args.facet, available_facets)

    screenshot(
        route=args.route,
        output_path=args.output,
        port=port,
        width=args.width,
        height=args.height,
        script=args.script,
        facet=args.facet,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
