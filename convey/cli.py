# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI entry points for Convey web interface."""

from __future__ import annotations

import argparse
import logging
import os

from flask import Flask

from apps.events import discover_handlers, start_dispatcher, stop_dispatcher

from .bridge import start_bridge, stop_bridge

logger = logging.getLogger(__name__)


def _resolve_config_password() -> str:
    """Return the configured Convey password from journal config."""
    from think.utils import get_config

    try:
        config = get_config()
        convey_config = config.get("convey", {})
        return convey_config.get("password", "")
    except Exception:
        return ""


def run_service(
    app: Flask,
    *,
    host: str = "0.0.0.0",
    port: int,
    debug: bool = False,
    start_watcher: bool = True,
) -> None:
    """Run the Convey service, optionally starting the Cortex watcher."""

    if start_watcher:
        # In debug mode with reloader, only start in child process
        # In non-debug mode, always start (no reloader)
        # WERKZEUG_RUN_MAIN is set to 'true' only in the child/main process
        should_start = not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
        if should_start:
            # Discover and start event handlers before bridge
            discover_handlers()
            start_dispatcher()
            logger.info("Starting Callosum bridge")
            start_bridge()
        else:
            logger.debug("Skipping bridge start in reloader parent process")

    try:
        app.run(host=host, port=port, debug=debug)
    finally:
        stop_bridge()
        stop_dispatcher()


def main() -> None:
    """Main CLI entry point for convey command."""
    from pathlib import Path

    from think.utils import (
        get_journal,
        setup_cli,
        write_service_port,
    )

    from . import create_app
    from .maint import run_pending_tasks

    parser = argparse.ArgumentParser(description="Convey web interface")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to serve on",
    )
    parser.add_argument(
        "--skip-maint",
        action="store_true",
        help="Skip running pending maintenance tasks",
    )
    args = setup_cli(parser)
    journal = get_journal()

    # Run pending maintenance tasks before starting
    if not args.skip_maint:
        ran, succeeded = run_pending_tasks(Path(journal))
        if ran > 0:
            logger.info(f"Completed {succeeded}/{ran} maintenance task(s)")

    app = create_app(journal)
    password = _resolve_config_password()
    if password:
        logger.info("Password authentication enabled")
    else:
        logger.warning(
            "No password configured - add to config/journal.json to enable authentication"
        )

    # Write port to health directory for discovery by other tools
    write_service_port("convey", args.port)
    logger.info(f"Convey starting on port {args.port}")

    run_service(app, host="0.0.0.0", port=args.port, debug=args.debug)
