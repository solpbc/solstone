# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI interface for app tools via Typer.

Provides ``sol call <app> <command> [args]`` as a human-friendly CLI that
parallels app tool functions. Each app can contribute a ``call.py``
module exporting a ``app = typer.Typer()`` instance whose commands are
auto-discovered and mounted as sub-commands.

Discovery scans ``apps/*/call.py``, imports modules, and mounts subcommands.
"""

import importlib
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

call_app = typer.Typer(
    name="call",
    help="Call app functions from the command line.",
    no_args_is_help=True,
)


def _discover_app_calls() -> None:
    """Discover and mount Typer sub-apps from apps/*/call.py.

    Each ``call.py`` must export an ``app`` variable that is a
    ``typer.Typer`` instance.  The app directory name becomes the
    sub-command name (e.g. ``sol call todos list ...``).

    Errors in one app do not prevent others from loading.
    """
    apps_dir = Path(__file__).parent.parent / "apps"

    if not apps_dir.exists():
        logger.debug("No apps/ directory found, skipping app call discovery")
        return

    for app_dir in sorted(apps_dir.iterdir()):
        if not app_dir.is_dir() or app_dir.name.startswith("_"):
            continue

        call_file = app_dir / "call.py"
        if not call_file.exists():
            continue

        app_name = app_dir.name

        try:
            module = importlib.import_module(f"apps.{app_name}.call")

            sub_app = getattr(module, "app", None)
            if not isinstance(sub_app, typer.Typer):
                logger.warning(
                    f"apps/{app_name}/call.py has no 'app' Typer instance, skipping"
                )
                continue

            call_app.add_typer(sub_app, name=app_name)
            logger.info(f"Loaded CLI commands from app: {app_name}")
        except Exception as e:
            logger.error(
                f"Failed to load CLI from app '{app_name}': {e}", exc_info=True
            )


_discover_app_calls()

# Mount built-in journal CLI (not auto-discovered since it lives under think/)
from think.tools.call import app as journal_app

call_app.add_typer(journal_app, name="journal")


def main() -> None:
    """Entry point for ``sol call``."""
    call_app()
