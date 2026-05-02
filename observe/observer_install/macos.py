# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""macOS observer install redirect."""

from __future__ import annotations

from .common import emit_json

REDIRECT_TEXT = """solstone for macOS is delivered as a signed app bundle.
Install it from https://solstone.app/observers

After installing, the macOS app pairs itself with this solstone host
using the same registration flow surfaced via 'sol observer create'."""


class MacosDriver:
    """Redirect macOS users to the signed app bundle."""

    def run(self, args) -> int:
        if args.json_output:
            emit_json(
                {
                    "platform": "macos",
                    "name": args.name,
                    "source_path": None,
                    "service_unit": None,
                    "key_prefix": None,
                    "server_url": args.server_url,
                    "config_path": None,
                    "marker_path": None,
                    "status": "redirected",
                    "version": None,
                    "dry_run": bool(args.dry_run),
                }
            )
            return 0

        if args.dry_run:
            print("Dry-run: would direct you to download solstone-macos:")
            print()
        print(REDIRECT_TEXT)
        return 0
