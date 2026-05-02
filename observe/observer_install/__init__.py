# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Observer install command dispatcher."""

from __future__ import annotations

from .common import InstallError, detect_platform, emit_json, print_summary


def run_install(args) -> int:
    """Run the platform-specific observer installer."""
    try:
        platform_name = detect_platform(args.platform)
        if platform_name == "macos":
            from .macos import MacosDriver

            driver = MacosDriver()
        elif platform_name == "linux":
            from .linux import LinuxDriver

            driver = LinuxDriver()
        elif platform_name == "tmux":
            from .tmux import TmuxDriver

            driver = TmuxDriver()
        else:
            raise InstallError(
                "unsupported observer platform",
                hint="pass --platform linux, --platform tmux, or install the macOS app",
            )
        return driver.run(args)
    except InstallError as exc:
        result = {
            "platform": getattr(args, "platform", None),
            "name": getattr(args, "name", None),
            "source_path": None,
            "service_unit": None,
            "key_prefix": None,
            "server_url": getattr(args, "server_url", None),
            "config_path": None,
            "marker_path": None,
            "status": "error",
            "version": None,
            "dry_run": bool(getattr(args, "dry_run", False)),
            "error": str(exc),
            "hint": exc.hint,
        }
        if getattr(args, "json_output", False):
            emit_json(result)
        else:
            print_summary(result)
        return exc.code
