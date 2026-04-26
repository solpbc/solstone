# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""sol config — show and rewrite the embedded journal path in the wrapper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from think.install_guard import (
    alias_path,
    parse_wrapper,
    render_wrapper,
    validate_journal_path_for_wrapper,
    wrapper_lock,
    write_wrapper_atomic,
)
from think.utils import SolstoneNotConfigured, get_journal_info, get_project_root


def _read_wrapper_status() -> tuple[str, str | None]:
    alias = alias_path()
    if not alias.exists() and not alias.is_symlink():
        return "absent", None
    if alias.is_symlink():
        return "legacy-symlink", None

    try:
        content = alias.read_text(encoding="utf-8")
    except OSError:
        return "foreign", None

    parsed = parse_wrapper(content)
    if parsed is None:
        return "foreign", None
    return "managed", parsed["journal"]


def cmd_show() -> int:
    wrapper_status, embedded_journal = _read_wrapper_status()

    try:
        path, info_source = get_journal_info()
    except SolstoneNotConfigured as exc:
        print(f"sol config: {exc}", file=sys.stderr)
        return 1

    if info_source == "env":
        if (
            embedded_journal is not None
            and os.environ.get("SOLSTONE_JOURNAL") == embedded_journal
        ):
            user_source = "wrapper-embedded"
        else:
            user_source = "caller-override"
    else:
        user_source = "source-tree fallback"

    print(f"path: {path}")
    print(f"source: {user_source}")
    print(f"wrapper-status: {wrapper_status}")
    return 0


def cmd_journal(target_path: str) -> int:
    target = Path(target_path).expanduser().resolve()
    target_str = str(target)

    try:
        validate_journal_path_for_wrapper(target_str)
    except ValueError as exc:
        print(f"sol config: refused: {exc}", file=sys.stderr)
        return 1

    project_root = Path(get_project_root())
    is_source_checkout = (project_root / "pyproject.toml").exists() and (
        project_root / ".git"
    ).exists()
    source_tree_journal = (project_root / "journal").resolve()
    if target == source_tree_journal and not is_source_checkout:
        print(
            "sol config: refused: "
            f"{target_str} is the source-tree fallback path but this is not a "
            "source checkout",
            file=sys.stderr,
        )
        return 1

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"sol config: refused: cannot create {target_str}: {exc}", file=sys.stderr
        )
        return 1

    alias = alias_path()
    if not alias.exists() or alias.is_symlink():
        print(
            "sol config: refused: "
            f"{alias} is not a managed wrapper (run 'make install-service' to "
            "install the wrapper first)",
            file=sys.stderr,
        )
        return 1

    try:
        content = alias.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"sol config: refused: cannot read {alias}: {exc}", file=sys.stderr)
        return 1

    parsed = parse_wrapper(content)
    if parsed is None:
        print(
            "sol config: refused: "
            f"{alias} is not a managed wrapper (run 'make install-service' to "
            "install the wrapper first)",
            file=sys.stderr,
        )
        return 1

    if parsed["journal"] == target_str:
        print(f"sol config: journal already set to {target_str}")
        return 0

    restart_sol = parsed["sol_bin"]
    with wrapper_lock():
        try:
            current_content = alias.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"sol config: refused: cannot read {alias}: {exc}", file=sys.stderr)
            return 1

        current = parse_wrapper(current_content)
        if current is None:
            print(
                "sol config: refused: "
                f"{alias} is not a managed wrapper (run 'make install-service' to "
                "install the wrapper first)",
                file=sys.stderr,
            )
            return 1

        if current["journal"] == target_str:
            print(f"sol config: journal already set to {target_str}")
            return 0

        new_content = render_wrapper(target_str, current["sol_bin"])
        write_wrapper_atomic(alias, new_content)
        restart_sol = current["sol_bin"]

    try:
        result = subprocess.run(
            [restart_sol, "service", "restart", "--if-installed"],
            check=False,
        )
    except FileNotFoundError as exc:
        print(
            "sol config: wrapper rewritten to "
            f"{target_str} but service restart could not run ({exc}); restart "
            "manually",
            file=sys.stderr,
        )
        return 2

    if result.returncode != 0:
        print(
            "sol config: wrapper rewritten to "
            f"{target_str} but 'sol service restart --if-installed' exited "
            f"{result.returncode}; investigate and restart manually",
            file=sys.stderr,
        )
        return 2

    print(f"sol config: journal set to {target_str}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="sol config")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("show", help="show the configured journal path and source")
    journal_parser = subparsers.add_parser(
        "journal",
        help="rewrite the wrapper's embedded journal path",
    )
    journal_parser.add_argument(
        "path", help="absolute path to the new journal directory"
    )
    args = parser.parse_args()

    if args.cmd == "show":
        return cmd_show()
    if args.cmd == "journal":
        return cmd_journal(args.path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
