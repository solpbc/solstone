# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Guard user-level sol alias ownership."""

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path

import userpath


class AliasState(Enum):
    WORKTREE = "worktree"
    ABSENT = "absent"
    OWNED = "owned"
    CROSS_REPO = "cross_repo"
    DANGLING = "dangling"
    NOT_SYMLINK = "not_symlink"


def alias_path() -> Path:
    return Path.home() / ".local" / "bin" / "sol"


def expected_target(curdir: Path) -> Path:
    return curdir / ".venv" / "bin" / "sol"


def check_alias(curdir: Path) -> tuple[AliasState, Path | None]:
    if (curdir / ".git").is_file():
        return AliasState.WORKTREE, None

    alias = alias_path()
    if not alias.exists() and not alias.is_symlink():
        return AliasState.ABSENT, None

    if alias.is_symlink():
        target = Path(os.readlink(alias))
        if not target.is_absolute():
            target = alias.parent / target
        target = target.resolve()
        if not target.exists():
            return AliasState.DANGLING, target
        if target == expected_target(curdir).resolve():
            return AliasState.OWNED, target
        return AliasState.CROSS_REPO, target

    return AliasState.NOT_SYMLINK, None


def format_error(
    state: AliasState,
    curdir: Path,
    _alias: Path,
    other_target: Path | None,
) -> str:
    if state is AliasState.WORKTREE:
        return (
            f"ERROR: refusing to run from a git worktree ({curdir}). "
            "Run from the primary clone."
        )

    if state is AliasState.CROSS_REPO:
        installed = f"  installed:  {other_target}"
    elif state is AliasState.DANGLING:
        installed = f"  installed:  dangling: {other_target} does not exist"
    else:
        installed = "  installed:  not a symlink"

    return "\n".join(
        [
            "ERROR: Another solstone install owns ~/.local/bin/sol.",
            f"  this repo:  {curdir}",
            installed,
            "Run 'make uninstall-service' from the installed repo first,",
            "or remove ~/.local/bin/sol manually if that repo is gone. No --force available.",
        ]
    )


def _print_error(
    state: AliasState,
    curdir: Path,
    alias: Path,
    other_target: Path | None,
) -> None:
    sys.stderr.write(format_error(state, curdir, alias, other_target) + "\n")


def _ensure_user_bin_on_path(user_bin: Path) -> None:
    user_bin_str = str(user_bin)
    try:
        if userpath.in_current_path(user_bin_str):
            print("path: ~/.local/bin already on PATH")
            return
        if userpath.append(user_bin_str, app_name="solstone", all_shells=True):
            if userpath.need_shell_restart(user_bin_str):
                print(
                    "path: added ~/.local/bin to shell PATH — restart your shell or run 'exec $SHELL -l' to pick it up"
                )
            else:
                print("path: added ~/.local/bin to shell PATH")
            return
        print(
            'path: could not auto-add ~/.local/bin to PATH — add this line to your shell rc manually: export PATH="$HOME/.local/bin:$PATH"'
        )
    except Exception as exc:
        print(
            f'path: could not auto-add ~/.local/bin to PATH ({type(exc).__name__}: {exc}) — add this line to your shell rc manually: export PATH="$HOME/.local/bin:$PATH"'
        )


def cmd_check(curdir: Path) -> int:
    alias = alias_path()
    state, other_target = check_alias(curdir)

    if state is AliasState.ABSENT:
        print("fresh")
        return 0
    if state is AliasState.OWNED:
        print("upgrade")
        return 0

    print(state.value)
    _print_error(state, curdir, alias, other_target)
    return 1


def cmd_install(curdir: Path) -> int:
    alias = alias_path()
    state, other_target = check_alias(curdir)

    if state is AliasState.WORKTREE:
        _print_error(state, curdir, alias, other_target)
        return 1
    if state is AliasState.ABSENT:
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(expected_target(curdir))
        print("installed")
        _ensure_user_bin_on_path(alias.parent)
        return 0
    if state is AliasState.OWNED:
        alias.unlink()
        alias.symlink_to(expected_target(curdir))
        print("installed")
        _ensure_user_bin_on_path(alias.parent)
        return 0

    _print_error(state, curdir, alias, other_target)
    return 1


def cmd_uninstall(curdir: Path) -> int:
    alias = alias_path()
    state, other_target = check_alias(curdir)

    if state is AliasState.WORKTREE:
        _print_error(state, curdir, alias, other_target)
        return 1
    if state is AliasState.ABSENT:
        print("absent")
        return 0
    if state is AliasState.OWNED:
        alias.unlink()
        print("removed")
        return 0

    _print_error(state, curdir, alias, other_target)
    return 1


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1 or argv[0] not in {"check", "install", "uninstall"}:
        sys.stderr.write(
            "usage: python -m think.install_guard <check|install|uninstall>\n"
        )
        return 2

    curdir = Path.cwd().resolve()
    if argv[0] == "check":
        return cmd_check(curdir)
    if argv[0] == "install":
        return cmd_install(curdir)
    return cmd_uninstall(curdir)


if __name__ == "__main__":
    sys.exit(main())
