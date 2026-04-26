# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Guard user-level sol alias ownership."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator

try:
    import userpath  # type: ignore[import-not-found]
except ImportError:  # system python without the venv: doctor.stale_alias_symlink path
    userpath = None  # type: ignore[assignment]


WRAPPER_TEMPLATE = """\
#!/bin/sh
# sol — managed by 'sol config'. Edits will be overwritten.
# managed-version: 1
: "${{SOLSTONE_JOURNAL:={journal}}}"
export SOLSTONE_JOURNAL
SOL_BIN='{sol_bin}'
if [ ! -x "$SOL_BIN" ]; then
    printf 'sol: venv binary missing or not executable: %s\\n' "$SOL_BIN" >&2
    exit 127
fi
exec "$SOL_BIN" "$@"
"""

WRAPPER_MARKER = "# managed-version: 1"
WRAPPER_VERSION = 1

_RE_MARKER = re.compile(r"(?m)^# managed-version: 1$")
_RE_JOURNAL = re.compile(r'(?m)^: "\$\{SOLSTONE_JOURNAL:=(?P<journal>[^\n]*)\}"$')
_RE_SOL_BIN = re.compile(r"(?m)^SOL_BIN='(?P<sol_bin>(?:[^']|'\\'')*)'$")

_INVALID_JOURNAL_CHARS = ("$", "`", '"', "\\")


class AliasState(Enum):
    WORKTREE = "worktree"
    ABSENT = "absent"
    OWNED = "owned"
    CROSS_REPO = "cross_repo"
    DANGLING = "dangling"
    FOREIGN = "foreign"


def alias_path() -> Path:
    return Path.home() / ".local" / "bin" / "sol"


def expected_target(curdir: Path) -> Path:
    return curdir / ".venv" / "bin" / "sol"


def render_wrapper(journal: str, sol_bin: str) -> str:
    """Render the managed wrapper for ~/.local/bin/sol."""
    escaped_sol_bin = sol_bin.replace("'", "'\\''")
    return WRAPPER_TEMPLATE.format(journal=journal, sol_bin=escaped_sol_bin)


def parse_wrapper(content: str) -> dict[str, str] | None:
    """Return embedded paths if the content is a managed wrapper."""
    if not _RE_MARKER.search(content):
        return None
    journal_match = _RE_JOURNAL.search(content)
    sol_bin_match = _RE_SOL_BIN.search(content)
    if not journal_match or not sol_bin_match:
        return None
    return {
        "journal": journal_match.group("journal"),
        "sol_bin": sol_bin_match.group("sol_bin").replace("'\\''", "'"),
    }


def write_wrapper_atomic(path: Path, content: str) -> None:
    """Atomically rewrite the managed wrapper and restore exec mode."""
    from think.entities.core import atomic_write

    atomic_write(path, content)
    os.chmod(path, 0o755)


@contextmanager
def wrapper_lock(lock_path: Path | None = None) -> Iterator[None]:
    """Hold an exclusive advisory lock while rewriting the wrapper."""
    if lock_path is None:
        lock_path = Path.home() / ".local" / "bin" / ".sol.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_fd:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


def validate_journal_path_for_wrapper(path: str) -> None:
    """Reject shell-active characters that would corrupt wrapper embedding."""
    for char in _INVALID_JOURNAL_CHARS:
        if char in path:
            raise ValueError(
                f"journal path contains shell-active character {char!r}: {path!r}"
            )


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

    try:
        content = alias.read_text(encoding="utf-8")
    except OSError:
        return AliasState.FOREIGN, None

    parsed = parse_wrapper(content)
    if parsed is None:
        return AliasState.FOREIGN, None

    target = Path(parsed["sol_bin"])
    if target.resolve() == expected_target(curdir).resolve():
        return AliasState.OWNED, target.resolve()
    return AliasState.FOREIGN, None


def _current_journal_for_alias() -> str:
    """Return the journal path a wrapper install would embed right now."""
    from think import utils as think_utils

    try:
        path, _ = think_utils.get_journal_info()
    except getattr(think_utils, "SolstoneNotConfigured", RuntimeError):
        path = str(Path.home() / "Documents" / "Solstone")
    return path


def check_alias_detail(curdir: Path) -> tuple[AliasState, str]:
    """Return alias state plus the cmd_check token for owned aliases."""
    state, _other_target = check_alias(curdir)
    if state is not AliasState.OWNED:
        return state, state.value

    alias = alias_path()
    if alias.is_symlink():
        return state, "upgrade"

    try:
        content = alias.read_text(encoding="utf-8")
    except OSError:
        return state, "upgrade"

    parsed = parse_wrapper(content)
    if parsed is None:
        return state, "upgrade"

    if (
        parsed["journal"] == _current_journal_for_alias()
        and parsed["sol_bin"] == str(expected_target(curdir))
        and (alias.stat().st_mode & 0o111) == 0o111
    ):
        return state, "current"

    return state, "upgrade"


def format_error(
    state: AliasState,
    curdir: Path,
    _alias: Path,
    other_target: Path | None,
    *,
    allow_force: bool = False,
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

    lines = [
        "ERROR: Another solstone install owns ~/.local/bin/sol.",
        f"  this repo:  {curdir}",
        installed,
        "Run 'make uninstall-service' from the installed repo first,",
        "or remove ~/.local/bin/sol manually if that repo is gone.",
    ]
    if allow_force:
        lines.append(
            "Rerun 'python -m think.install_guard install --force' only if you intend to replace it from this repo."
        )
    return "\n".join(lines)


def _print_error(
    state: AliasState,
    curdir: Path,
    alias: Path,
    other_target: Path | None,
    *,
    allow_force: bool = False,
) -> None:
    sys.stderr.write(
        format_error(
            state,
            curdir,
            alias,
            other_target,
            allow_force=allow_force,
        )
        + "\n"
    )


def _ensure_user_bin_on_path(user_bin: Path) -> None:
    # `userpath` is imported at module top with an ImportError guard, so this
    # module is importable from system python (where doctor runs) even when
    # `userpath` is not installed. This code path is only reached via
    # `cmd_install`, which only runs from inside the venv where `userpath` is
    # present; if somehow reached without `userpath`, we want a hard failure.
    if userpath is None:
        raise RuntimeError("userpath is not available; run `make install` first")
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
    state, token = check_alias_detail(curdir)

    if state is AliasState.WORKTREE:
        print("worktree")
        _print_error(state, curdir, alias, None)
        return 1
    if state is AliasState.ABSENT:
        print("fresh")
        return 0
    if state is AliasState.OWNED:
        print(token)
        return 0
    if state is AliasState.CROSS_REPO:
        print("cross_repo")
        _print_error(state, curdir, alias, check_alias(curdir)[1], allow_force=True)
        return 1
    if state is AliasState.DANGLING:
        print("dangling")
        _print_error(state, curdir, alias, check_alias(curdir)[1], allow_force=True)
        return 1
    if state is AliasState.FOREIGN:
        print("not_symlink")
        _print_error(state, curdir, alias, None, allow_force=True)
        return 1

    return 1


def cmd_install(curdir: Path, *, force: bool = False) -> int:
    alias = alias_path()
    state, other_target = check_alias(curdir)

    if state is AliasState.WORKTREE:
        _print_error(state, curdir, alias, other_target)
        return 1
    if (
        state
        in {
            AliasState.CROSS_REPO,
            AliasState.DANGLING,
            AliasState.FOREIGN,
        }
        and not force
    ):
        _print_error(state, curdir, alias, other_target, allow_force=True)
        return 1

    journal = _current_journal_for_alias()
    try:
        validate_journal_path_for_wrapper(journal)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    content = render_wrapper(journal, str(expected_target(curdir)))
    alias.parent.mkdir(parents=True, exist_ok=True)
    with wrapper_lock():
        locked_state, locked_other_target = check_alias(curdir)
        if locked_state is AliasState.WORKTREE:
            _print_error(locked_state, curdir, alias, locked_other_target)
            return 1
        if (
            locked_state
            in {
                AliasState.CROSS_REPO,
                AliasState.DANGLING,
                AliasState.FOREIGN,
            }
            and not force
        ):
            _print_error(
                locked_state,
                curdir,
                alias,
                locked_other_target,
                allow_force=True,
            )
            return 1
        if alias.is_symlink():
            alias.unlink()
        write_wrapper_atomic(alias, content)

    print("installed")
    _ensure_user_bin_on_path(alias.parent)
    return 0


def cmd_uninstall(curdir: Path) -> int:
    alias = alias_path()
    state, other_target = check_alias(curdir)

    if state is AliasState.WORKTREE:
        _print_error(state, curdir, alias, other_target)
        return 1
    if state is AliasState.ABSENT:
        print("absent")
        return 0
    if state is not AliasState.OWNED:
        _print_error(state, curdir, alias, other_target)
        return 1

    with wrapper_lock():
        locked_state, locked_other_target = check_alias(curdir)
        if locked_state is AliasState.ABSENT:
            print("absent")
            return 0
        if locked_state is not AliasState.OWNED:
            _print_error(locked_state, curdir, alias, locked_other_target)
            return 1
        alias.unlink()

    print("uninstalled")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m think.install_guard")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("check")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("uninstall")
    args = parser.parse_args(argv)
    curdir = Path.cwd().resolve()
    if args.cmd == "check":
        return cmd_check(curdir)
    if args.cmd == "install":
        return cmd_install(curdir, force=args.force)
    return cmd_uninstall(curdir)


if __name__ == "__main__":
    sys.exit(main())
