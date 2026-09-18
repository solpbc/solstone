#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Check repository files for UTF-8 encoding, trailing newlines, and trailing whitespace."""

from pathlib import Path
import sys

TEXT_EXTS = {".py", ".json", ".md", ".sh", ".txt", ".pub", ".keyid", "Makefile", "LICENSE"}


def check_file(path: Path) -> list[str]:
    errors = []
    try:
        raw = path.read_bytes()
    except Exception as err:
        return [f"could not read: {err}"]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        return [f"not valid UTF-8: {err}"]

    if raw and not raw.endswith(b"\n"):
        errors.append("missing trailing newline")

    lines = text.splitlines()
    for idx, line in enumerate(lines, 1):
        if line.endswith(" ") or line.endswith("\t"):
            errors.append(f"line {idx} has trailing whitespace")

    return errors


def main() -> int:
    repo_root = Path(__file__).parent.parent
    has_errors = False

    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in (".git", "__pycache__", "testdata") for part in path.parts):
            continue

        if path.suffix in TEXT_EXTS or path.name in TEXT_EXTS:
            errs = check_file(path)
            if errs:
                has_errors = True
                for e in errs:
                    print(f"{path.relative_to(repo_root)}: {e}")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
