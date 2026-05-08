#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""Lint: [all] extra must equal union of [pdf] + [whisper]."""

import sys
from pathlib import Path

import tomllib


def main() -> int:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    expected = sorted(set(extras["pdf"]) | set(extras["whisper"]))
    actual = sorted(extras["all"])
    if expected != actual:
        print("ERROR: [all] extra drift detected", file=sys.stderr)
        print(f"  expected (= [pdf] ∪ [whisper]): {expected}", file=sys.stderr)
        print(f"  actual                        : {actual}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
