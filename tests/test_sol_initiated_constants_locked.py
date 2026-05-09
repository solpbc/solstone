# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

LOCKED_LITERALS = (
    "sol_chat_request",
    "sol_chat_request_superseded",
    "owner_chat_open",
    "owner_chat_dismissed",
    "sol_initiated",
    "SOLSTONE_SOL_CHAT_REQUEST",
)

ALLOWED_PATHS = {
    Path("solstone/convey/sol_initiated/copy.py"),
    Path("solstone/convey/chat_stream.py"),
    Path("tests/test_sol_initiated_constants_locked.py"),
    Path("docs/design/sol_initiated_chat_lode1.md"),
}

SEARCH_ROOTS = (
    Path("solstone"),
    Path("tests"),
    Path("docs/design"),
)


def test_locked_literals_stay_in_the_contract_files() -> None:
    hits: list[str] = []

    for path in _iter_files():
        if path in ALLOWED_PATHS:
            continue
        for line_no, literal in _locked_literal_hits(path):
            hits.append(f"{path}:{line_no}: {literal}")

    assert hits == []


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".md"}
        )
    return sorted(files)


def _locked_literal_hits(path: Path) -> list[tuple[int, str]]:
    if path.suffix == ".py":
        return _python_string_literal_hits(path)

    hits: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for literal in LOCKED_LITERALS:
            if literal in line:
                hits.append((line_no, literal))
    return hits


def _python_string_literal_hits(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        value = _literal_value(token.string)
        if not isinstance(value, str):
            value = token.string
        for literal in LOCKED_LITERALS:
            if literal in value:
                hits.append((token.start[0], literal))
    return hits


def _literal_value(raw: str) -> object:
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw
