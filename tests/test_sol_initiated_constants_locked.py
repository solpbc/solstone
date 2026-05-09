# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

from solstone.convey.sol_initiated import copy

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
    Path("solstone/convey/static/sol_initiated_constants.js"),
    Path("tests/test_sol_initiated_constants_locked.py"),
    Path("docs/design/sol_initiated_chat_lode1.md"),
    Path("docs/design/sol_initiated_chat_lode2.md"),
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


def test_browser_constants_match_python_contract() -> None:
    text = Path("solstone/convey/static/sol_initiated_constants.js").read_text(
        encoding="utf-8"
    )

    assert _js_constant(text, "KIND_SOL_CHAT_REQUEST") == copy.KIND_SOL_CHAT_REQUEST
    assert (
        _js_constant(text, "KIND_SOL_CHAT_REQUEST_SUPERSEDED")
        == copy.KIND_SOL_CHAT_REQUEST_SUPERSEDED
    )
    assert _js_constant(text, "KIND_OWNER_CHAT_OPEN") == copy.KIND_OWNER_CHAT_OPEN
    assert (
        _js_constant(text, "KIND_OWNER_CHAT_DISMISSED")
        == copy.KIND_OWNER_CHAT_DISMISSED
    )
    assert 'SURFACE_CONVEY: "convey"' in text
    assert _js_constant(text, "SURFACE_CONVEY") == copy.SURFACE_CONVEY
    assert (
        _js_constant(text, "SOL_PINGED_OFFLINE_TOOLTIP")
        == copy.SOL_PINGED_OFFLINE_TOOLTIP
    )


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
            and path.suffix in {".py", ".md", ".js"}
        )
    return sorted(files)


def _js_constant(text: str, name: str) -> str:
    match = re.search(rf'{name}: "([^"]*)"', text)
    assert match is not None, name
    return match.group(1).encode("utf-8").decode("unicode_escape")


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
