#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Layer-hygiene lint.

Low-bar static check for the invariants in ``docs/coding-standards.md`` §
"Layer Hygiene" (L1, L2, L3, L6, L7). Warns when code inside infrastructure
modules (``think/indexer/``, ``think/importers/``, ``think/search/``,
``think/graph/``) or inside a read-verb CLI handler (a function in
``apps/*/call.py`` whose name contains a read verb such as ``load``, ``show``,
``check``, ``validate``, ``find``, ``list``, ``scan``, ``get``) performs a
direct write (``atomic_write``, ``json.dump``, ``.write_text``,
``open(..., "w")``, ``unlink``, ``rmtree``) against a path under
``journal/entities/``, ``journal/facets/``, or ``journal/observations``.

By design this is a grep-level check with known false-positive surface. Known
audit-tracked violations are allowlisted below with a TODO and an audit
reference. An allowlist entry is expected to disappear once its bundle ships —
see ``vpe/workspace/solstone-layer-violations-audit.md`` in the sol pbc
internal extro repo for the canonical list (V1-V14).

Exit codes:
  0 — no un-tracked violations
  1 — new violations found outside the allowlist
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Module families scrutinized as "infrastructure" per L1/L6/L7.
INFRASTRUCTURE_SCOPES: tuple[str, ...] = (
    "think/indexer",
    "think/importers",
    "think/search",
    "think/graph",
)

# Direct-write operations. Indirect writes via helper methods (e.g.
# ``checklist.save()``) are out of scope by design — the audit notes that
# indirect writes are not reachable by grep.
WRITE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\batomic_write\s*\("), "atomic_write"),
    (re.compile(r"\bjson\.dump\s*\("), "json.dump"),
    (re.compile(r"\.write_text\s*\("), ".write_text"),
    (re.compile(r"""\bopen\s*\([^)]*["']w[+b]?["']"""), 'open(..., "w")'),
    (re.compile(r"\bos\.unlink\s*\("), "os.unlink"),
    (re.compile(r"\.unlink\s*\(\s*(?:missing_ok|\))"), ".unlink()"),
    (re.compile(r"\b(?:shutil\.)?rmtree\s*\("), "rmtree"),
)

# Strings / identifiers that indicate the write target sits under one of the
# protected domains. The window-based proximity check below uses these to
# decide whether a flagged write is on a domain path.
TARGET_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"journal/entities\b"),
    re.compile(r"journal/facets\b"),
    re.compile(r"journal/observations"),
    re.compile(r'["\']entities["\']'),
    re.compile(r'["\']facets["\']'),
    re.compile(r'["\']observations'),
    re.compile(
        r"\b(?:entity|facet|observation|observations?)_(?:path|dir|file|json)\b"
    ),
)

# Read verbs per docs/coding-standards.md § L3. Match against any
# underscore-split segment of the function name.
READ_VERBS: frozenset[str] = frozenset(
    {
        "load",
        "get",
        "read",
        "scan",
        "list",
        "show",
        "find",
        "match",
        "resolve",
        "query",
        "lookup",
        "status",
        "check",
        "validate",
        "discover",
        "format",
        "render",
        "extract",
        "parse",
        "view",
        "inspect",
        "info",
        "describe",
        "search",
    }
)

# Temporary, file-scoped exceptions for known layer-hygiene violations.
# Keep this empty by default; add entries only with a tracking identifier
# and remove them in the same bundle that fixes the violation.
ALLOWLIST: dict[str, str] = {}

CONTEXT_WINDOW = 8  # lines above and below each write to search for paths


def tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def in_infrastructure_scope(rel: Path) -> bool:
    path_str = rel.as_posix()
    return any(path_str.startswith(scope + "/") for scope in INFRASTRUCTURE_SCOPES)


def is_call_py(rel: Path) -> bool:
    parts = rel.parts
    return len(parts) >= 3 and parts[0] == "apps" and parts[-1] == "call.py"


def has_target_path_nearby(lines: list[str], idx: int) -> bool:
    start = max(0, idx - CONTEXT_WINDOW)
    end = min(len(lines), idx + CONTEXT_WINDOW + 1)
    window = "\n".join(lines[start:end])
    return any(p.search(window) for p in TARGET_PATH_PATTERNS)


def scan_lines(lines: list[str]) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        for pat, label in WRITE_PATTERNS:
            if pat.search(line) and has_target_path_nearby(lines, idx):
                findings.append((idx + 1, label))
                break
    return findings


def has_read_verb(name: str) -> bool:
    base = name.lstrip("_")
    return any(part in READ_VERBS for part in base.split("_") if part)


def check_call_py(rel: Path, source: str) -> list[tuple[int, str, str]]:
    """Flag writes inside read-verb function bodies.

    Returns a list of ``(line_no, write_label, function_name)`` tuples.
    """
    try:
        tree = ast.parse(source, filename=str(rel))
    except SyntaxError:
        return []

    findings: list[tuple[int, str, str]] = []
    src_lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not has_read_verb(node.name):
            continue
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        body_lines = src_lines[start : end + 1]
        sub_findings = scan_lines(body_lines)
        for local_line, label in sub_findings:
            findings.append((start + local_line, label, node.name))
    return findings


def main() -> int:
    new: list[str] = []
    tracked: list[str] = []

    for rel in sorted(tracked_python_files()):
        abs_path = ROOT / rel
        if not abs_path.is_file():
            continue
        try:
            source = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        rel_str = rel.as_posix()
        issues: list[str] = []

        if in_infrastructure_scope(rel):
            for line_no, label in scan_lines(source.splitlines()):
                issues.append(
                    f"{rel_str}:{line_no}: {label} "
                    f"on journal-domain path (infrastructure scope)"
                )

        if is_call_py(rel):
            for line_no, label, func_name in check_call_py(rel, source):
                issues.append(
                    f"{rel_str}:{line_no}: {label} in read-verb handler '{func_name}()'"
                )

        if not issues:
            continue

        audit_ref = ALLOWLIST.get(rel_str)
        for issue in issues:
            if audit_ref:
                tracked.append(f"{issue}  [tracked: {audit_ref}]")
            else:
                new.append(issue)

    if tracked:
        print("layer-hygiene: known violations (tracked, expected to disappear):")
        for line in tracked:
            print(f"  {line}")
        print()

    if new:
        print("layer-hygiene: NEW violations:", file=sys.stderr)
        for line in new:
            print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "See docs/coding-standards.md § Layer Hygiene (L1/L2/L3/L6/L7).",
            file=sys.stderr,
        )
        return 1

    print("layer-hygiene: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
