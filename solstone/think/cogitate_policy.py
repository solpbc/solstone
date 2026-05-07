# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import tomllib

from solstone.think.utils import get_journal

_BASE_POLICY_PATH = Path(__file__).parent / "policies" / "cogitate.toml"
_POLICY_DIR = Path("/tmp/sol-cogitate-policies")
_READ_TOOL_RULES = (
    ("read_file", '"file_path":"', "strict"),
    ("list_directory", '"dir_path":"', "strict"),
    ("glob", '"(?:pattern|path)":"', "pattern"),
    ("grep_search", '"(?:path|dir_path|include|pattern)":"', "pattern"),
)


def _normalize_day(day: date | str) -> str:
    if isinstance(day, date):
        return day.strftime("%Y%m%d")
    if day:
        return str(day)
    return datetime.now().strftime("%Y%m%d")


def _day_value(day: str) -> date:
    return datetime.strptime(day, "%Y%m%d").date()


def _expand_day_placeholders(value: str, day: str) -> str:
    base_day = _day_value(day)

    def replace(match: re.Match[str]) -> str:
        offset = int(match.group("offset") or 0)
        return (base_day - timedelta(days=offset)).strftime("%Y%m%d")

    return re.sub(r"<day(?:-(?P<offset>\d+))?>", replace, value)


def resolve_read_scope(
    talent_config: dict[str, Any],
    day: date | str,
    span: int = 0,
) -> list[str]:
    day_str = _normalize_day(day)
    configured_scope = talent_config.get("read_scope")
    if configured_scope:
        return [
            _expand_day_placeholders(str(scope), day_str) for scope in configured_scope
        ]

    effective_span = int(talent_config.get("read_scope_span", span or 0) or 0)
    if effective_span <= 0:
        return [f"chronicle/{day_str}"]

    base_day = _day_value(day_str)
    return [
        f"chronicle/{(base_day - timedelta(days=offset)).strftime('%Y%m%d')}"
        for offset in range(effective_span, -1, -1)
    ]


def _is_file_scope(scope: str) -> bool:
    clean = scope.rstrip("/")
    # Dotted basenames are files; ambiguous entries are treated as directories.
    return not scope.endswith("/") and "." in Path(clean).name


def _scope_body(scope: str, suffix: str) -> str:
    clean = scope.strip("/")
    escaped = re.escape(clean)
    return f'{escaped}"' if _is_file_scope(scope) else f"{escaped}{suffix}"


def _allowed_path_regex(scope: list[str], *, mode: str) -> str:
    journal_prefix = re.escape(str(get_journal()).rstrip("/") + "/")
    prefix = f"(?:{journal_prefix}|\\./)?"
    suffix = '(?:/|")' if mode == "strict" else '(?:/|[^"]*")'

    if all(not _is_file_scope(entry) for entry in scope):
        joined = "|".join(re.escape(entry.strip("/")) for entry in scope)
        return f"{prefix}(?:{joined}){suffix}"

    joined = "|".join(_scope_body(entry, suffix) for entry in scope)
    return f"{prefix}(?:{joined})"


def _append_read_rules(base_text: str, scope: list[str]) -> str:
    chunks = [base_text.rstrip(), ""]
    for tool_name, key_pattern, mode in _READ_TOOL_RULES:
        path_pattern = _allowed_path_regex(scope, mode=mode)
        chunks.extend(
            [
                "[[rule]]",
                f'toolName = "{tool_name}"',
                f"argsPattern = '{key_pattern}{path_pattern}'",
                'decision = "allow"',
                "priority = 260",
                "",
                "[[rule]]",
                f'toolName = "{tool_name}"',
                'decision = "deny"',
                "priority = 160",
                "",
            ]
        )
    return "\n".join(chunks)


def _policy_filename(talent_name: str, day: str) -> str:
    talent_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", talent_name).strip("-")
    if not talent_slug:
        talent_slug = "cogitate"
    return f"{talent_slug}-{day}-{os.getpid()}.toml"


def _write_policy(content: str, path: Path) -> Path:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def build_per_task_policy(
    talent_name: str,
    talent_config: dict[str, Any],
    day: date | str,
    span: int,
    base_policy_path: Path = _BASE_POLICY_PATH,
) -> Path:
    day_str = _normalize_day(day)
    scope = resolve_read_scope(talent_config, day_str, span=span)
    base_text = base_policy_path.read_text(encoding="utf-8")
    tomllib.loads(base_text)
    content = _append_read_rules(base_text, scope)

    _POLICY_DIR.mkdir(parents=True, exist_ok=True)
    filename = _policy_filename(talent_name, day_str)
    target = _POLICY_DIR / filename
    for attempt in range(100):
        path = target if attempt == 0 else target.with_stem(f"{target.stem}-{attempt}")
        try:
            return _write_policy(content, path)
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique cogitate policy path for {target}")
