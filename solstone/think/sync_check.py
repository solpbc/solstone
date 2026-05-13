# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Detect concurrent writers for a synced solstone journal.

The detector uses explicit heartbeat files, not path-prefix heuristics, because
two machines may mount the same journal at different local paths. A journal is
the shared dataset; a machine id identifies the physical writer. This module
does not merge, coordinate, or repair multi-device state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from solstone.think.utils import get_journal

DEFAULT_INTERVAL_SECONDS: float = 15.0
FRESH_WINDOW_MULTIPLIER: int = 4
SCHEMA_VERSION: int = 1

_MACHINE_ID: str | None = None
_DARWIN_UUID_RE = re.compile(r'"IOPlatformUUID"\s*=\s*"([^"]+)"')
_HOSTNAME_RE = re.compile(r"[^a-z0-9_-]+")


@dataclass(frozen=True)
class ForeignWriter:
    path: Path
    hostname: str
    journal_path: str
    pid: int | None
    machine_id: str
    machine_id_prefix: str
    solstone_version: str
    wall_time: str
    mtime: float
    sha256: str
    is_fresh: bool
    changed_since_snapshot: bool
    appeared_since_snapshot: bool
    is_live: bool
    parse_error: str | None

    @property
    def display_hostname(self) -> str:
        return self.hostname or "(unknown)"


@dataclass(frozen=True)
class SyncCheckSnapshot:
    files: dict[str, tuple[float, str]]


@dataclass(frozen=True)
class SyncCheckResult:
    journal_path: Path
    sync_dir: Path
    snapshot: SyncCheckSnapshot
    self_machine_id: str
    self_hostname: str
    self_heartbeat_path: Path
    foreign_writers: tuple[ForeignWriter, ...]
    live_foreign_writers: tuple[ForeignWriter, ...]
    history_foreign_writers: tuple[ForeignWriter, ...]
    primary_conflict: ForeignWriter | None

    @property
    def is_conflict(self) -> bool:
        return bool(self.live_foreign_writers)


def get_machine_id() -> str:
    global _MACHINE_ID
    if _MACHINE_ID is not None:
        return _MACHINE_ID

    if sys.platform.startswith("linux"):
        try:
            machine_id = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
            if machine_id:
                _MACHINE_ID = machine_id
                return machine_id
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            match = _DARWIN_UUID_RE.search(completed.stdout)
            if match and match.group(1).strip():
                _MACHINE_ID = match.group(1).strip()
                return _MACHINE_ID
        except (OSError, subprocess.SubprocessError):
            pass

    return ""


def get_self_hostname_sanitized() -> str:
    hostname = socket.gethostname().lower()
    hostname = _HOSTNAME_RE.sub("-", hostname).strip("-")
    return hostname or "unknown-host"


def _self_heartbeat_filename() -> str:
    return f"{get_self_hostname_sanitized()}.check"


def _self_heartbeat_path(journal: Path | None = None) -> Path:
    return _sync_dir(journal) / _self_heartbeat_filename()


def write_self_heartbeat(journal: Path | None = None) -> Path:
    journal_path = _journal_path(journal)
    sync_dir = _sync_dir(journal_path)
    target = sync_dir / _self_heartbeat_filename()
    now = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "machine_id": get_machine_id(),
        "hostname": get_self_hostname_sanitized(),
        "pid": os.getpid(),
        "wall_time": now,
        "solstone_version": _solstone_version(),
        "interval_seconds": DEFAULT_INTERVAL_SECONDS,
        "journal_path": str(journal_path),
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    sync_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, content)
    return target


def clear_self_heartbeat(journal: Path | None = None) -> None:
    try:
        _self_heartbeat_path(journal).unlink(missing_ok=True)
    except OSError:
        pass


def check_journal_sync(
    previous: SyncCheckSnapshot | None = None,
    *,
    journal: Path | None = None,
    now: float | None = None,
) -> SyncCheckResult:
    journal_path = _journal_path(journal)
    sync_dir = _sync_dir(journal_path)
    self_machine_id = get_machine_id()
    self_hostname = get_self_hostname_sanitized()
    self_path = sync_dir / _self_heartbeat_filename()
    current_snapshot: dict[str, tuple[float, str]] = {}
    foreign_writers: list[ForeignWriter] = []
    now_ts = time.time() if now is None else now

    if not sync_dir.is_dir():
        snapshot = SyncCheckSnapshot(current_snapshot)
        return SyncCheckResult(
            journal_path=journal_path,
            sync_dir=sync_dir,
            snapshot=snapshot,
            self_machine_id=self_machine_id,
            self_hostname=self_hostname,
            self_heartbeat_path=self_path,
            foreign_writers=(),
            live_foreign_writers=(),
            history_foreign_writers=(),
            primary_conflict=None,
        )

    for path in sorted(sync_dir.glob("*.check")):
        try:
            stat_result = path.stat()
            data = path.read_bytes()
        except OSError:
            continue

        mtime = stat_result.st_mtime
        digest = hashlib.sha256(data).hexdigest()
        current_snapshot[path.name] = (mtime, digest)

        if path.name == self_path.name:
            continue

        previous_tuple = previous.files.get(path.name) if previous else None
        changed_since_snapshot = previous_tuple is not None and previous_tuple != (
            mtime,
            digest,
        )
        appeared_since_snapshot = previous is not None and previous_tuple is None
        is_fresh = (now_ts - mtime) <= (
            FRESH_WINDOW_MULTIPLIER * DEFAULT_INTERVAL_SECONDS
        )

        payload: dict[str, Any] = {}
        parse_error = None
        try:
            parsed = json.loads(data.decode("utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
            else:
                parse_error = "invalid JSON"
        except (UnicodeDecodeError, json.JSONDecodeError):
            parse_error = "invalid JSON"

        if parse_error:
            hostname = ""
            journal_value = ""
            pid = None
            machine_id = ""
            solstone_version = ""
            wall_time = ""
            is_live = is_fresh
        else:
            hostname = str(payload.get("hostname") or "")
            journal_value = str(payload.get("journal_path") or "")
            pid = _coerce_pid(payload.get("pid"))
            machine_id = str(payload.get("machine_id") or "")
            solstone_version = str(payload.get("solstone_version") or "")
            wall_time = str(payload.get("wall_time") or "")
            is_live = is_fresh or changed_since_snapshot or appeared_since_snapshot

        foreign_writers.append(
            ForeignWriter(
                path=path,
                hostname=hostname,
                journal_path=journal_value,
                pid=pid,
                machine_id=machine_id,
                machine_id_prefix=machine_id[:8],
                solstone_version=solstone_version,
                wall_time=wall_time,
                mtime=mtime,
                sha256=digest,
                is_fresh=is_fresh,
                changed_since_snapshot=changed_since_snapshot,
                appeared_since_snapshot=appeared_since_snapshot,
                is_live=is_live,
                parse_error=parse_error,
            )
        )

    live = tuple(writer for writer in foreign_writers if writer.is_live)
    history = tuple(writer for writer in foreign_writers if not writer.is_live)
    primary = max(live, key=lambda writer: writer.mtime) if live else None

    return SyncCheckResult(
        journal_path=journal_path,
        sync_dir=sync_dir,
        snapshot=SyncCheckSnapshot(current_snapshot),
        self_machine_id=self_machine_id,
        self_hostname=self_hostname,
        self_heartbeat_path=self_path,
        foreign_writers=tuple(foreign_writers),
        live_foreign_writers=live,
        history_foreign_writers=history,
        primary_conflict=primary,
    )


def format_conflict_message(result: SyncCheckResult) -> str:
    primary = result.primary_conflict
    if primary is None:
        return ""

    self_prefix = result.self_machine_id[:8] if result.self_machine_id else "(unknown)"
    primary_prefix = _display_machine_prefix(primary)
    primary_age = _format_age(time.time() - primary.mtime)
    lines = [
        "Refusing to start - another solstone service is active on this journal.",
        "",
        f"Journal: {result.journal_path}",
        f"This device: {result.self_hostname} (machine {self_prefix}...)",
        (
            "Active service: "
            f"{primary.display_hostname} (machine {primary_prefix}..., "
            f"last seen {primary_age})"
        ),
        f"  PID: {_display_pid(primary.pid)}",
        f"  Journal path on that device: {primary.journal_path or '(unknown)'}",
    ]

    others = [writer for writer in result.live_foreign_writers if writer != primary]
    if others:
        lines.extend(["", "Other active instances:"])
        for writer in sorted(others, key=lambda item: item.mtime, reverse=True):
            age = _format_age(time.time() - writer.mtime)
            lines.append(
                "  • "
                f"{writer.display_hostname} "
                f"(machine {_display_machine_prefix(writer)}..., last seen {age})"
            )

    lines.extend(
        [
            "",
            "Use one service per journal.",
            "Multiple observers attached to a single service are fine.",
            f"To continue here, stop solstone on {primary.display_hostname} first.",
        ]
    )
    return "\n".join(lines)


def format_doctor_report(result: SyncCheckResult) -> str:
    self_prefix = result.self_machine_id[:8] if result.self_machine_id else "(unknown)"
    clean = f"this device only ({result.self_hostname}, machine {self_prefix}...)"
    if result.is_conflict:
        return format_conflict_message(result)
    if not result.foreign_writers:
        return clean

    history = max(result.history_foreign_writers, key=lambda writer: writer.mtime)
    return (
        f"{clean}\n"
        "  last foreign writer: "
        f"{history.display_hostname} "
        f"(machine {_display_machine_prefix(history)}..., "
        f"{_format_age(time.time() - history.mtime)})"
    )


def _sync_dir(journal: Path | None) -> Path:
    return _journal_path(journal) / "health" / "sync"


def _journal_path(journal: Path | None) -> Path:
    if journal is None:
        return Path(get_journal()).resolve()
    return Path(journal).resolve()


def _atomic_write_text(path: Path, content: str, *, mode: int | None = None) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
        os.replace(tmp_path, path)
        if mode is not None:
            os.chmod(path, mode)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _coerce_pid(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _solstone_version() -> str:
    try:
        return version("solstone")
    except Exception:
        return ""


def _format_age(seconds: float) -> str:
    if seconds < 0:
        return "in the future"
    total = int(seconds)
    if total < 60:
        return f"{total}s ago"
    minutes, remaining_seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s ago"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes}m ago"


def _display_machine_prefix(writer: ForeignWriter) -> str:
    if writer.machine_id:
        return writer.machine_id[:8]
    return "(unknown)"


def _display_pid(pid: int | None) -> str:
    return str(pid) if pid is not None else "(unknown)"
