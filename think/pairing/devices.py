# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Paired-device storage."""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from think.entities.core import atomic_write
from think.utils import get_journal

logger = logging.getLogger(__name__)


class Device(TypedDict):
    id: str
    name: str
    platform: str
    public_key: str
    session_key_hash: str
    bundle_id: str
    app_version: str
    paired_at: str
    last_seen_at: str | None


def _devices_path() -> Path:
    return Path(get_journal()) / "config" / "paired_devices.json"


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _validate_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device timestamp fields must be strings or null")
    return value


def _validate_store(payload: Any) -> list[Device]:
    if not isinstance(payload, dict):
        raise ValueError("paired device store must be a JSON object")
    devices = payload.get("devices")
    if not isinstance(devices, list):
        raise ValueError("paired device store must contain a devices list")
    normalized: list[Device] = []
    for device in devices:
        if not isinstance(device, dict):
            raise ValueError("paired device rows must be JSON objects")
        normalized.append(
            Device(
                id=_require_field(device, "id"),
                name=_require_field(device, "name"),
                platform=_require_field(device, "platform"),
                public_key=_require_field(device, "public_key"),
                session_key_hash=_require_field(device, "session_key_hash"),
                bundle_id=_require_field(device, "bundle_id"),
                app_version=_require_field(device, "app_version"),
                paired_at=_require_field(device, "paired_at"),
                last_seen_at=_validate_timestamp(device.get("last_seen_at")),
            )
        )
    return normalized


def _require_field(device: dict[str, Any], field: str) -> str:
    value = _clean_str(device.get(field))
    if not value:
        raise ValueError(f"paired device row missing required field: {field}")
    return value


def _read_store() -> list[Device]:
    path = _devices_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _validate_store(payload)
    except Exception as exc:
        logger.warning("paired device store unreadable path=%s error=%s", path, exc)
        return []


def _write_store(devices: list[Device]) -> None:
    payload = json.dumps({"devices": devices}, indent=2, ensure_ascii=False) + "\n"
    atomic_write(_devices_path(), payload, prefix=".paired_devices_")


def load_devices() -> list[Device]:
    return _read_store()


def find_device_by_id(device_id: str) -> Device | None:
    target = _clean_str(device_id)
    if not target:
        return None
    for device in load_devices():
        if device["id"] == target:
            return device
    return None


def find_device_by_session_key_hash(session_key_hash: str) -> Device | None:
    target = _clean_str(session_key_hash)
    if not target:
        return None
    for device in load_devices():
        if device["session_key_hash"] == target:
            return device
    return None


def register_device(
    *,
    name: str,
    platform: str,
    public_key: str,
    session_key_hash: str,
    bundle_id: str,
    app_version: str,
    paired_at: str | None = None,
) -> Device:
    devices = load_devices()
    row: Device = Device(
        id=f"dev_{secrets.token_urlsafe(16)}",
        name=_clean_str(name),
        platform=_clean_str(platform),
        public_key=_clean_str(public_key),
        session_key_hash=_clean_str(session_key_hash),
        bundle_id=_clean_str(bundle_id),
        app_version=_clean_str(app_version),
        paired_at=_clean_str(paired_at) or _utc_now_iso(),
        last_seen_at=None,
    )
    for index, device in enumerate(devices):
        if device["public_key"] != row["public_key"]:
            continue
        row["id"] = device["id"]
        devices[index] = row
        _write_store(devices)
        return row
    devices.append(row)
    _write_store(devices)
    return row


def touch_last_seen(device_id: str, *, last_seen_at: str | None = None) -> bool:
    target = _clean_str(device_id)
    if not target:
        return False
    devices = load_devices()
    timestamp = _clean_str(last_seen_at) or _utc_now_iso()
    for device in devices:
        if device["id"] != target:
            continue
        device["last_seen_at"] = timestamp
        _write_store(devices)
        return True
    return False


def remove_device(device_id: str) -> bool:
    target = _clean_str(device_id)
    if not target:
        return False
    devices = load_devices()
    remaining = [device for device in devices if device["id"] != target]
    if len(remaining) == len(devices):
        return False
    _write_store(remaining)
    return True


def status_view(device: Device) -> dict[str, Any]:
    return {
        "id": device["id"],
        "name": device["name"],
        "platform": device["platform"],
        "paired_at": device["paired_at"],
        "last_seen_at": device["last_seen_at"],
    }


__all__ = [
    "Device",
    "find_device_by_id",
    "find_device_by_session_key_hash",
    "load_devices",
    "register_device",
    "remove_device",
    "status_view",
    "touch_last_seen",
]
