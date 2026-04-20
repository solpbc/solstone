# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Push device storage."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from think.entities.core import atomic_write
from think.utils import get_journal

logger = logging.getLogger("solstone.push.devices")


def _devices_path() -> Path:
    return Path(get_journal()) / "config" / "push_devices.json"


def _empty_store() -> dict[str, list[dict[str, Any]]]:
    return {"devices": []}


def _validate_store(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("push device store must be a JSON object")
    devices = payload.get("devices")
    if not isinstance(devices, list):
        raise ValueError("push device store must contain a devices list")
    normalized: list[dict[str, Any]] = []
    for device in devices:
        if not isinstance(device, dict):
            raise ValueError("push device rows must be JSON objects")
        token = str(device.get("token") or "").strip()
        bundle_id = str(device.get("bundle_id") or "").strip()
        environment = str(device.get("environment") or "").strip()
        platform = str(device.get("platform") or "").strip()
        registered_at = device.get("registered_at")
        if (
            not token
            or not bundle_id
            or not environment
            or not platform
            or not isinstance(registered_at, (int, float))
        ):
            raise ValueError("push device row missing required fields")
        normalized.append(
            {
                "token": token,
                "bundle_id": bundle_id,
                "environment": environment,
                "platform": platform,
                "registered_at": int(registered_at),
            }
        )
    return normalized


def _read_store() -> list[dict[str, Any]]:
    path = _devices_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _validate_store(payload)
    except Exception as exc:
        logger.warning("push device store unreadable path=%s error=%s", path, exc)
        return []


def _write_store(devices: list[dict[str, Any]]) -> None:
    payload = json.dumps({"devices": devices}, indent=2, ensure_ascii=False) + "\n"
    atomic_write(_devices_path(), payload, prefix=".push_devices_")


def load_devices() -> list[dict[str, Any]]:
    return _read_store()


def register_device(
    *, token: str, bundle_id: str, environment: str, platform: str
) -> int:
    devices = load_devices()
    registered_at = int(time.time())
    updated = False
    for device in devices:
        if device["token"] != token:
            continue
        device.update(
            {
                "bundle_id": bundle_id,
                "environment": environment,
                "platform": platform,
                "registered_at": registered_at,
            }
        )
        updated = True
        break
    if not updated:
        devices.append(
            {
                "token": token,
                "bundle_id": bundle_id,
                "environment": environment,
                "platform": platform,
                "registered_at": registered_at,
            }
        )
    _write_store(devices)
    return len(devices)


def remove_device(token: str) -> bool:
    devices = load_devices()
    remaining = [device for device in devices if device["token"] != token]
    if len(remaining) == len(devices):
        return False
    _write_store(remaining)
    return True


def mask_token(token: str) -> str:
    return "..." + str(token or "")[-4:]


def status_view(device: dict[str, Any]) -> dict[str, Any]:
    registered_at = int(device["registered_at"])
    registered_at_label = (
        datetime.fromtimestamp(registered_at, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return {
        "token_suffix": mask_token(device.get("token", "")),
        "bundle_id": device["bundle_id"],
        "environment": device["environment"],
        "platform": device["platform"],
        "registered_at": registered_at_label,
    }


__all__ = [
    "load_devices",
    "mask_token",
    "register_device",
    "remove_device",
    "status_view",
]
