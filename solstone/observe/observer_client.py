# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, NamedTuple

import requests

from solstone.think.utils import get_config, get_journal, read_service_port

logger = logging.getLogger(__name__)
HOST = socket.gethostname()
PLATFORM = platform.system().lower()
RETRY_BACKOFF = [1, 5, 15]
MAX_RETRIES = 3
UPLOAD_TIMEOUT = 300
EVENT_TIMEOUT = 30


class UploadResult(NamedTuple):
    success: bool
    duplicate: bool = False


def cleanup_draft(draft_dir: str) -> None:
    """Remove all files in a draft directory and delete the directory."""
    try:
        for name in os.listdir(draft_dir):
            fp = os.path.join(draft_dir, name)
            if os.path.isfile(fp):
                os.remove(fp)
        os.rmdir(draft_dir)
    except OSError:
        pass


def finalize_draft(draft_dir: str, segment_key: str) -> str | None:
    """Rename a draft directory to its final segment name.

    Preserves captured data locally when observer upload fails, so the
    think pipeline can process it later.

    Args:
        draft_dir: Path to the draft directory (e.g. .../HHMMSS_draft/)
        segment_key: Final segment name (e.g. "091551_300")

    Returns:
        Path to the finalized directory, or None on failure.
    """
    final_dir = os.path.join(os.path.dirname(draft_dir), segment_key)
    try:
        os.rename(draft_dir, final_dir)
        logger.info(f"Finalized draft locally: {final_dir}")
        return final_dir
    except OSError as e:
        logger.error(f"Failed to finalize draft {draft_dir} -> {final_dir}: {e}")
        return None


class ObserverClient:
    """HTTP client for uploading observer segments to the ingest server."""

    def __init__(
        self,
        stream: str,
        host: str = HOST,
        platform_name: str = PLATFORM,
    ):
        config = get_config()
        observer_cfg = config.get("observe", {}).get("observer", {})
        self._url = observer_cfg.get("url", "").rstrip("/")
        if not self._url:
            # Discover local convey port from health directory
            port = read_service_port("convey")
            if port:
                self._url = f"http://localhost:{port}"
                logger.info(f"Discovered convey at port {port}")
            else:
                logger.warning("No convey port found in health directory")
                self._url = ""
        self._key = observer_cfg.get("key")
        self._auto_register = observer_cfg.get("auto_register", True)
        self._name = observer_cfg.get("name") or stream
        self._stream = stream
        self._host = host
        self._platform = platform_name
        self._revoked = False
        self._session = requests.Session()

    def _persist_key(self, key: str) -> None:
        journal = get_journal()
        config_path = Path(journal) / "config" / "journal.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(
                    f"Cannot read {config_path}: {e} — skipping key persistence"
                )
                return

        config.setdefault("observe", {}).setdefault("observer", {})["key"] = key

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        os.chmod(config_path, 0o600)

        logger.info(f"Persisted observer key to {config_path}")

    def _ensure_registered(self) -> None:
        if self._key:
            return
        if not self._url:
            return
        if not self._auto_register:
            logger.error(
                "No observer key configured and auto_register disabled. "
                "Set observe.observer.key in journal config or enable auto_register."
            )
            return

        url = f"{self._url}/app/observer/api/create"
        for attempt, delay in enumerate(RETRY_BACKOFF):
            try:
                resp = self._session.post(
                    url,
                    json={"name": self._name},
                    timeout=EVENT_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._key = data["key"]
                    self._persist_key(self._key)
                    logger.info(
                        f"Auto-registered as '{self._name}' (key: {self._key[:8]}...)"
                    )
                    return
                elif resp.status_code == 403:
                    self._revoked = True
                    logger.error("Registration rejected (403)")
                    return
                else:
                    logger.warning(
                        f"Registration attempt {attempt + 1} failed: {resp.status_code}"
                    )
            except requests.RequestException as e:
                logger.warning(f"Registration attempt {attempt + 1} failed: {e}")
            if attempt < len(RETRY_BACKOFF) - 1:
                time.sleep(delay)
        logger.error(f"Registration failed after {MAX_RETRIES} attempts")

    def upload_segment(
        self,
        day: str,
        segment: str,
        files: list[Path],
        meta: dict[str, Any] | None = None,
    ) -> UploadResult:
        if self._revoked:
            logger.warning("Client revoked, skipping upload")
            return UploadResult(False)

        self._ensure_registered()
        if not self._key:
            return UploadResult(False)

        url = f"{self._url}/app/observer/ingest"
        for attempt, delay in enumerate(RETRY_BACKOFF):
            file_handles = []
            files_data = []
            try:
                for path in files:
                    if not path.exists():
                        logger.warning(f"File not found, skipping: {path}")
                        continue
                    fh = open(path, "rb")
                    file_handles.append(fh)
                    files_data.append(
                        ("files", (path.name, fh, "application/octet-stream"))
                    )

                if not files_data:
                    logger.error("No valid files to upload")
                    return UploadResult(False)

                data: dict[str, Any] = {
                    "day": day,
                    "segment": segment,
                }
                if not meta or "host" not in meta:
                    data["host"] = self._host
                if not meta or "platform" not in meta:
                    data["platform"] = self._platform
                if meta:
                    data["meta"] = json.dumps(meta)

                headers = {}
                if self._key:
                    headers["Authorization"] = f"Bearer {self._key}"
                    logger.debug(
                        f"Sending Authorization header: Bearer {self._key[:8]}..."
                    )

                response = self._session.post(
                    url,
                    data=data,
                    files=files_data,
                    headers=headers,
                    timeout=UPLOAD_TIMEOUT,
                )

                if response.status_code == 200:
                    resp_data = response.json()
                    is_duplicate = resp_data.get("status") == "duplicate"
                    return UploadResult(True, duplicate=is_duplicate)
                if response.status_code == 403:
                    self._revoked = True
                    logger.error("Upload rejected (403)")
                    return UploadResult(False)

                logger.warning(
                    f"Upload attempt {attempt + 1} failed: "
                    f"{response.status_code} {response.text}"
                )
            except requests.RequestException as e:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
            finally:
                for fh in file_handles:
                    try:
                        fh.close()
                    except Exception:
                        pass

            if attempt < len(RETRY_BACKOFF) - 1:
                time.sleep(delay)

        logger.error(f"Upload failed after {MAX_RETRIES} attempts: {day}/{segment}")
        return UploadResult(False)

    def relay_event(self, tract: str, event: str, **fields: Any) -> bool:
        if self._revoked:
            return False

        self._ensure_registered()
        if not self._key:
            return False

        url = f"{self._url}/app/observer/ingest/{self._key}/event"
        payload = {"tract": tract, "event": event, **fields}
        try:
            resp = self._session.post(url, json=payload, timeout=EVENT_TIMEOUT)
            if resp.status_code == 200:
                return True
            if resp.status_code == 403:
                self._revoked = True
                logger.error("Event relay rejected (403)")
                return False
            logger.warning(f"Event relay failed: {resp.status_code} {resp.text}")
            return False
        except requests.RequestException as e:
            logger.debug(f"Event relay failed: {e}")
            return False

    def stop(self) -> None:
        self._session.close()
