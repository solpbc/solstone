# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple
from urllib.parse import quote

import requests

from solstone.apps.observer.routes import OBSERVER_CALLOSUM_SSE_ROUTE
from solstone.think.utils import get_config, get_journal, read_service_port

logger = logging.getLogger(__name__)
HOST = socket.gethostname()
PLATFORM = platform.system().lower()
RETRY_BACKOFF = [1, 5, 15]
MAX_RETRIES = 3
UPLOAD_TIMEOUT = 300
EVENT_TIMEOUT = 30
CALLOSUM_RECONNECT_BACKOFF = [1, 2, 4, 8, 16, 30]


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
        self._callosum_thread: threading.Thread | None = None
        self._callosum_stop = threading.Event()
        self._callosum_response: requests.Response | None = None
        self._callosum_error: Exception | None = None

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

    def subscribe_callosum(self, callback: Callable[[dict], None]) -> None:
        if self._callosum_thread is not None and self._callosum_thread.is_alive():
            raise RuntimeError("subscribe_callosum already active")

        self._callosum_stop.clear()
        self._callosum_error = None
        self._callosum_thread = threading.Thread(
            target=self._callosum_loop,
            args=(callback,),
            daemon=True,
        )
        self._callosum_thread.start()

    def _callosum_loop(self, callback: Callable[[dict], None]) -> None:
        if self._revoked:
            return

        self._ensure_registered()
        if not self._key or not self._url:
            return

        path = OBSERVER_CALLOSUM_SSE_ROUTE.replace("<key>", quote(self._key, safe=""))
        url = f"{self._url}{path}"
        headers = {"Authorization": f"Bearer {self._key}"}
        backoff_index = 0

        while not self._callosum_stop.is_set():
            response: requests.Response | None = None
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(EVENT_TIMEOUT, None),
                )
                self._callosum_response = response

                if response.status_code == 200:
                    backoff_index = 0
                    self._consume_callosum_response(response, callback)
                elif response.status_code in {401, 403}:
                    self._revoked = True
                    self._callosum_error = RuntimeError(
                        f"Callosum subscription rejected ({response.status_code})"
                    )
                    logger.warning(
                        "Callosum subscription rejected (%s)", response.status_code
                    )
                    return
                else:
                    self._callosum_error = RuntimeError(
                        f"Callosum subscription failed ({response.status_code})"
                    )
                    logger.debug(
                        "Callosum subscription failed: %s %s",
                        response.status_code,
                        response.text,
                    )
            except requests.RequestException as e:
                self._callosum_error = e
                logger.debug(f"Callosum subscription transport failed: {e}")
            except Exception as e:
                self._callosum_error = e
                if self._callosum_stop.is_set():
                    logger.debug(f"Callosum subscription stopped: {e}")
                else:
                    logger.debug(f"Callosum subscription failed: {e}", exc_info=True)
            finally:
                if self._callosum_response is response:
                    self._callosum_response = None
                if response is not None:
                    self._close_callosum_response(response)

            if self._callosum_stop.is_set():
                return

            delay = CALLOSUM_RECONNECT_BACKOFF[
                min(backoff_index, len(CALLOSUM_RECONNECT_BACKOFF) - 1)
            ]
            if self._callosum_stop.wait(delay):
                return
            if backoff_index < len(CALLOSUM_RECONNECT_BACKOFF) - 1:
                backoff_index += 1

    def _consume_callosum_response(
        self,
        response: requests.Response,
        callback: Callable[[dict], None],
    ) -> None:
        data_lines: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if self._callosum_stop.is_set():
                return
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line == "":
                self._dispatch_callosum_frame(data_lines, callback)
                data_lines = []
            elif line.startswith(":"):
                continue
            elif line.startswith("data:"):
                data = line[5:]
                if data.startswith(" "):
                    data = data[1:]
                data_lines.append(data)

        self._dispatch_callosum_frame(data_lines, callback)

    def _dispatch_callosum_frame(
        self,
        data_lines: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        if not data_lines:
            return

        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid callosum SSE payload: {e}")
            return

        try:
            callback(payload)
        except Exception:
            logger.exception("Callosum subscription callback failed")

    def _close_callosum_response(self, response: requests.Response) -> None:
        self._shutdown_callosum_response_socket(response)
        try:
            response.close()
        except Exception as e:
            logger.debug(f"Callosum response close failed: {e}")

    def _shutdown_callosum_response_socket(self, response: requests.Response) -> None:
        try:
            raw = getattr(response, "raw", None)
            fp = getattr(raw, "_fp", None)
            socket_fp = getattr(fp, "fp", None)
            socket_raw = getattr(socket_fp, "raw", None)
            sock = getattr(socket_raw, "_sock", None)
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

    def stop(self) -> None:
        self._callosum_stop.set()
        if self._callosum_response is not None:
            self._close_callosum_response(self._callosum_response)
        if (
            self._callosum_thread is not None
            and self._callosum_thread.is_alive()
            and self._callosum_thread is not threading.current_thread()
        ):
            self._callosum_thread.join(timeout=5.0)
        self._session.close()
