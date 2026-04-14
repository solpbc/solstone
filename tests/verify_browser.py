# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Browser scenario verification using Pinchtab snapshots and screenshots."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


SCENARIOS: list[dict[str, Any]] = [
    # smoke scenarios
    {
        "app": "sol",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/sol/20260304"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "calendar",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/calendar/20260304"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "graph",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/graph"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "speakers",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/speakers/20260304"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "todos",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/todos/20260304"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "tokens",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/tokens/20260304"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "transcripts",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/transcripts/20260304"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "dev",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/dev"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "entities",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/entities"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "health",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/health"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "import",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/import"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "observer",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/observer"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "search",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/search"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "settings",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/settings"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "stats",
        "name": "smoke",
        "steps": [
            {"do": "navigate", "path": "/app/stats"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    # interactive scenarios
    {
        "app": "search",
        "name": "search-flow",
        "steps": [
            {"do": "navigate", "path": "/app/search"},
            {"do": "wait", "ms": 1000},
            {"do": "snapshot"},
            {"do": "find_input", "as": "search_input"},
            {"do": "type", "var": "search_input", "text": "romeo"},
            {"do": "wait", "ms": 1500},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "graph",
        "name": "load",
        "steps": [
            {"do": "navigate", "path": "/app/graph"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "entities",
        "name": "entity-detail",
        "steps": [
            {"do": "navigate", "path": "/app/entities/work/romeo_montague"},
            {"do": "wait", "ms": 1000},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "todos",
        "name": "todo-states",
        "steps": [
            {"do": "evaluate", "expression": "document.cookie='facet=work;path=/'"},
            {"do": "navigate", "path": "/app/todos/20260304"},
            {"do": "wait", "ms": 1200},
            {"do": "screenshot"},
        ],
    },
    {
        "app": "graph",
        "name": "facet-filter",
        "steps": [
            {"do": "evaluate", "expression": "document.cookie='facet=montague;path=/'"},
            {"do": "navigate", "path": "/app/graph"},
            {"do": "wait", "ms": 1200},
            {"do": "screenshot"},
        ],
    },
]


_ERROR_LISTENER_JS = (
    "window.__pt_errors=[];"
    "window.addEventListener('error',e=>window.__pt_errors.push(e.message));"
    "window.onerror=(_,__,___,____,e)=>window.__pt_errors.push(e?.message||'unknown')"
)


def baseline_path(scenario: dict[str, Any]) -> Path:
    return Path("tests/baselines/visual") / scenario["app"] / f"{scenario['name']}.jpg"


class PinchTab:
    """Minimal pinchtab HTTP client with process lifecycle.

    Pinchtab v0.7.x uses a flat API — endpoints are at the root level
    (e.g., /navigate, /screenshot, /snapshot) rather than nested under
    /tabs/<id>/ or /instances/. Chrome is auto-managed by the server.
    """

    def __init__(self, port: int = 19867) -> None:
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self._process: subprocess.Popen | None = None
        self._session = requests.Session()

    def start(self, timeout: int = 30) -> None:
        """Launch pinchtab and wait for health check."""
        env = {
            **os.environ,
            "BRIDGE_PORT": str(self.port),
            "BRIDGE_HEADLESS": "true",
        }
        self._stderr_path = f"/tmp/pinchtab-{self.port}.log"
        self._stderr_file = open(self._stderr_path, "w")
        try:
            self._process = subprocess.Popen(
                ["pinchtab"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
                start_new_session=True,
            )
        except Exception as exc:
            self._stderr_file.close()
            raise RuntimeError("failed to start pinchtab") from exc

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._stderr_file.close()
                try:
                    stderr = Path(self._stderr_path).read_text()
                except Exception:
                    stderr = ""
                raise RuntimeError(
                    f"pinchtab exited with code {self._process.returncode}\n{stderr}"
                )
            try:
                response = self._session.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    health = response.json()
                    if health.get("status") == "ok":
                        return
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("pinchtab failed to start")

    def stop(self) -> None:
        """Terminate pinchtab process and all children."""
        if hasattr(self, "_stderr_file") and self._stderr_file:
            try:
                self._stderr_file.close()
            except Exception:
                pass
        if self._process:
            pid = self._process.pid
            if self._process.poll() is None:
                self._session.close()
                # Kill the entire process group to catch the Go binary child
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        self._process.send_signal(signal.SIGKILL)
                    self._process.wait()
            self._process = None

    def navigate(self, url: str) -> None:
        response = self._session.post(
            f"{self.base_url}/navigate",
            json={"url": url},
            timeout=30,
        )
        response.raise_for_status()

    def screenshot(self) -> bytes:
        response = self._session.get(
            f"{self.base_url}/screenshot",
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return base64.b64decode(payload["base64"])

    def snapshot(self) -> dict:
        response = self._session.get(
            f"{self.base_url}/snapshot",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def text(self) -> str:
        response = self._session.get(
            f"{self.base_url}/text",
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("text", "")
        if isinstance(payload, str):
            return payload
        return ""

    def action(self, kind: str, **kwargs: Any) -> None:
        response = self._session.post(
            f"{self.base_url}/action",
            json={"kind": kind, **kwargs},
            timeout=30,
        )
        response.raise_for_status()

    def evaluate(self, expression: str) -> Any:
        response = self._session.post(
            f"{self.base_url}/evaluate",
            json={"expression": expression},
            timeout=30,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return response.text


def inject_error_listener(pt: PinchTab) -> None:
    pt.evaluate(_ERROR_LISTENER_JS)


def collect_console_errors(pt: PinchTab) -> list[str]:
    result = pt.evaluate("JSON.stringify(window.__pt_errors||[])")
    value = result if isinstance(result, str) else result.get("result", "[]")
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def find_input_ref(snapshot: dict) -> str | None:
    """Find first text input node ref from snapshot."""
    for node in snapshot.get("nodes", []):
        role = str(node.get("role", "")).lower()
        tag = str(node.get("tag", "")).lower()
        if role in ("textbox", "searchbox", "combobox") or tag == "input":
            return node.get("ref")
    return None


def find_ref(snapshot: dict, text: str) -> str | None:
    needle = str(text).lower()
    for node in snapshot.get("nodes", []):
        ref = node.get("ref")
        if not ref:
            continue
        if needle == "":
            return ref
        if (
            needle in str(node.get("name", "")).lower()
            or needle in str(node.get("text", "")).lower()
            or needle in str(node.get("label", "")).lower()
            or needle in str(node.get("value", "")).lower()
        ):
            return ref
    return None


def _resolve_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def run_scenario(
    pt: PinchTab, scenario: dict[str, Any], base_url: str, mode: str
) -> dict[str, Any]:
    """Execute one scenario. Returns {ok, errors, console_errors}."""
    identifier = f"{scenario['app']}/{scenario['name']}"
    errors: list[str] = []
    variables: dict[str, str] = {}
    last_snapshot: dict[str, Any] | None = None
    console_errors: list[str] = []

    logger.info("  %s", identifier)

    try:
        inject_error_listener(pt)
    except Exception:
        pass

    for step in scenario["steps"]:
        action = step["do"]
        try:
            if action == "navigate":
                url = _resolve_url(base_url, step["path"])
                pt.navigate(url)
                time.sleep(0.3)
                try:
                    inject_error_listener(pt)
                except Exception:
                    pass

            elif action == "wait":
                time.sleep(float(step["ms"]) / 1000)

            elif action == "snapshot":
                last_snapshot = pt.snapshot()

            elif action == "screenshot":
                png = pt.screenshot()
                path = baseline_path(scenario)
                if mode == "update":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(png)
                else:
                    if not path.exists():
                        errors.append(f"baseline not found: {path}")
                    # No pixel comparison — baselines are for human review

            elif action == "find":
                if last_snapshot is None:
                    errors.append("find without prior snapshot")
                    continue
                ref = find_ref(last_snapshot, step["text"])
                if ref is None:
                    errors.append(f"find: text not found: {step['text']!r}")
                    continue
                variables[step["as"]] = ref

            elif action == "find_input":
                if last_snapshot is None:
                    errors.append("find_input without prior snapshot")
                    continue
                ref = find_input_ref(last_snapshot)
                if ref is None:
                    errors.append("no text input found in snapshot")
                    continue
                variables[step["as"]] = ref

            elif action == "click":
                ref = step.get("ref") or variables.get(step.get("var", ""))
                if not ref:
                    errors.append(f"click: no ref resolved for {step}")
                    continue
                pt.action("click", ref=ref)

            elif action == "type":
                ref = step.get("ref") or variables.get(step.get("var", ""))
                if not ref:
                    errors.append(f"type: no ref resolved for {step}")
                    continue
                pt.action("type", ref=ref, text=step["text"])

            elif action == "assert_text":
                text = step["text"]
                page_text = pt.text().lower()
                if str(text).lower() not in page_text:
                    errors.append(f"assert_text: '{text}' not found")

            elif action == "evaluate":
                pt.evaluate(step["expression"])

            else:
                errors.append(f"unknown step type: {action}")

        except Exception as exc:
            errors.append(f"step {action} failed: {exc}")

    try:
        console_errors = collect_console_errors(pt)
    except Exception:
        logger.debug("Unable to collect console errors for %s", identifier)

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "console_errors": console_errors,
    }


def run_all(
    pt: PinchTab, base_url: str, mode: str
) -> tuple[list[dict[str, Any]], list[tuple[str, list[str]]]]:
    """Run all scenarios. Returns (results, console_error_pairs)."""
    results: list[dict[str, Any]] = []
    all_console_errors: list[tuple[str, list[str]]] = []
    for scenario in SCENARIOS:
        identifier = f"{scenario['app']}/{scenario['name']}"
        result = run_scenario(pt, scenario, base_url, mode)
        results.append({"scenario": identifier, **result})
        if result["console_errors"]:
            all_console_errors.append((identifier, result["console_errors"]))
    return results, all_console_errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser scenario verification")
    parser.add_argument(
        "command",
        choices=["verify", "update"],
        help="Verify or update baselines",
    )
    parser.add_argument("--base-url", required=True, help="Convey base URL")
    parser.add_argument(
        "--pinchtab-port",
        type=int,
        default=19867,
        help="Pinchtab bridge port",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pt = PinchTab(port=args.pinchtab_port)
    logger.info("Starting pinchtab on port %d...", args.pinchtab_port)
    pt.start()

    try:
        logger.info("Running %d scenarios (%s)...", len(SCENARIOS), args.command)
        results, console_errors = run_all(pt, args.base_url, args.command)

        passed = sum(1 for r in results if r["ok"])
        failed = sum(1 for r in results if not r["ok"])

        if failed:
            logger.info("")
            logger.info("Failures:")
            for result in results:
                if result["ok"]:
                    continue
                for err in result["errors"]:
                    logger.info("  %s: %s", result["scenario"], err)

        if console_errors:
            logger.info("")
            logger.info("JS console errors:")
            for scenario, errors in console_errors:
                for err in errors:
                    logger.info("  %s: %s", scenario, err)

        logger.info("")
        if args.command == "update":
            logger.info("Updated %d scenario baselines.", passed + failed)
        else:
            logger.info("Browser verification: %d passed, %d failed.", passed, failed)

        if failed:
            logger.info("Run 'make update-browser-baselines' to update baselines")
            return 1

        return 0
    finally:
        pt.stop()


if __name__ == "__main__":
    raise SystemExit(main())
