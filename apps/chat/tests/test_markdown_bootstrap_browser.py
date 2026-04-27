# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from werkzeug.serving import make_server

from convey import create_app
from convey.chat_stream import append_chat_event

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright
PlaywrightError = playwright_sync.Error


def _ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute).timestamp() * 1000)


def _copytree_tracked(src: Path, dst: Path) -> None:
    result = subprocess.run(
        ["git", "ls-files", "."],
        cwd=str(src),
        capture_output=True,
        text=True,
        check=True,
    )
    for rel in result.stdout.splitlines():
        if not rel:
            continue
        src_file = src / rel
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.is_symlink():
            os.symlink(os.readlink(src_file), dst_file)
        else:
            shutil.copy2(src_file, dst_file)


@contextlib.contextmanager
def _isolated_app_env(journal: Path) -> Iterator[Path]:
    previous = os.environ.get("SOLSTONE_JOURNAL")
    os.environ["SOLSTONE_JOURNAL"] = str(journal.resolve())
    try:
        yield journal
    finally:
        if previous is None:
            os.environ.pop("SOLSTONE_JOURNAL", None)
        else:
            os.environ["SOLSTONE_JOURNAL"] = previous


@contextlib.contextmanager
def _running_chat_server(journal: Path) -> Iterator[str]:
    app = create_app(str(journal))
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _write_localhost_auth_config(journal: Path) -> None:
    config_path = journal / "config" / "journal.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
    config.setdefault("setup", {})["completed_at"] = 1
    config.setdefault("convey", {})["trust_localhost"] = True
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _launch_chromium(playwright: Any) -> Any:
    try:
        return playwright.chromium.launch()
    except PlaywrightError as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "playwright install" in message:
            pytest.skip("Playwright chromium is not installed")
        raise


def test_ssr_talent_markdown_bootstraps_to_rendered_dom(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _copytree_tracked(Path("tests/fixtures/journal").resolve(), journal)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    _write_localhost_auth_config(journal)

    day = "20990102"
    append_chat_event(
        "talent_finished",
        ts=_ms(2099, 1, 2, 9, 3),
        use_id="use-md-browser-1",
        name="exec",
        summary="**done**",
    )

    with _isolated_app_env(journal):
        with _running_chat_server(journal) as base_url:
            with sync_playwright() as playwright:
                browser = _launch_chromium(playwright)
                try:
                    page = browser.new_page()
                    page.goto(f"{base_url}/app/chat/{day}", wait_until="domcontentloaded")

                    assert (
                        page.locator(".chat-talent-card-detail--markdown strong").count()
                        > 0
                    )
                    assert page.locator('[data-markdown="1"]').count() == 0
                finally:
                    browser.close()
