# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import re
from pathlib import Path

from solstone.convey import copy


def test_console_copy_matches_browser_mirror() -> None:
    text = Path("solstone/convey/static/convey_copy.js").read_text(encoding="utf-8")

    for python_name in copy.__all__:
        if not python_name.startswith("CONVEY_CONSOLE_"):
            continue
        js_name = python_name.removeprefix("CONVEY_")
        assert _js_constant(text, js_name) == getattr(copy, python_name)


def _js_constant(text: str, name: str) -> str:
    match = re.search(rf'{name}: "((?:[^"\\]|\\.)*)"', text)
    assert match is not None, name
    return json.loads(f'"{match.group(1)}"')
