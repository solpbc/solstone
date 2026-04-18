# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

import frontmatter

SENSE_PATH = Path(__file__).resolve().parents[1] / "talent" / "sense.md"


def _role_section() -> str:
    content = frontmatter.load(SENSE_PATH).content
    start = content.index("#### role")
    end = content.index("#### source", start)
    return content[start:end]


def test_sense_role_section_contains_contamination_guard():
    role_section = _role_section()

    assert "tool or product names visible on screen" in role_section
    assert "`source: screen`" in role_section
    assert "`role: mentioned`" in role_section
    assert "Google Meet" in role_section
    assert "Zoom" in role_section
    assert "quoted or referenced in transcripts" in role_section
    assert "actively speaking as participants" in role_section


def test_sense_role_section_has_screen_and_mentioned_guidance_for_tools_and_apps():
    role_section = _role_section()

    assert "screen" in role_section
    assert "mentioned" in role_section
    assert "tool" in role_section
    assert "Video-conference app names" in role_section
