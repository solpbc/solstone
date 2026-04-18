# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

import frontmatter

SENSE_PATH = Path(__file__).resolve().parents[1] / "talent" / "sense.md"


def _section(text: str, start: str, end: str | None = None) -> str:
    section_start = text.index(start)
    if end is None:
        return text[section_start:]
    section_end = text.index(end, section_start)
    return text[section_start:section_end]


def test_sense_prompt_parses_and_documents_role_and_source():
    post = frontmatter.load(SENSE_PATH)

    assert post.metadata["tier"] == 3

    schema = _section(
        post.content, "## Output Schema", "## Field-by-Field Instructions"
    )
    entities = _section(post.content, "### entities", "### facets")

    assert '"role": "attendee|mentioned"' in schema
    assert '"source": "voice|speaker_label|transcript|screen|other"' in schema
    assert "#### role" in entities
    assert "#### source" in entities


def test_role_and_source_do_not_leak_into_other_sense_sections():
    content = frontmatter.load(SENSE_PATH).content

    sections = [
        _section(content, "### density", "### content_type"),
        _section(content, "### content_type", "### activity_summary"),
        _section(content, "### activity_summary", "### entities"),
        _section(content, "### facets", "### meeting_detected"),
        _section(content, "### meeting_detected", "### speakers"),
        _section(content, "### speakers", "### recommend"),
        _section(content, "### recommend", "### emotional_register"),
        _section(content, "### emotional_register", "## Rules"),
        _section(content, "## Rules"),
    ]

    for section in sections:
        assert "attendee|mentioned" not in section
        assert "voice|speaker_label|transcript|screen|other" not in section
        assert "#### role" not in section
        assert "#### source" not in section
