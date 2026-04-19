# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

import frontmatter

PARTICIPATION_PATH = Path(__file__).resolve().parents[1] / "talent" / "participation.md"


def test_participation_talent_frontmatter_and_placeholders():
    post = frontmatter.load(PARTICIPATION_PATH)

    assert post.metadata["schedule"] == "activity"
    assert post.metadata["activities"] == ["*"]
    assert post.metadata["tier"] == 3
    assert post.metadata["output"] == "json"
    assert post.metadata["schema"] == "participation.schema.json"
    assert post.metadata["priority"] == 10
    assert post.metadata["load"]["talents"]["sense"] is True

    body = post.content
    assert "$facets" in body
    assert "$activity_context" in body
    assert "$activity_preamble" in body


def test_participation_talent_rules_block_is_present():
    body = frontmatter.load(PARTICIPATION_PATH).content

    expected_rules = [
        "1. Exclude the journal owner.",
        "2. Never mark someone `role: attendee` in a non-meeting activity.",
        "3. No fabrication — if you didn't see them, don't list them.",
        "4. Empty `participation: []` when no entities were involved.",
        "5. Confidence is subjective but should reflect signal strength (`voice` > `speaker_label` > `transcript` > `screen`).",
        '6. Dedupe variants (e.g., "JB" and "John B." → one entry with the richer name).',
    ]

    for rule in expected_rules:
        assert rule in body

    assert "`entity_id` must always be `null`" in body
