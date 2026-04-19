# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import pytest


def test_append_edit_payload_validation():
    from think.activities import append_edit

    merged = append_edit(
        {"id": "coding_090000_300"},
        actor="cli:update",
        fields=["details"],
        note="updated",
        payload={"foo": "bar"},
    )
    assert merged["edits"][-1]["foo"] == "bar"

    with pytest.raises(
        ValueError, match="payload cannot overwrite canonical edit fields"
    ):
        append_edit(
            {"id": "coding_090000_300"},
            actor="cli:update",
            fields=["details"],
            note="updated",
            payload={"timestamp": "x"},
        )

    with pytest.raises(TypeError, match="payload must be dict\\[str, Any\\]"):
        append_edit(
            {"id": "coding_090000_300"},
            actor="cli:update",
            fields=["details"],
            note="updated",
            payload="not a dict",
        )
