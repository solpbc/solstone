# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from think.talents import validate_config


def test_validate_config_allows_cogitate_with_user_instruction_only():
    config = {
        "type": "cogitate",
        "prompt": "",
        "user_instruction": "Digest instructions from talent body.",
    }

    assert validate_config(config) is None


def test_validate_config_rejects_cogitate_when_prompt_and_user_instruction_are_empty():
    config = {
        "type": "cogitate",
        "prompt": "",
        "user_instruction": "",
    }

    assert (
        validate_config(config)
        == "Cogitate talent requires non-empty 'prompt' or 'user_instruction'"
    )


def test_validate_config_generate_branch_is_unchanged():
    config = {
        "type": "generate",
        "prompt": "",
        "user_instruction": "",
        "day": "",
    }

    assert (
        validate_config(config)
        == "Invalid config: must have 'type', 'day', or 'prompt'"
    )
