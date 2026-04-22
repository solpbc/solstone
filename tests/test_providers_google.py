# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import logging


def test_build_generate_config_passes_through_at_cap(caplog):
    from think.providers.google import GEMINI_MAX_OUTPUT_TOKENS, _build_generate_config

    with caplog.at_level(logging.WARNING, logger="think.providers.google"):
        config = _build_generate_config(
            temperature=0.3,
            max_output_tokens=49152,
            system_instruction=None,
            json_output=False,
            thinking_budget=16384,
        )

    warnings = [
        record
        for record in caplog.records
        if record.name == "think.providers.google" and record.levelno == logging.WARNING
    ]
    assert config.max_output_tokens == GEMINI_MAX_OUTPUT_TOKENS
    assert config.thinking_config.thinking_budget == 16384
    assert warnings == []


def test_build_generate_config_clamps_and_warns_once(caplog):
    from think.providers.google import GEMINI_MAX_OUTPUT_TOKENS, _build_generate_config

    with caplog.at_level(logging.WARNING, logger="think.providers.google"):
        config = _build_generate_config(
            temperature=0.3,
            max_output_tokens=49152,
            system_instruction=None,
            json_output=False,
            thinking_budget=24576,
        )

    warnings = [
        record
        for record in caplog.records
        if record.name == "think.providers.google" and record.levelno == logging.WARNING
    ]
    assert config.max_output_tokens <= GEMINI_MAX_OUTPUT_TOKENS
    assert config.max_output_tokens == GEMINI_MAX_OUTPUT_TOKENS
    assert config.thinking_config.thinking_budget == 16384
    assert len(warnings) == 1
    assert "max_output_tokens=49152" in warnings[0].message
    assert "thinking_budget=24576" in warnings[0].message
    assert "clamped_thinking_budget=16384" in warnings[0].message
