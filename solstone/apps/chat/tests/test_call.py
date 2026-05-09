# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json

import pytest
from typer.testing import CliRunner

import solstone.apps.chat.call as chat_call
from solstone.convey.sol_initiated.copy import (
    CATEGORIES,
    THROTTLE_CATEGORY_CAP,
    THROTTLE_CATEGORY_SELF_MUTE,
    THROTTLE_DAILY_CAP,
    THROTTLE_MUTE_WINDOW,
    THROTTLE_RATE_FLOOR,
)
from solstone.convey.sol_initiated.start import StartChatResult

runner = CliRunner()


def _args() -> list[str]:
    return [
        "start",
        "--summary",
        "summary",
        "--category",
        CATEGORIES[0],
        "--dedupe",
        "k",
        "--since-ts",
        "1",
        "--trigger-talent",
        "reflection",
    ]


def test_start_command_prints_result(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_call,
        "start_chat",
        lambda **kwargs: StartChatResult(True, False, None, "req"),
    )

    result = runner.invoke(chat_call.app, [*_args(), "--message", "body"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "written": True,
        "deduped": False,
        "throttled": None,
        "request_id": "req",
    }


def test_start_command_prints_deduped(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_call,
        "start_chat",
        lambda **kwargs: StartChatResult(False, True, None, None),
    )

    result = runner.invoke(chat_call.app, _args())

    assert result.exit_code == 0
    assert json.loads(result.output)["deduped"] is True


@pytest.mark.parametrize(
    "reason",
    [
        THROTTLE_MUTE_WINDOW,
        THROTTLE_RATE_FLOOR,
        THROTTLE_CATEGORY_SELF_MUTE,
        THROTTLE_CATEGORY_CAP,
        THROTTLE_DAILY_CAP,
    ],
)
def test_start_command_prints_throttle(monkeypatch, reason) -> None:
    monkeypatch.setattr(
        chat_call,
        "start_chat",
        lambda **kwargs: StartChatResult(False, False, reason, None),
    )

    result = runner.invoke(chat_call.app, _args())

    assert result.exit_code == 0
    assert json.loads(result.output)["throttled"] == reason


def test_start_command_validation_error(monkeypatch) -> None:
    def fail(**kwargs):
        raise ValueError("bad input")

    monkeypatch.setattr(chat_call, "start_chat", fail)

    result = runner.invoke(chat_call.app, _args())

    assert result.exit_code == 1
    assert "bad input" in result.output
