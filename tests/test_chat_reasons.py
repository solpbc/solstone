# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

from solstone.convey.chat_reasons import (
    CHAT_REASONS,
    DISPLAY_NAMES,
    ChatReason,
    ChatReasonAction,
    render_chat_reason,
)

EXPECTED_CODES = {
    "provider_key_invalid",
    "provider_quota_exceeded",
    "network_unreachable",
    "provider_response_invalid",
    "provider_unavailable",
    "chat_timeout",
    "unknown",
}


def _extract_frozen_object(text: str, name: str) -> dict:
    marker = f"const {name} = Object.freeze("
    start = text.index(marker) + len(marker)
    depth = 0
    in_string = False
    escaped = False
    object_start = None

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if object_start is None:
                object_start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and object_start is not None:
                return json.loads(text[object_start : index + 1])

    raise AssertionError(f"Could not extract {name}")


def test_registry_shape():
    assert set(CHAT_REASONS) == EXPECTED_CODES
    assert len(CHAT_REASONS) == 7
    for code, reason in CHAT_REASONS.items():
        assert isinstance(reason, ChatReason)
        assert reason.code == code
        assert reason.template
        assert reason.action is None or isinstance(reason.action, ChatReasonAction)


def test_render_known_codes():
    for code, reason in CHAT_REASONS.items():
        rendered = render_chat_reason(code, "google")
        assert rendered["code"] == code
        assert rendered["message"]
        if "{provider}" in reason.template:
            assert "Gemini" in rendered["message"]
        if code == "provider_key_invalid":
            assert rendered["action"] == {
                "label": "Open Settings",
                "href": "/app/settings/#providers",
            }
        else:
            assert rendered["action"] is None


def test_render_display_names():
    for slug, display in DISPLAY_NAMES.items():
        rendered = render_chat_reason("provider_key_invalid", slug)
        assert display in rendered["message"]


def test_render_unknown_code():
    assert render_chat_reason("not_a_real_code", "") == {
        "code": "not_a_real_code",
        "message": "not_a_real_code",
        "action": None,
    }


def test_render_empty_provider():
    assert render_chat_reason("network_unreachable", "") == {
        "code": "network_unreachable",
        "message": "I couldn't reach the network",
        "action": None,
    }


def test_js_parity():
    js_path = Path("solstone/convey/static/chat_reasons.js")
    text = js_path.read_text(encoding="utf-8")
    js_reasons = _extract_frozen_object(text, "CHAT_REASONS")
    js_display_names = _extract_frozen_object(text, "CHAT_REASON_DISPLAY_NAMES")

    py_reasons = {
        code: {
            "template": reason.template,
            "action": (
                {"label": reason.action.label, "href": reason.action.href}
                if reason.action
                else None
            ),
        }
        for code, reason in CHAT_REASONS.items()
    }

    assert js_reasons == py_reasons
    assert js_display_names == DISPLAY_NAMES

    for code, reason in CHAT_REASONS.items():
        for provider, display in DISPLAY_NAMES.items():
            expected = reason.template.replace("{provider}", display)
            assert render_chat_reason(code, provider)["message"] == expected

    removed_constants = ("CHAT_" + "TROUBLE_REASON", "CHAT_" + "WATCHDOG_REASON")
    assert all(name not in text for name in removed_constants)
