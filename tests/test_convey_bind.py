# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from unittest.mock import patch

from convey.cli import _resolve_bind_host


def test_resolve_bind_host_returns_localhost_when_network_access_disabled():
    with patch(
        "think.utils.get_config",
        return_value={"convey": {"allow_network_access": False}},
    ):
        assert _resolve_bind_host() == "127.0.0.1"


def test_resolve_bind_host_returns_localhost_when_key_absent():
    """Defaults to localhost when the convey block has no allow_network_access."""
    with patch("think.utils.get_config", return_value={"convey": {}}):
        assert _resolve_bind_host() == "127.0.0.1"


def test_resolve_bind_host_returns_localhost_when_convey_section_absent():
    """Defaults to localhost when there's no convey block at all."""
    with patch("think.utils.get_config", return_value={}):
        assert _resolve_bind_host() == "127.0.0.1"


def test_resolve_bind_host_returns_all_interfaces_when_network_access_enabled():
    with patch(
        "think.utils.get_config",
        return_value={"convey": {"allow_network_access": True}},
    ):
        assert _resolve_bind_host() == "0.0.0.0"


def test_resolve_bind_host_falls_back_to_localhost_when_config_raises():
    """Defensive default — config read errors must not open the bind to all interfaces."""
    with patch("think.utils.get_config", side_effect=RuntimeError("config unreadable")):
        assert _resolve_bind_host() == "127.0.0.1"
