# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Locked copy for convey onboarding, settings, and pairing surfaces."""

from __future__ import annotations

INIT_PASSWORD_HINT = "protects your solstone web interface when you allow network access. minimum 8 characters. you can reset it anytime by using a terminal and running the command: <code>sol password set</code>. password is not required when accessing solstone from the same system."
CONVEY_RELOAD_HINT = "reload to try again."
CONVEY_ACTION_TRY_AGAIN = "Try again"  # generic load-failure retry button
CONVEY_ACTION_RELOAD = "Reload"  # full-surface reload affordance
CONVEY_ACTION_RECONNECT = "Reconnect"  # transport/socket reconnection affordance
CONVEY_ACTION_RESTART = "Restart"  # service/observer restart affordance
SETTINGS_SECURITY_DESC = (
    "network access and password protection for the convey web interface."
)
SETTINGS_NETWORK_MODE_LABEL = "network access"
SETTINGS_NETWORK_MODE_OFF = "localhost only"
SETTINGS_NETWORK_MODE_ON = "on"
SETTINGS_NETWORK_DESC_OFF = (
    "convey is reachable only from this machine. no password is required."
)
SETTINGS_NETWORK_DESC_ON = "convey is reachable from other devices on the local network. password is required for non-localhost clients."
SETTINGS_LAN_URL_LABEL = "local network url"
SETTINGS_NETWORK_BUTTON_ENABLE = "allow network access"
SETTINGS_NETWORK_BUTTON_DISABLE = "restrict to localhost only"
SETTINGS_NETWORK_NEEDS_PASSWORD = "set a password below first."
SETTINGS_NETWORK_RESTARTING = "restarting convey…"
SETTINGS_PASSWORD_HINT = "protects the web interface when network access is on. not required for localhost-only mode."
OBSERVER_CALLOSUM_LIVE_LABEL = "live"
PAIRING_LOCALHOST_BANNER_TITLE = "convey is in localhost-only mode"
PAIRING_LOCALHOST_BANNER_BODY_1 = (
    "paired devices won't be able to connect until network access is enabled."
)
PAIRING_LOCALHOST_BANNER_BODY_2 = "enabling network access should only be done on trusted networks and requires a password."
PAIRING_LOCALHOST_BANNER_ACTION = "enable network access in settings →"
PAIRING_NO_LAN_BANNER_TITLE = "couldn't detect a local network address"
PAIRING_NO_LAN_BANNER_BODY = "the QR code below uses localhost, which paired devices can't reach. set a host URL manually with: sol call settings convey host-url <url>"


__all__ = [
    "CONVEY_ACTION_TRY_AGAIN",
    "CONVEY_ACTION_RELOAD",
    "CONVEY_ACTION_RECONNECT",
    "CONVEY_ACTION_RESTART",
    "CONVEY_RELOAD_HINT",
    "INIT_PASSWORD_HINT",
    "OBSERVER_CALLOSUM_LIVE_LABEL",
    "PAIRING_LOCALHOST_BANNER_ACTION",
    "PAIRING_LOCALHOST_BANNER_BODY_1",
    "PAIRING_LOCALHOST_BANNER_BODY_2",
    "PAIRING_LOCALHOST_BANNER_TITLE",
    "PAIRING_NO_LAN_BANNER_BODY",
    "PAIRING_NO_LAN_BANNER_TITLE",
    "SETTINGS_LAN_URL_LABEL",
    "SETTINGS_NETWORK_BUTTON_DISABLE",
    "SETTINGS_NETWORK_BUTTON_ENABLE",
    "SETTINGS_NETWORK_DESC_OFF",
    "SETTINGS_NETWORK_DESC_ON",
    "SETTINGS_NETWORK_MODE_LABEL",
    "SETTINGS_NETWORK_MODE_OFF",
    "SETTINGS_NETWORK_MODE_ON",
    "SETTINGS_NETWORK_NEEDS_PASSWORD",
    "SETTINGS_NETWORK_RESTARTING",
    "SETTINGS_PASSWORD_HINT",
    "SETTINGS_SECURITY_DESC",
]
