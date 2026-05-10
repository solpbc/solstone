# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from solstone.think.link.interface_watcher import (
    InterfaceWatcher,
    _classify,
    _is_excluded_interface,
)


def test_classify_lan_ipv4() -> None:
    assert _classify("10.0.0.1") == "lan"
    assert _classify("172.16.5.5") == "lan"
    assert _classify("172.31.255.254") == "lan"
    assert _classify("192.168.1.1") == "lan"


def test_classify_ula_ipv6() -> None:
    assert _classify("fc00::1") == "ula"
    assert _classify("fd00::1") == "ula"


def test_classify_excludes_non_lan_addresses() -> None:
    assert _classify("8.8.8.8") is None
    assert _classify("172.32.0.1") is None
    assert _classify("127.0.0.1") is None
    assert _classify("169.254.1.1") is None
    assert _classify("fe80::1") is None
    assert _classify("fe80::1%eth0") is None
    assert _classify("::1") is None
    assert _classify("ff02::1") is None
    assert _classify("2001:db8::1") is None


def test_is_excluded_interface() -> None:
    for name in ("lo", "lo0", "docker0", "br-abc", "vboxnet0", "vmnet8"):
        assert _is_excluded_interface(name)
    for name in ("tun0", "tap0", "utun3"):
        assert _is_excluded_interface(name)
    for name in ("eth0", "en0", "wlan0", "wlp3s0", "enp0s31f6"):
        assert not _is_excluded_interface(name)


def test_update_snapshot_filters_and_sorts() -> None:
    watcher = InterfaceWatcher()

    watcher._update_snapshot(
        [
            ("eth0", "192.168.1.10"),
            ("docker0", "172.17.0.1"),
            ("en0", "fd00::1"),
            ("lo", "127.0.0.1"),
            ("wlan0", "10.0.0.2"),
            ("eth1", "8.8.8.8"),
            ("en1", "fe80::1%en1"),
        ]
    )

    endpoints = watcher.snapshot()
    assert [(ep.scope, ep.ip, ep.port) for ep in endpoints] == [
        ("lan", "10.0.0.2", 7657),
        ("lan", "192.168.1.10", 7657),
        ("ula", "fd00::1", 7657),
    ]
    assert endpoints == sorted(endpoints, key=lambda ep: (ep.scope, ep.ip))
