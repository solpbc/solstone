# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""LAN interface endpoint watcher for link pairing surfaces."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import socket
from collections.abc import Iterable

from .local_endpoints import LocalEndpoint

log = logging.getLogger("link.interface_watcher")

LINK_DIRECT_PORT = 7657
_EXCLUDED_INTERFACE_PREFIXES = (
    "lo",
    "docker",
    "br-",
    "vbox",
    "vmnet",
    "tun",
    "tap",
    "utun",
)
_LAN_V4_NETWORKS = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
)
_ULA_V6_NETWORK = ipaddress.IPv6Network("fc00::/7")


class InterfaceWatcher:
    def __init__(self, *, poll_interval: float = 1.5) -> None:
        self._poll_interval = poll_interval
        self._endpoints: tuple[LocalEndpoint, ...] = ()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start watching interfaces.

        Linux netlink can replace this later; polling satisfies the current
        1-2s freshness requirement without brittle binary parsing.
        """
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._poll_loop(),
            name="link-interface-watcher",
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        task = self._task
        self._task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._endpoints = ()

    def snapshot(self) -> list[LocalEndpoint]:
        return list(self._endpoints)

    def _update_snapshot(self, raw: Iterable[tuple[str, str]]) -> None:
        endpoints: list[LocalEndpoint] = []
        for ifname, ip_str in raw:
            if _is_excluded_interface(ifname):
                continue
            normalized = _normalize_ip(ip_str)
            scope = _classify(normalized)
            if scope is None:
                continue
            endpoints.append(
                LocalEndpoint(ip=normalized, port=LINK_DIRECT_PORT, scope=scope)
            )
        next_snapshot = tuple(sorted(endpoints, key=lambda ep: (ep.scope, ep.ip)))
        if next_snapshot == self._endpoints:
            return
        self._endpoints = next_snapshot
        scope_counts: dict[str, int] = {}
        for ep in next_snapshot:
            scope_counts[ep.scope] = scope_counts.get(ep.scope, 0) + 1
        log.info(
            "local endpoints changed count=%d scopes=%s",
            len(next_snapshot),
            scope_counts,
        )

    async def _poll_loop(self) -> None:
        import psutil

        while True:
            try:
                raw: set[tuple[str, str]] = set()
                for ifname, addrs in psutil.net_if_addrs().items():
                    if _is_excluded_interface(ifname):
                        continue
                    for addr in addrs:
                        if addr.family not in (socket.AF_INET, socket.AF_INET6):
                            continue
                        raw.add((ifname, _normalize_ip(addr.address)))
                self._update_snapshot(raw)
            except Exception:
                log.exception("interface polling failed")
            await asyncio.sleep(self._poll_interval)


def _is_excluded_interface(name: str) -> bool:
    return name.startswith(_EXCLUDED_INTERFACE_PREFIXES)


def _normalize_ip(ip_str: str) -> str:
    return ip_str.split("%", 1)[0].split("/", 1)[0]


def _classify(ip_str: str) -> str | None:
    try:
        ip_addr = ipaddress.ip_address(_normalize_ip(ip_str))
    except ValueError:
        return None
    if ip_addr.is_loopback or ip_addr.is_link_local or ip_addr.is_multicast:
        return None
    if isinstance(ip_addr, ipaddress.IPv4Address):
        if any(ip_addr in network for network in _LAN_V4_NETWORKS):
            return "lan"
        return None
    if ip_addr in _ULA_V6_NETWORK:
        return "ula"
    return None


_WATCHER: InterfaceWatcher | None = None


def set_interface_watcher(watcher: InterfaceWatcher | None) -> None:
    global _WATCHER
    _WATCHER = watcher


def get_interface_watcher() -> InterfaceWatcher | None:
    return _WATCHER


__all__ = [
    "InterfaceWatcher",
    "LINK_DIRECT_PORT",
    "_classify",
    "_is_excluded_interface",
    "get_interface_watcher",
    "set_interface_watcher",
]
