# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Frozen wire values for LAN-direct link endpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalEndpoint:
    ip: str
    port: int
    scope: str  # "lan" or "ula"


@dataclass(frozen=True)
class LocalEndpointsResponse:
    v: int
    endpoints: tuple[LocalEndpoint, ...]
    ttl_s: int
    generated_at: str


def endpoint_to_dict(ep: LocalEndpoint) -> dict[str, object]:
    return {
        "ip": ep.ip,
        "port": ep.port,
        "scope": ep.scope,
    }


def response_to_dict(r: LocalEndpointsResponse) -> dict[str, object]:
    return {
        "v": r.v,
        "endpoints": [endpoint_to_dict(ep) for ep in r.endpoints],
        "ttl_s": r.ttl_s,
        "generated_at": r.generated_at,
    }


__all__ = [
    "LocalEndpoint",
    "LocalEndpointsResponse",
    "endpoint_to_dict",
    "response_to_dict",
]
