# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Single canonical architecture and distribution target mapping table."""

from dataclasses import dataclass
from typing import Optional

from solstone_platform.refusals import SCHEMA_INVALID, Refusal


@dataclass(frozen=True)
class TargetMapping:
    arch: str             # platform.json arch key: "x86_64" | "aarch64"
    native_target: str    # journal/desktop/rust target: "linux-x86_64" | "linux-aarch64"
    deb_arch: str         # deb package arch: "amd64" | "arm64"
    rpm_arch: str         # rpm package arch: "x86_64" | "aarch64"
    musl_target: str      # tmux target triple: "x86_64-unknown-linux-musl" | "aarch64-unknown-linux-musl"


# The authoritative mapping table
TARGET_MAPPINGS: dict[str, TargetMapping] = {
    "x86_64": TargetMapping(
        arch="x86_64",
        native_target="linux-x86_64",
        deb_arch="amd64",
        rpm_arch="x86_64",
        musl_target="x86_64-unknown-linux-musl",
    ),
    "aarch64": TargetMapping(
        arch="aarch64",
        native_target="linux-aarch64",
        deb_arch="arm64",
        rpm_arch="aarch64",
        musl_target="aarch64-unknown-linux-musl",
    ),
}

NATIVE_TO_ARCH: dict[str, str] = {
    m.native_target: m.arch for m in TARGET_MAPPINGS.values()
}

MUSL_TO_ARCH: dict[str, str] = {
    m.musl_target: m.arch for m in TARGET_MAPPINGS.values()
}


def get_target_mapping(arch: str) -> TargetMapping:
    if arch not in TARGET_MAPPINGS:
        raise Refusal(SCHEMA_INVALID, f"unsupported architecture '{arch}', must be one of {sorted(TARGET_MAPPINGS.keys())}")
    return TARGET_MAPPINGS[arch]


def arch_for_native_target(native_target: str) -> Optional[str]:
    return NATIVE_TO_ARCH.get(native_target)


def arch_for_musl_target(musl_target: str) -> Optional[str]:
    return MUSL_TO_ARCH.get(musl_target)
