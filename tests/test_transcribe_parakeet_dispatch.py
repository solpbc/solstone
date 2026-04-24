# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import platform
import sys

import pytest

import observe.transcribe.parakeet as parakeet


def test_nemo_module_imports_without_nemo_installed():
    import observe.transcribe._parakeet_nemo  # noqa: F401


def test_dispatch_transcribe_routes_darwin_arm64(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        parakeet._parakeet_coreml,
        "transcribe",
        lambda *args, **kwargs: [{"arm": "coreml"}],
    )
    monkeypatch.setattr(
        parakeet._parakeet_nemo, "transcribe", lambda *args, **kwargs: [{"arm": "nemo"}]
    )

    assert parakeet.transcribe([], 16000, {}) == [{"arm": "coreml"}]


def test_dispatch_transcribe_routes_linux_x86_64(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        parakeet._parakeet_coreml,
        "transcribe",
        lambda *args, **kwargs: [{"arm": "coreml"}],
    )
    monkeypatch.setattr(
        parakeet._parakeet_nemo, "transcribe", lambda *args, **kwargs: [{"arm": "nemo"}]
    )

    assert parakeet.transcribe([], 16000, {}) == [{"arm": "nemo"}]


@pytest.mark.parametrize(
    ("os_name", "arch", "expected"),
    [
        ("darwin", "x86_64", "darwin/x86_64"),
        ("linux", "aarch64", "linux/aarch64"),
        ("win32", "AMD64", "win32/amd64"),
    ],
)
def test_dispatch_transcribe_unsupported_platforms_raise(
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
    arch: str,
    expected: str,
):
    monkeypatch.setattr(sys, "platform", os_name)
    monkeypatch.setattr(platform, "machine", lambda: arch)

    with pytest.raises(RuntimeError, match="Supported platforms"):
        parakeet.transcribe([], 16000, {})

    with pytest.raises(RuntimeError) as exc_info:
        parakeet.transcribe([], 16000, {})

    message = str(exc_info.value)
    assert expected in message
    assert "darwin/arm64" in message
    assert "linux/x86_64" in message


def test_dispatch_get_model_info_routes_linux_x86_64(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        parakeet._parakeet_nemo, "get_model_info", lambda config: {"arm": "nemo"}
    )

    assert parakeet.get_model_info({}) == {"arm": "nemo"}
