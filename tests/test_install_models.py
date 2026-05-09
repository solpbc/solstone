# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from solstone.think import install_models


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_model_files(base_dir: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        target = base_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ok")


@pytest.mark.parametrize(
    ("flag_value", "env_value", "os_name", "arch", "expected"),
    [
        ("cpu", None, "linux", "x86_64", "cpu"),
        ("cpu", "cuda", "linux", "x86_64", "cpu"),
        ("auto", "cpu", "linux", "x86_64", "cpu"),
        ("auto", "cuda", "linux", "x86_64", "cuda"),
        ("auto", None, "darwin", "arm64", "coreml"),
        ("auto", None, "windows", "amd64", None),
    ],
)
def test_resolve_variant_precedence(
    monkeypatch: pytest.MonkeyPatch,
    flag_value: str,
    env_value: str | None,
    os_name: str,
    arch: str,
    expected: str | None,
):
    monkeypatch.setattr(install_models, "_detect_linux_variant", lambda: "cpu")

    assert (
        install_models._resolve_variant(flag_value, env_value, os_name, arch)
        == expected
    )


def test_resolve_variant_autodetects_linux_gpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(install_models, "_detect_linux_variant", lambda: "cuda")

    assert install_models._resolve_variant("auto", None, "linux", "x86_64") == "cuda"


def test_resolve_variant_rejects_invalid_env_value():
    with pytest.raises(SystemExit, match="invalid PARAKEET_ONNX_VARIANT='bogus'"):
        install_models._resolve_variant("auto", "bogus", "linux", "x86_64")


def test_resolve_variant_rejects_incompatible_explicit_variant():
    with pytest.raises(SystemExit, match="variant 'coreml' not supported on linux"):
        install_models._resolve_variant("coreml", None, "linux", "x86_64")
    with pytest.raises(SystemExit, match="variant 'cpu' not supported on darwin"):
        install_models._resolve_variant("cpu", None, "darwin", "arm64")


def test_verify_bundled_assets_returns_when_hashes_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    wespeaker = tmp_path / "wespeaker.onnx"
    pyannote = tmp_path / "pyannote.onnx"
    wespeaker.write_bytes(b"wespeaker")
    pyannote.write_bytes(b"pyannote")
    monkeypatch.setattr(install_models, "WESPEAKER_MODEL_PATH", wespeaker)
    monkeypatch.setattr(install_models, "WESPEAKER_MODEL_SHA256", _sha256(b"wespeaker"))
    monkeypatch.setattr(install_models, "PYANNOTE_OVERLAP_MODEL_PATH", pyannote)
    monkeypatch.setattr(
        install_models,
        "PYANNOTE_OVERLAP_MODEL_SHA256",
        _sha256(b"pyannote"),
    )

    install_models._verify_bundled_assets()


def test_verify_bundled_assets_reports_mutated_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    wespeaker = tmp_path / "wespeaker.onnx"
    pyannote = tmp_path / "pyannote.onnx"
    wespeaker.write_bytes(b"mutated")
    pyannote.write_bytes(b"pyannote")
    expected = _sha256(b"original")
    actual = _sha256(b"mutated")
    monkeypatch.setattr(install_models, "WESPEAKER_MODEL_PATH", wespeaker)
    monkeypatch.setattr(install_models, "WESPEAKER_MODEL_SHA256", expected)
    monkeypatch.setattr(install_models, "PYANNOTE_OVERLAP_MODEL_PATH", pyannote)
    monkeypatch.setattr(
        install_models,
        "PYANNOTE_OVERLAP_MODEL_SHA256",
        _sha256(b"pyannote"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        install_models._verify_bundled_assets()

    message = str(exc_info.value)
    assert f"bundled asset SHA mismatch: {wespeaker}" in message
    assert f"expected: {expected}" in message
    assert f"actual:   {actual}" in message


def test_verify_returns_true_when_files_at_fluidaudio_sibling(tmp_path: Path):
    cache_dir = tmp_path / "models"
    repo_dir = tmp_path / install_models.MAC_FLUIDAUDIO_REPO_NAME
    _write_model_files(repo_dir, install_models.MAC_MODEL_FILES)

    assert install_models._verify_mac_cache(cache_dir) is True


def test_verify_returns_false_when_sibling_empty(tmp_path: Path):
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    (tmp_path / install_models.MAC_FLUIDAUDIO_REPO_NAME).mkdir()

    assert install_models._verify_mac_cache(cache_dir) is False


def test_verify_returns_false_when_files_at_literal_path(tmp_path: Path):
    cache_dir = tmp_path / "models"
    _write_model_files(cache_dir, install_models.MAC_MODEL_FILES)

    assert install_models._verify_mac_cache(cache_dir) is False


def test_helper_path_env_override_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fake = tmp_path / "custom" / "parakeet-helper"
    monkeypatch.setenv(install_models.HELPER_ENV_KEY, str(fake))
    monkeypatch.setattr(install_models, "_package_root", lambda: tmp_path)
    assert install_models._helper_path() == fake.expanduser().resolve()


def test_helper_path_prefers_bundled_bin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv(install_models.HELPER_ENV_KEY, raising=False)
    monkeypatch.setattr(install_models, "_package_root", lambda: tmp_path)
    bundled = (
        tmp_path
        / "observe"
        / "transcribe"
        / "parakeet_helper"
        / "_bin"
        / "parakeet-helper"
    )
    bundled.parent.mkdir(parents=True)
    bundled.write_text("")
    assert install_models._helper_path() == bundled


def test_helper_path_falls_back_to_swift_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv(install_models.HELPER_ENV_KEY, raising=False)
    monkeypatch.setattr(install_models, "_package_root", lambda: tmp_path)
    expected = (
        tmp_path
        / "observe"
        / "transcribe"
        / "parakeet_helper"
        / ".build"
        / "release"
        / "parakeet-helper"
    )
    assert install_models._helper_path() == expected


def _prepare_check_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    sentinel_path = tmp_path / "sentinel.json"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["sol install-models", "--check"])
    monkeypatch.delenv(install_models.PARAKEET_ONNX_VARIANT_ENV, raising=False)
    monkeypatch.setattr(install_models, "_platform_info", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(install_models, "_detect_linux_variant", lambda: "cpu")
    monkeypatch.setattr(install_models, "_verify_bundled_assets", lambda: None)
    monkeypatch.setattr(install_models, "_sentinel_path", lambda variant: sentinel_path)
    monkeypatch.setattr(install_models, "_cache_dir", lambda variant: cache_dir)
    return sentinel_path, cache_dir


def test_main_check_missing_sentinel_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    _prepare_check_main(monkeypatch, tmp_path)

    assert install_models.main() == 1
    assert "parakeet check failed: sentinel not ready" in capsys.readouterr().err


def test_main_check_ready_cache_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    sentinel_path, cache_dir = _prepare_check_main(monkeypatch, tmp_path)
    cache_dir.mkdir()
    install_models._write_sentinel(
        sentinel_path,
        install_models._build_payload("linux", "x86_64", "cpu", cache_dir),
    )
    monkeypatch.setattr(install_models, "_verify_linux_cache", lambda path: True)

    assert install_models.main() == 0
    assert f"model ready: {cache_dir}" in capsys.readouterr().out


def test_run_mac_helper_soft_fails_on_packaged_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    cache_dir = tmp_path / "cache"
    sentinel_path = tmp_path / "sentinel.json"
    missing_helper = tmp_path / "missing" / "parakeet-helper"

    monkeypatch.delenv(install_models.HELPER_ENV_KEY, raising=False)
    monkeypatch.setattr(install_models, "_helper_path", lambda: missing_helper)
    monkeypatch.setattr(install_models, "is_packaged_install", lambda: True)
    monkeypatch.setattr(install_models, "_sentinel_path", lambda variant: sentinel_path)
    monkeypatch.setattr(install_models, "_cache_dir", lambda variant: cache_dir)

    assert install_models._run_mac_helper(cache_dir) is None
    stderr = capsys.readouterr().err
    assert "Apple Silicon Macs running macOS 14" in stderr
    assert "Intel Mac" in stderr
    assert "source checkout" in stderr

    assert install_models._install_models("darwin", "arm64", "coreml") == 0


def test_fetch_linux_model_soft_fails_on_packaged_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    cache_dir = tmp_path / "cache"
    sentinel_path = tmp_path / "sentinel.json"

    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    monkeypatch.setattr(install_models, "is_packaged_install", lambda: True)
    monkeypatch.setattr(install_models, "_sentinel_path", lambda variant: sentinel_path)
    monkeypatch.setattr(install_models, "_cache_dir", lambda variant: cache_dir)

    assert install_models._fetch_linux_model(cache_dir) is False
    stderr = capsys.readouterr().err
    assert "packaged installs on Linux don't include the parakeet-onnx" in stderr
    assert "Whisper, Gemini, OpenAI" in stderr

    assert install_models._install_models("linux", "x86_64", "cpu") == 0
    assert not sentinel_path.exists()


def test_fetch_linux_model_raises_on_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    cache_dir = tmp_path / "cache"
    sentinel_path = tmp_path / "sentinel.json"

    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    monkeypatch.setattr(install_models, "is_packaged_install", lambda: False)
    monkeypatch.setattr(install_models, "_sentinel_path", lambda variant: sentinel_path)
    monkeypatch.setattr(install_models, "_cache_dir", lambda variant: cache_dir)

    assert install_models._install_models("linux", "x86_64", "cpu") == 1
    assert "parakeet install failed" in capsys.readouterr().err
