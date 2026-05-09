# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from solstone.observe.transcribe.main import (
    PYANNOTE_OVERLAP_MODEL_PATH,
    PYANNOTE_OVERLAP_MODEL_SHA256,
    WESPEAKER_MODEL_PATH,
    WESPEAKER_MODEL_SHA256,
)
from solstone.observe.transcribe.parakeet_hints import PACKAGED_COREML_HINT
from solstone.observe.utils import compute_file_sha256
from solstone.think.utils import is_packaged_install

BACKEND = "parakeet"
MODEL_VERSION = "v3"
MODEL_ID = "istupakov/parakeet-tdt-0.6b-v3-onnx"
LINUX_LOAD_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"
PARAKEET_ONNX_VARIANT_ENV = "PARAKEET_ONNX_VARIANT"
HELPER_ENV_KEY = "SOLSTONE_PARAKEET_HELPER"
MAC_CACHE_DIR = Path.home() / "Library/Application Support/solstone/parakeet/models"
# FluidAudio's downloadAndLoad(to:) treats the passed dir as the parent and writes into <parent>/<repo-folder>; verified empirically against FluidAudio v0.14.0 v3 on Apple Silicon (helper invocation against a fresh cache dir wrote to <parent>/parakeet-tdt-0.6b-v3/).
MAC_FLUIDAUDIO_REPO_NAME = "parakeet-tdt-0.6b-v3"
MAC_SENTINEL = MAC_CACHE_DIR / ".install-complete"
LINUX_HUB_DIR = Path.home() / ".cache/huggingface/hub"
LINUX_MODEL_DIR = LINUX_HUB_DIR / "models--istupakov--parakeet-tdt-0.6b-v3-onnx"
LINUX_SENTINEL = LINUX_HUB_DIR / ".solstone-install-complete"
LEGACY_NEMO_MODEL_DIR = LINUX_HUB_DIR / "models--nvidia--parakeet-tdt-0.6b-v3"
LINUX_MODEL_FILES = (
    "encoder-model.onnx",
    "decoder_joint-model.onnx",
    "config.json",
    "vocab.txt",
)
LINUX_MIN_CACHE_BYTES = 2_400_000_000
MAC_MODEL_FILES = (
    "Encoder.mlmodelc/weights/weight.bin",
    "Decoder.mlmodelc/weights/weight.bin",
    "JointDecision.mlmodelc/weights/weight.bin",
    "Preprocessor.mlmodelc/weights/weight.bin",
)


def _now_utc() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _quarantine_suffix() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _platform_info() -> tuple[str, str]:
    os_name = "linux" if sys.platform.startswith("linux") else sys.platform
    return os_name, platform.machine().lower()


def _detect_linux_variant() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "cpu"
    if result.returncode == 0 and result.stdout.strip():
        return "cuda"
    return "cpu"


def _resolve_variant(
    flag_value: str,
    env_value: str | None,
    os_name: str,
    arch: str,
) -> str | None:
    if flag_value in {"cpu", "cuda"}:
        if os_name != "linux":
            raise SystemExit(f"variant {flag_value!r} not supported on {os_name}")
        if arch != "x86_64":
            raise SystemExit(
                f"variant {flag_value!r} not supported on {os_name}/{arch}"
            )
        return flag_value

    if flag_value == "coreml":
        if os_name != "darwin":
            raise SystemExit(f"variant 'coreml' not supported on {os_name}")
        if arch != "arm64":
            raise SystemExit(f"variant 'coreml' not supported on {os_name}/{arch}")
        return flag_value

    if os_name == "darwin" and arch == "arm64":
        return "coreml"
    if os_name == "linux" and arch == "x86_64":
        if env_value:
            if env_value not in {"cpu", "cuda"}:
                raise SystemExit(
                    f"invalid {PARAKEET_ONNX_VARIANT_ENV}={env_value!r}; use 'cpu' or 'cuda'"
                )
            return env_value
        return _detect_linux_variant()
    return None


def _sentinel_path(variant: str) -> Path:
    return MAC_SENTINEL if variant == "coreml" else LINUX_SENTINEL


def _cache_dir(variant: str) -> Path:
    return MAC_CACHE_DIR if variant == "coreml" else LINUX_MODEL_DIR


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fixture_audio_path() -> Path:
    return Path(
        str(
            resources.files("solstone.observe.transcribe._fixtures")
            / "parakeet_sample.wav"
        )
    )


def _helper_path() -> Path:
    env_path = os.getenv(HELPER_ENV_KEY)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (
        _package_root()
        / "observe"
        / "transcribe"
        / "parakeet_helper"
        / ".build"
        / "release"
        / "parakeet-helper"
    ).resolve()


def _load_sentinel(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sentinel_ready(
    payload: dict[str, Any] | None,
    os_name: str,
    arch: str,
    variant: str,
) -> Path | None:
    if payload is None:
        return None
    platform_info = payload.get("platform")
    if not isinstance(platform_info, dict):
        return None
    if payload.get("schema_version") != 1 or payload.get("backend") != BACKEND:
        return None
    if (
        payload.get("variant") != variant
        or payload.get("model_version") != MODEL_VERSION
        or payload.get("quantization") != "fp32"
    ):
        return None
    if platform_info.get("os") != os_name or platform_info.get("arch") != arch:
        return None
    cache_dir = payload.get("cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        return None
    if variant == "coreml" and not payload.get("fluidaudio_version"):
        return None
    resolved = Path(cache_dir).expanduser()
    return resolved if resolved.exists() else None


def _write_sentinel(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _remove_sentinel(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _quarantine_path(cache_dir: Path) -> Path:
    base = cache_dir.with_name(f"{cache_dir.name}.partial-{_quarantine_suffix()}")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = cache_dir.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def _fail(message: str, code: int = 1) -> int:
    print(message, file=sys.stderr)
    return code


def _fail_with_quarantine(message: str, cache_dir: Path, sentinel_path: Path) -> int:
    _remove_sentinel(sentinel_path)
    if cache_dir.exists():
        quarantine = _quarantine_path(cache_dir)
        cache_dir.rename(quarantine)
        print(
            f"{message}; quarantined partial cache to {quarantine}",
            file=sys.stderr,
        )
        print(f"reclaim space with: rm -rf {quarantine}", file=sys.stderr)
        return 1
    return _fail(message)


def _disk_full_message(cache_dir: Path) -> str:
    usage_root = cache_dir if cache_dir.exists() else cache_dir.parent
    free_bytes = shutil.disk_usage(usage_root).free
    return (
        f"parakeet install failed: disk full at {cache_dir} "
        f"(free {free_bytes} bytes); free space and retry"
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    message = str(exc).lower()
    return "429" in message and "rate" in message and "limit" in message


def _verify_bundled_assets() -> None:
    for asset_path, expected_sha256 in (
        (WESPEAKER_MODEL_PATH, WESPEAKER_MODEL_SHA256),
        (PYANNOTE_OVERLAP_MODEL_PATH, PYANNOTE_OVERLAP_MODEL_SHA256),
    ):
        try:
            actual_sha256 = compute_file_sha256(asset_path)
        except OSError as exc:
            raise RuntimeError(
                f"bundled asset SHA mismatch: {asset_path}\n"
                f"  expected: {expected_sha256}\n"
                f"  actual:   unavailable ({exc})"
            ) from exc
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"bundled asset SHA mismatch: {asset_path}\n"
                f"  expected: {expected_sha256}\n"
                f"  actual:   {actual_sha256}"
            )


def _verify_linux_cache(cache_dir: Path) -> bool:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    for child in snapshots_dir.iterdir():
        if not child.is_dir():
            continue
        if not all(
            (child / relative_path).is_file() for relative_path in LINUX_MODEL_FILES
        ):
            continue
        total_bytes = sum(
            path.stat().st_size for path in child.rglob("*") if path.is_file()
        )
        if total_bytes >= LINUX_MIN_CACHE_BYTES:
            return True
    return False


def _verify_mac_cache(cache_dir: Path) -> bool:
    return all(
        (cache_dir.parent / MAC_FLUIDAUDIO_REPO_NAME / relative_path).is_file()
        for relative_path in MAC_MODEL_FILES
    )


def _verify_variant_cache(variant: str, cache_dir: Path) -> bool:
    if variant in {"cpu", "cuda"}:
        return _verify_linux_cache(cache_dir)
    return _verify_mac_cache(cache_dir)


def _check_parakeet_ready(
    os_name: str,
    arch: str,
    variant: str,
    sentinel_path: Path,
) -> Path:
    ready_cache = _sentinel_ready(
        _load_sentinel(sentinel_path),
        os_name,
        arch,
        variant,
    )
    if ready_cache is None:
        raise RuntimeError(
            f"parakeet check failed: sentinel not ready at {sentinel_path}"
        )
    if not _verify_variant_cache(variant, ready_cache):
        raise RuntimeError(
            f"parakeet check failed: cache verification failed at {ready_cache}"
        )
    return ready_cache


def _restore_linux_sentinel_if_cache_ready(
    os_name: str,
    arch: str,
    variant: str,
    sentinel_path: Path,
    cache_dir: Path,
) -> Path | None:
    if variant not in {"cpu", "cuda"} or not _verify_linux_cache(cache_dir):
        return None
    _write_sentinel(
        sentinel_path,
        _build_payload(os_name, arch, variant, cache_dir),
    )
    return cache_dir


def _build_payload(
    os_name: str,
    arch: str,
    variant: str,
    cache_dir: Path,
    *,
    fluidaudio_version: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "backend": BACKEND,
        "platform": {"os": os_name, "arch": arch},
        "variant": variant,
        "model_version": MODEL_VERSION,
        "quantization": "fp32",
        "fetched_at": _now_utc(),
        "cache_dir": str(cache_dir),
    }
    if fluidaudio_version is not None:
        payload["fluidaudio_version"] = fluidaudio_version
    return payload


def _fetch_linux_model(cache_dir: Path) -> None:
    if LEGACY_NEMO_MODEL_DIR.exists():
        print(
            "WARNING: legacy NeMo cache detected at "
            f"{LEGACY_NEMO_MODEL_DIR}; remove it manually if no longer needed: "
            f"rm -rf {LEGACY_NEMO_MODEL_DIR}"
        )
    try:
        try:
            from huggingface_hub.utils import HfHubHTTPError
        except Exception:  # pragma: no cover - older/newer hub versions vary
            HfHubHTTPError = None

        import onnx_asr

        model = onnx_asr.load_model(LINUX_LOAD_MODEL_ID, quantization=None)
        if model is None:
            raise RuntimeError(
                "parakeet install failed: onnx-asr.load_model returned no model"
            )
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise RuntimeError(_disk_full_message(cache_dir)) from exc
        raise
    except Exception as exc:
        if (
            HfHubHTTPError is not None and isinstance(exc, HfHubHTTPError)
        ) or _is_rate_limit_error(exc):
            raise RuntimeError(
                "parakeet install failed: Hugging Face rate limit (HTTP 429); retry in a few minutes"
            ) from exc
        raise


def _run_mac_helper(cache_dir: Path) -> dict[str, Any] | None:
    helper_env = os.getenv(HELPER_ENV_KEY)
    helper_path = _helper_path()
    if not helper_path.is_file() or not os.access(helper_path, os.X_OK):
        if not helper_env and is_packaged_install():
            print(PACKAGED_COREML_HINT, file=sys.stderr)
            return None
        raise RuntimeError(
            "parakeet install failed: helper not found or not executable at "
            f"{helper_path} run `make parakeet-helper` from the solstone repo to build it"
        )
    fixture_audio = _fixture_audio_path()
    if not fixture_audio.is_file():
        raise RuntimeError(
            f"parakeet install failed: fixture audio not found at {fixture_audio}"
        )

    try:
        result = subprocess.run(
            [
                str(helper_path),
                "--cache-dir",
                str(cache_dir),
                "--model",
                MODEL_VERSION,
                str(fixture_audio),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise RuntimeError(_disk_full_message(cache_dir)) from exc
        raise

    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        try:
            stderr_payload = json.loads(stderr_text) if stderr_text else {}
        except json.JSONDecodeError:
            stderr_payload = {}
        message = (
            stderr_payload.get("message") or stderr_text or "unknown helper failure"
        )
        raise RuntimeError(f"parakeet install failed: {message}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"parakeet install failed: helper returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("parakeet install failed: helper returned non-object JSON")
    return payload


def _install_models(os_name: str, arch: str, variant: str) -> int:
    sentinel_path = _sentinel_path(variant)
    cache_dir = _cache_dir(variant)
    _remove_sentinel(sentinel_path)

    if variant in {"cpu", "cuda"}:
        try:
            _fetch_linux_model(cache_dir)
        except RuntimeError as exc:
            return _fail_with_quarantine(str(exc), cache_dir, sentinel_path)
        except Exception as exc:
            return _fail_with_quarantine(
                f"parakeet install failed: {exc}",
                cache_dir,
                sentinel_path,
            )
        if not _verify_linux_cache(cache_dir):
            return _fail_with_quarantine(
                "parakeet install failed: Linux cache verification failed",
                cache_dir,
                sentinel_path,
            )
        _write_sentinel(
            sentinel_path,
            _build_payload(os_name, arch, variant, cache_dir),
        )
        print(f"model ready: {cache_dir}")
        return 0

    try:
        payload = _run_mac_helper(cache_dir)
    except RuntimeError as exc:
        return _fail_with_quarantine(str(exc), cache_dir, sentinel_path)
    except Exception as exc:
        return _fail_with_quarantine(
            f"parakeet install failed: {exc}",
            cache_dir,
            sentinel_path,
        )
    if payload is None:
        return 0
    if not _verify_mac_cache(cache_dir):
        return _fail_with_quarantine(
            "parakeet install failed: macOS cache verification failed",
            cache_dir,
            sentinel_path,
        )
    fluidaudio_version = payload.get("fluidaudio_version")
    if not isinstance(fluidaudio_version, str) or not fluidaudio_version:
        return _fail_with_quarantine(
            "parakeet install failed: helper success JSON missing fluidaudio_version",
            cache_dir,
            sentinel_path,
        )
    _write_sentinel(
        sentinel_path,
        _build_payload(
            os_name,
            arch,
            variant,
            cache_dir,
            fluidaudio_version=fluidaudio_version,
        ),
    )
    print(f"model ready: {cache_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install and verify solstone's bundled ML models (parakeet ASR plus "
            "bundled wespeaker/pyannote assets). Default action checks the "
            "parakeet sentinel + cache and fetches if missing; --force re-fetches; "
            "--check verifies only and exits nonzero on any problem."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="Verify bundled assets and the parakeet sentinel/cache without fetching.",
    )
    mode_group.add_argument(
        "--force",
        action="store_true",
        help="Ignore the sentinel and refetch/verify the model cache.",
    )
    parser.add_argument(
        "--variant",
        choices=("auto", "cpu", "cuda", "coreml"),
        default="auto",
        help=(
            "Parakeet variant to install or verify. auto honors "
            "PARAKEET_ONNX_VARIANT on linux/x86_64, then autodetects."
        ),
    )
    args = parser.parse_args()

    os_name, arch = _platform_info()
    variant = _resolve_variant(
        args.variant,
        os.getenv(PARAKEET_ONNX_VARIANT_ENV),
        os_name,
        arch,
    )

    try:
        # Why: local asset corruption makes downloading parakeet pointless.
        _verify_bundled_assets()
    except RuntimeError as exc:
        return _fail(str(exc))

    if variant is None:
        print(
            "parakeet install: unsupported platform "
            f"{os_name}/{arch}; supported: darwin/arm64, linux/x86_64"
        )
        return 0

    if os_name == "linux" and arch == "x86_64" and variant == "cuda":
        try:
            import onnxruntime

            providers = onnxruntime.get_available_providers()
        except Exception:
            sys.exit(
                "device=cuda requires CUDAExecutionProvider; rerun: "
                "PARAKEET_ONNX_VARIANT=cuda make install"
            )
        if "CUDAExecutionProvider" not in providers:
            sys.exit(
                "device=cuda requires CUDAExecutionProvider; rerun: "
                "PARAKEET_ONNX_VARIANT=cuda make install"
            )

    sentinel_path = _sentinel_path(variant)
    cache_dir = _cache_dir(variant)
    if args.check:
        try:
            ready_cache = _check_parakeet_ready(
                os_name,
                arch,
                variant,
                sentinel_path,
            )
        except RuntimeError as exc:
            return _fail(str(exc))
        print(f"model ready: {ready_cache}")
        return 0

    if not args.force:
        ready_cache = _sentinel_ready(
            _load_sentinel(sentinel_path),
            os_name,
            arch,
            variant,
        )
        if ready_cache is not None and _verify_variant_cache(variant, ready_cache):
            print(f"model ready: {ready_cache}")
            return 0
        restored_cache = _restore_linux_sentinel_if_cache_ready(
            os_name,
            arch,
            variant,
            sentinel_path,
            cache_dir,
        )
        if restored_cache is not None:
            print(f"model ready: {restored_cache}")
            return 0

    return _install_models(os_name, arch, variant)


if __name__ == "__main__":
    raise SystemExit(main())
