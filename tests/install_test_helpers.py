# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Helper utilities and mock loopback servers for installer unit tests."""

from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import shutil
import threading

from solstone_platform.canonical import canonical_json_bytes
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.pins import MinisignPin, embedded_pins
from solstone_platform.sign import sign_manifest
from tools.fixture_builder import build_tiny_natives

REPO_ROOT = Path(__file__).resolve().parent.parent


def receipt_section(receipt: bytes, name: str) -> bytes:
    marker = f"[{name}]\n".encode()
    start = receipt.index(marker)
    next_section = receipt.find(b"\n[", start + len(marker))
    return receipt[start:] if next_section == -1 else receipt[start:next_section + 1]


def write_path_stub(bin_dir: Path, name: str, body: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def setup_package_launcher_spy(bin_dir: Path) -> Path:
    return write_path_stub(
        bin_dir,
        "package-journal-spy",
        "if [ -n \"${SOLSTONE_PACKAGE_SETUP_LOG:-}\" ]; then\n"
        "  printf 'argv0=%s\\n' \"$0\" >> \"$SOLSTONE_PACKAGE_SETUP_LOG\"\n"
        "  for arg do printf 'arg=%s\\n' \"$arg\" >> \"$SOLSTONE_PACKAGE_SETUP_LOG\"; done\n"
        "  printf '%s\\n' -- >> \"$SOLSTONE_PACKAGE_SETUP_LOG\"\n"
        "fi\n"
        "exit \"${SOLSTONE_PACKAGE_SETUP_EXIT:-0}\"\n",
    )


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.server.request_paths.append(self.path)
        super().do_GET()

    def log_message(self, format, *args):
        pass


class LoopbackServer:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        handler = partial(QuietHTTPRequestHandler, directory=str(root_dir))
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.httpd.request_paths = []
        self.port = self.httpd.server_port
        self.origin = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def request_paths(self):
        return self.httpd.request_paths

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def setup_fake_sudo(bin_dir: Path) -> Path:
    """Create a mock sudo binary in bin_dir that enforces -n flag and delegates without root."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    sudo_script = bin_dir / "sudo"
    sudo_script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" != \"-n\" ]; then\n"
        "  echo 'sudo: a password is required' >&2\n"
        "  exit 1\n"
        "fi\n"
        "shift\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    sudo_script.chmod(0o755)
    return sudo_script


def make_v2_bootstrap_script(revision: int = 2) -> bytes:
    return (
        f"#!/bin/sh\n"
        f"BOOTSTRAP_REVISION={revision}\n"
        f"BOOTSTRAP_CONTRACT_VERSION=2\n"
        f"ROLE='journal'\n"
        f"PREFIX=''\n"
        f"DRY_RUN=0\n"
        f"if [ -n \"${{SOLSTONE_BOOTSTRAP_ARGS_LOG:-}}\" ]; then\n"
        f"  : > \"$SOLSTONE_BOOTSTRAP_ARGS_LOG\"\n"
        f"  for ARG do printf '%s\\n' \"$ARG\" >> \"$SOLSTONE_BOOTSTRAP_ARGS_LOG\"; done\n"
        f"fi\n"
        f"while [ $# -gt 0 ]; do\n"
        f"  case \"$1\" in\n"
        f"    --role) ROLE=\"$2\"; shift 2 ;;\n"
        f"    --prefix) PREFIX=\"$2\"; shift 2 ;;\n"
        f"    --dry-run) DRY_RUN=1; shift 1 ;;\n"
        f"    *) shift 1 ;;\n"
        f"  esac\n"
        f"done\n"
        f"if [ -n \"$PREFIX\" ] && [ \"$DRY_RUN\" -eq 0 ]; then\n"
        f"  mkdir -p \"$PREFIX/bin\" \"$PREFIX/versions/2.0.6-fixture/bin\"\n"
        f"  echo \"#!/bin/sh\\necho $ROLE 2.0.6\" > \"$PREFIX/versions/2.0.6-fixture/bin/journal\"\n"
        f"  chmod +x \"$PREFIX/versions/2.0.6-fixture/bin/journal\"\n"
        f"  ln -sfn \"versions/2.0.6-fixture\" \"$PREFIX/current\"\n"
        f"  ln -sfn \"$PREFIX/current/bin/journal\" \"$PREFIX/bin/journal\"\n"
        f"  printf 'schema_version=1\\njournal_version=2.0.6\\nlane=release\\norigin=fixture\\narchitecture=x86_64\\ninstaller_revision=1\\nbootstrap_revision={revision}\\nroute=tree\\nsignature_verification=skipped\\nrole=%s\\njournal_state=existing\\nservice_policy=skip-service\\nsetup_status=complete\\n' \"$ROLE\" > \"$PREFIX/install-receipt\"\n"
        f"fi\n"
        f"exit 0\n"
    ).encode("utf-8")


def setup_test_release_server(
    work_dir: Path,
    sec_key_path: Path,
    pin: MinisignPin,
    lane: str = "release",
    version: str = "2.0.0",
    min_installer_revision: int = 1,
    bootstrap_revision: int = 2,
    corrupt_manifest: bool = False,
    corrupt_signature: bool = False,
    duplicate_key: bool = False,
    native_version: str = "2.0.3",
    desktop_version: str | None = None,
    tmux_version: str | None = None,
    desktop_runtime_version: str | None = None,
    native_build_marker: str = "",
) -> tuple[LoopbackServer, Path]:
    server_root = work_dir / "www"
    server_root.mkdir(parents=True, exist_ok=True)

    # Build tiny natives with ephemeral keys and v2 bootstrap double
    v2_boot = make_v2_bootstrap_script(revision=bootstrap_revision)
    pin_set, native_dirs = build_tiny_natives(
        target_dir=work_dir / "natives",
        bootstrap_script=v2_boot,
        min_bootstrap_revision=bootstrap_revision,
        native_version=native_version,
        desktop_version=desktop_version,
        tmux_version=tmux_version,
        desktop_runtime_version=desktop_runtime_version,
        native_build_marker=native_build_marker,
    )

    server = LoopbackServer(server_root)
    server.start()

    manifest_bytes = generate_platform_manifest(
        version=version,
        lane=lane,
        created_unix=1773820000,
        source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
        platform_key_id=pin.key_id,
        repo_root=REPO_ROOT,
        journal_dir=native_dirs["journal"],
        desktop_dir=native_dirs["desktop"],
        tmux_dir=native_dirs["tmux"],
        journal_origin=server.origin,
        pins=pin_set,
    )

    # Existing installer tests intentionally use --skip-signature for native
    # fixtures. Keep those catalogues pinned to the production identities while
    # retaining the ephemeral signatures/hashes for digest and semantic tests.
    manifest_obj = json.loads(manifest_bytes.decode("utf-8"))
    production_pins = embedded_pins()
    for component_name, selected_pin in (
        ("journal", production_pins.journal),
        ("desktop", production_pins.desktop),
        ("tmux", production_pins.tmux),
    ):
        for arch_entry in manifest_obj["components"][component_name]["arches"].values():
            for route_entry in arch_entry.values():
                route_entry["authority"]["verifier_id"] = selected_pin.verifier_id()
    manifest_bytes = canonical_json_bytes(manifest_obj)

    if min_installer_revision != 1:
        m_obj = json.loads(manifest_bytes.decode("utf-8"))
        m_obj["minimum_installer_revision"] = min_installer_revision
        manifest_bytes = canonical_json_bytes(m_obj)

    signed_manifest_bytes = manifest_bytes
    if duplicate_key:
        manifest_bytes = manifest_bytes.replace(b'"schema_version":1,', b'"schema_version":1,"schema_version":1,')
    elif corrupt_manifest:
        manifest_bytes = manifest_bytes + b"\nINVALID_EXTRA_JSON"

    # Setup web root structure: solstone/{lane}/latest and solstone/{lane}/{version}/...
    lane_dir = server_root / "solstone" / lane
    ver_dir = lane_dir / version
    ver_dir.mkdir(parents=True, exist_ok=True)

    # Write latest
    (lane_dir / "latest").write_text(f"{version}\n", encoding="utf-8")

    # Write manifest
    manifest_path = ver_dir / "platform.json"
    manifest_path.write_bytes(manifest_bytes)

    # Sign manifest
    sig_bytes = sign_manifest(
        manifest_bytes=signed_manifest_bytes,
        secret_key_path=sec_key_path,
        selected_pin=pin,
        repo_root=REPO_ROOT,
        is_production=False,
    )
    if corrupt_signature:
        sig_bytes = b"corrupted signature bytes\n"
    (ver_dir / "platform.json.minisig").write_bytes(sig_bytes)

    # Copy journal bootstrap
    j_ver = "2.0.6"
    j_bootstrap_dir = server_root / "solstone-journal" / lane / j_ver
    j_bootstrap_dir.mkdir(parents=True, exist_ok=True)
    (j_bootstrap_dir / f"solstone-journal-{j_ver}-install.sh").write_bytes(v2_boot)

    # Copy all component variant archives (tar.gz, deb, rpm) to ver_dir.
    for comp_dir in native_dirs.values():
        for archive in comp_dir.rglob("*"):
            if archive.is_file() and archive.suffix in (".gz", ".deb", ".rpm"):
                shutil.copy2(archive, ver_dir / archive.name)

    # Journal native authority objects publish beside platform.json. The delegated
    # bootstrap remains under its separate solstone-journal coordinate above.
    for authority_file in native_dirs["journal"].rglob("*"):
        if not authority_file.is_file():
            continue
        if authority_file.name.endswith((".manifest.json", ".manifest.json.minisig", ".release", ".sha256")):
            shutil.copy2(authority_file, ver_dir / authority_file.name)

    # Desktop's signed producer manifest is also verified by the orchestrator
    # before payload work, so publish the exact authority pair beside platform.json.
    for authority_file in native_dirs["desktop"].glob("*.rust-release-manifest.json*"):
        if authority_file.is_file():
            shutil.copy2(authority_file, ver_dir / authority_file.name)

    # Tmux delegates trust to its signed aggregate sums plus one arch target.
    for authority_file in native_dirs["tmux"].iterdir():
        if authority_file.is_file() and (
            authority_file.name in ("SHA256SUMS", "SHA256SUMS.minisig")
            or authority_file.name.endswith(".target.json")
        ):
            shutil.copy2(authority_file, ver_dir / authority_file.name)

    return server, server_root
