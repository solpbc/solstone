# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Secret scanner and symlink verifier for CI gate."""

from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

RE_MINISIGN_SECRET_HEADER = re.compile(r"untrusted comment:\s*minisign secret key", re.IGNORECASE)
RE_MINISIGN_RWR = re.compile(r"(?<![A-Za-z0-9+/=])RWR[A-Za-z0-9+/=]{60,}")
RE_AWS_REAL_KEY = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AROA)[A-Z0-9]{16}(?![A-Z0-9])")
RE_AWS_EXAMPLE_KEY = re.compile(r"AKIDEXAMPLE")


def check_symlink() -> None:
    claude = REPO_ROOT / "CLAUDE.md"
    if not claude.is_symlink():
        print("ERROR: CLAUDE.md must be a symlink to AGENTS.md", file=sys.stderr)
        sys.exit(1)
    target = claude.resolve()
    agents = (REPO_ROOT / "AGENTS.md").resolve()
    if target != agents:
        print(f"ERROR: CLAUDE.md symlink points to {target}, expected {agents}", file=sys.stderr)
        sys.exit(1)


def get_tracked_and_staged_files() -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        files = [REPO_ROOT / p for p in proc.stdout.decode("utf-8").splitlines() if p.strip()]
        return files
    except Exception:
        # Fallback to traversing repo
        return [p for p in REPO_ROOT.rglob("*") if p.is_file() and not p.name.endswith(".pyc") and ".git" not in p.parts]


def scan_files() -> None:
    violations = []
    files = get_tracked_and_staged_files()

    allowed_example_files = {
        (REPO_ROOT / "testdata" / "sigv4" / "get-vanilla.json").resolve(),
        (REPO_ROOT / "testdata" / "sigv4" / "post-vanilla.json").resolve(),
        (REPO_ROOT / "tests" / "test_redact.py").resolve(),
        (REPO_ROOT / "tests" / "test_sigv4.py").resolve(),
        (REPO_ROOT / "tools" / "secretscan.py").resolve(),
    }

    this_tool = (REPO_ROOT / "tools" / "secretscan.py").resolve()
    redact_mod = (REPO_ROOT / "src" / "solstone_platform" / "redact.py").resolve()
    test_redact = (REPO_ROOT / "tests" / "test_redact.py").resolve()

    for path in files:
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        resolved = path.resolve()

        # Check for minisign secret header
        if resolved not in (this_tool, redact_mod, test_redact):
            if RE_MINISIGN_SECRET_HEADER.search(content):
                violations.append(f"{path.relative_to(REPO_ROOT)}: found minisign secret header")

        # Check for RWR private key
        if resolved not in (this_tool, redact_mod, test_redact):
            if RE_MINISIGN_RWR.search(content):
                violations.append(f"{path.relative_to(REPO_ROOT)}: found Minisign RWR secret key material")

        # Check for real AWS access keys (AKIA/ASIA/AROA)
        if resolved not in (this_tool, redact_mod, test_redact):
            if RE_AWS_REAL_KEY.search(content):
                violations.append(f"{path.relative_to(REPO_ROOT)}: found AWS access key (AKIA/ASIA/AROA)")

        # Check for AKIDEXAMPLE
        if RE_AWS_EXAMPLE_KEY.search(content):
            if resolved not in allowed_example_files:
                violations.append(f"{path.relative_to(REPO_ROOT)}: unauthorized occurrence of AKIDEXAMPLE")

    if violations:
        print("Secret scan failed with violations:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    check_symlink()
    scan_files()


if __name__ == "__main__":
    main()
