#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release.sh [--test]

Options:
  --test      Publish to TestPyPI.
  -h, --help  Show this help.
EOF
}

TARGET="pypi"
TOKEN_VAR="PYPI_TOKEN"
REPOSITORY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TARGET="testpypi"
            TOKEN_VAR="TESTPYPI_TOKEN"
            REPOSITORY_ARGS=(--repository-url https://test.pypi.org/legacy/)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${!TOKEN_VAR:-}" ]]; then
    echo "set \$${TOKEN_VAR} before re-running" >&2
    exit 1
fi

TOKEN="${!TOKEN_VAR}"

rm -rf dist/
uv build
uvx twine check dist/*

TWINE_USERNAME=__token__ TWINE_PASSWORD="$TOKEN" uvx twine upload "${REPOSITORY_ARGS[@]}" dist/*

VERSION=$(ls dist/solstone-*-py3-none-any.whl | head -1 | sed -E 's/.*solstone-([^-]+)-.*/\1/')
echo "published solstone ${VERSION} to ${TARGET}"
