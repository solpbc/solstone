#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/cleanroom-install.sh [--source wheel|testpypi|pypi] [--image python:3.12-slim|python:3.12] [--version X.Y.Z]
  --source    wheel, testpypi, or pypi. Default: wheel.
  --image     One image only. Default: python:3.12-slim then python:3.12.
  --version   Default: dist wheel version for wheel, else 0.1.1.
  -h, --help  Show help.
EOF
}

SOURCE="wheel"
IMAGE=""
VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source|--image|--version)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --source) SOURCE="$2" ;;
                --image) IMAGE="$2" ;;
                --version) VERSION="$2" ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$SOURCE" in wheel|testpypi|pypi) ;; *) echo "invalid --source: $SOURCE" >&2; exit 2 ;; esac
case "$IMAGE" in ""|python:3.12-slim|python:3.12) ;; *) echo "invalid --image: $IMAGE" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$VERSION" && "$SOURCE" == "wheel" ]]; then
    VERSION=$(ls dist/solstone-*-py3-none-any.whl 2>/dev/null | head -1 | sed -E 's/.*solstone-([^-]+)-.*/\1/')
    if [[ -z "$VERSION" ]]; then
        echo "no wheel found in dist/, run uv build first" >&2
        exit 1
    fi
elif [[ -z "$VERSION" ]]; then
    VERSION="0.1.1"
fi

case "$SOURCE" in
    wheel) INSTALL_ARGS="/work/dist/solstone-${VERSION}-py3-none-any.whl" ;;
    testpypi) INSTALL_ARGS="--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ solstone==${VERSION}" ;;
    pypi) INSTALL_ARGS="solstone==${VERSION}" ;;
esac

summaries=()
full_status=0

run_image() {
    local image="$1"
    local propagate_failure="$2"
    local label="full"
    local command="pip install ${INSTALL_ARGS} && sol --version"
    local mount_args=()
    local output status version_output

    if [[ "$image" == "python:3.12-slim" ]]; then
        label="slim"
        command="apt-get update && apt-get install -y --no-install-recommends python3-pip ca-certificates && pip install --break-system-packages ${INSTALL_ARGS} && sol --version"
    fi
    [[ "$SOURCE" == "wheel" ]] && mount_args=(-v "$(pwd)/dist:/work/dist:ro")

    set +e
    output=$(docker run --rm "${mount_args[@]}" "$image" bash -c "$command" 2>&1)
    status=$?
    set -e

    if [[ "$status" -eq 0 ]]; then
        version_output=$(printf '%s\n' "$output" | tail -1)
        summaries+=("${label}: PASS - sol --version: ${version_output}")
    elif [[ "$label" == "slim" ]]; then
        summaries+=("slim: FAIL (documentation only)")
    else
        summaries+=("full: FAIL")
    fi
    [[ "$propagate_failure" == "yes" ]] && full_status="$status"
    return 0
}

if [[ -n "$IMAGE" ]]; then
    if [[ "$IMAGE" == "python:3.12" ]]; then
        run_image "$IMAGE" yes
    else
        run_image "$IMAGE" no
    fi
else
    run_image "python:3.12-slim" no
    run_image "python:3.12" yes
fi

printf '%s\n' "${summaries[@]}"
exit "$full_status"
