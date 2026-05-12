#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
#
# Sign and notarize a bare Mach-O helper binary for the macOS platform wheel.
# Stapling is intentionally skipped — bare Mach-O binaries are not stapleable;
# Gatekeeper performs an online check on first run instead.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <binary-path>" >&2
    exit 2
fi

BINARY="$1"
IDENTITY="${CODESIGN_IDENTITY:-Developer ID Application: sol pbc (7QCG8V4M6H)}"
PROFILE="${NOTARY_KEYCHAIN_PROFILE:-sol-pbc-notary}"
NOTARY_KEYCHAIN="${NOTARY_KEYCHAIN:-$HOME/Library/Keychains/sol-signing.keychain-db}"

if [ ! -f "$BINARY" ]; then
    echo "error: not a regular file: $BINARY" >&2
    exit 1
fi

echo "==> codesigning $BINARY (identity: $IDENTITY)"
codesign --force --options runtime --timestamp \
    --keychain "$NOTARY_KEYCHAIN" \
    --sign "$IDENTITY" \
    "$BINARY"

echo "==> verifying signature"
codesign --verify --strict --verbose=2 "$BINARY"

ZIPDIR="$(mktemp -d)"
trap 'rm -rf "$ZIPDIR"' EXIT
ZIPPATH="$ZIPDIR/$(basename "$BINARY").zip"

echo "==> packaging $BINARY for notarytool"
ditto -c -k --keepParent "$BINARY" "$ZIPPATH"

echo "==> submitting to notarytool (keychain profile: $PROFILE in $NOTARY_KEYCHAIN)"
xcrun notarytool submit "$ZIPPATH" \
    --keychain-profile "$PROFILE" \
    --keychain "$NOTARY_KEYCHAIN" \
    --wait

echo "==> sign-and-notarize complete: $BINARY"
echo "note: bare Mach-O binaries cannot be stapled; Gatekeeper performs an online check on first run."
