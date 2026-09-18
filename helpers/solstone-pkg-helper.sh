#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

# Closed-vocabulary privileged package helper for Solstone platform installer.
# Invoked via sudo -n by the non-root installer when package-route mutation is required.

set -eu

# Default production paths (immutable unless explicitly overridden via CLI flags under test seam)
LOCK_DIR="/run/lock/solstone-platform"
ETC_ROOT="/etc"
FAKE_PKG_DB=""

while [ $# -gt 0 ]; do
    case "$1" in
        --lock-dir)
            LOCK_DIR="$2"
            shift 2
            ;;
        --etc-root)
            ETC_ROOT="$2"
            shift 2
            ;;
        --fake-pkg-db)
            FAKE_PKG_DB="$2"
            shift 2
            ;;
        *)
            echo "ERROR:unknown-flag:$1" >&2
            exit 1
            ;;
    esac
done

RECEIPT_FILE="${ETC_ROOT}/solstone/install.conf"

# Sanitization helpers: reject metacharacters, escapes, traversal
validate_name() {
    name="$1"
    case "$name" in
        *[!A-Za-z0-9._-]*)
            echo "ERROR:invalid-package-name" >&2
            exit 1
            ;;
        "")
            echo "ERROR:empty-package-name" >&2
            exit 1
            ;;
    esac
}

validate_path() {
    target_path="$1"
    case "$target_path" in
        /*) ;;
        *)
            echo "ERROR:path-not-absolute" >&2
            exit 1
            ;;
    esac
    case "$target_path" in
        *..*|*\;*|*\&*|*\|*|*\`*|*\$*|*\(*|*\)*|*\<*|*\>*|*\\*)
            echo "ERROR:path-unsafe" >&2
            exit 1
            ;;
    esac
}

acquire_lock() {
    # Verify no symlink in path components
    if [ -L "$LOCK_DIR" ]; then
        echo "ERROR:lock-dir-symlink" >&2
        exit 1
    fi
    mkdir -p "$LOCK_DIR"
    chmod 0755 "$LOCK_DIR" 2>/dev/null || true
    LOCK_FILE="${LOCK_DIR}/lock"
    if [ -L "$LOCK_FILE" ]; then
        echo "ERROR:lock-file-symlink" >&2
        exit 1
    fi
    # Open lock file on fd 9
    exec 9>>"$LOCK_FILE"
    if command -v flock >/dev/null 2>&1; then
        if ! flock -n 9; then
            echo "ERROR:package-locked" >&2
            exit 1
        fi
    fi
}

handle_query_pkg() {
    pkg_type="$1"
    pkg_name="$2"
    validate_name "$pkg_name"

    if [ -n "$FAKE_PKG_DB" ] && [ -d "$FAKE_PKG_DB" ]; then
        db_file="${FAKE_PKG_DB}/${pkg_type}/${pkg_name}"
        if [ -f "$db_file" ]; then
            cat "$db_file"
            return 0
        fi
        echo "NOT_INSTALLED"
        return 0
    fi

    if [ "$pkg_type" = "deb" ]; then
        if command -v dpkg-query >/dev/null 2>&1; then
            if dpkg-query -W -f='${Status}\t${Version}\n' "$pkg_name" 2>/dev/null; then
                return 0
            fi
        fi
        echo "NOT_INSTALLED"
    elif [ "$pkg_type" = "rpm" ]; then
        if command -v rpm >/dev/null 2>&1; then
            if rpm -q --queryformat '%{VERSION}-%{RELEASE}\n' "$pkg_name" 2>/dev/null; then
                return 0
            fi
        fi
        echo "NOT_INSTALLED"
    else
        echo "ERROR:unsupported-pkg-type" >&2
        exit 1
    fi
}

handle_install_pkg() {
    pkg_type="$1"
    archive_path="$2"
    validate_path "$archive_path"

    if [ ! -f "$archive_path" ]; then
        echo "ERROR:archive-missing" >&2
        exit 1
    fi

    if [ -n "$FAKE_PKG_DB" ] && [ -d "$FAKE_PKG_DB" ]; then
        pkg_name=$(basename "$archive_path" | sed -E 's/[-_][0-9].*//')
        mkdir -p "${FAKE_PKG_DB}/${pkg_type}"
        # Record installation in fake package DB and install log
        echo "install ok installed 2.0.6" > "${FAKE_PKG_DB}/${pkg_type}/${pkg_name}"
        echo "$archive_path" >> "${FAKE_PKG_DB}/install.log"
        echo "OK"
        return 0
    fi

    if [ "$pkg_type" = "deb" ]; then
        dpkg -i "$archive_path"
        echo "OK"
    elif [ "$pkg_type" = "rpm" ]; then
        rpm -U --replacepkgs "$archive_path"
        echo "OK"
    else
        echo "ERROR:unsupported-pkg-type" >&2
        exit 1
    fi
}

handle_remove_pkg() {
    pkg_type="$1"
    pkg_name="$2"
    validate_name "$pkg_name"

    if [ -n "$FAKE_PKG_DB" ] && [ -d "$FAKE_PKG_DB" ]; then
        rm -f "${FAKE_PKG_DB}/${pkg_type}/${pkg_name}"
        echo "OK"
        return 0
    fi

    if [ "$pkg_type" = "deb" ]; then
        dpkg -r "$pkg_name"
        echo "OK"
    elif [ "$pkg_type" = "rpm" ]; then
        rpm -e "$pkg_name"
        echo "OK"
    else
        echo "ERROR:unsupported-pkg-type" >&2
        exit 1
    fi
}

handle_write_receipt() {
    len="$1"
    case "$len" in
        ''|*[!0-9]*)
            echo "ERROR:invalid-length" >&2
            exit 1
            ;;
    esac

    # Ensure parent dir exists
    receipt_dir=$(dirname "$RECEIPT_FILE")
    if [ -L "$receipt_dir" ] || [ -L "$RECEIPT_FILE" ]; then
        echo "ERROR:receipt-symlink" >&2
        exit 1
    fi
    mkdir -p "$receipt_dir"
    chmod 0755 "$receipt_dir" 2>/dev/null || true

    tmp_file="${RECEIPT_FILE}.tmp.$$"
    # Read exactly $len bytes from stdin
    head -c "$len" > "$tmp_file"
    chmod 0644 "$tmp_file"
    mv -f "$tmp_file" "$RECEIPT_FILE"
    echo "OK"
}

handle_read_receipt() {
    if [ -f "$RECEIPT_FILE" ]; then
        cat "$RECEIPT_FILE"
    fi
}

# Main command processing loop
acquire_lock

while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    # Reject shell metacharacters in command line
    case "$line" in
        *\;*|*\&*|*\|*|*\`*|*\$*|*\(*|*\)*|*\<*|*\>*|*\\*)
            echo "ERROR:metacharacters-rejected" >&2
            exit 1
            ;;
    esac

    # shellcheck disable=SC2086
    set -- $line
    opcode="${1:-}"

    case "$opcode" in
        PING)
            if [ $# -ne 1 ]; then
                echo "ERROR:invalid-ping-operands" >&2
                exit 1
            fi
            echo "PONG"
            ;;
        QUERY_PKG)
            if [ $# -ne 3 ]; then
                echo "ERROR:invalid-query-operands" >&2
                exit 1
            fi
            handle_query_pkg "$2" "$3"
            ;;
        INSTALL_PKG)
            if [ $# -ne 3 ]; then
                echo "ERROR:invalid-install-operands" >&2
                exit 1
            fi
            handle_install_pkg "$2" "$3"
            ;;
        REMOVE_PKG)
            if [ $# -ne 3 ]; then
                echo "ERROR:invalid-remove-operands" >&2
                exit 1
            fi
            handle_remove_pkg "$2" "$3"
            ;;
        WRITE_ETC_RECEIPT)
            if [ $# -ne 2 ]; then
                echo "ERROR:invalid-write-receipt-operands" >&2
                exit 1
            fi
            handle_write_receipt "$2"
            ;;
        READ_ETC_RECEIPT)
            if [ $# -ne 1 ]; then
                echo "ERROR:invalid-read-receipt-operands" >&2
                exit 1
            fi
            handle_read_receipt
            ;;
        *)
            echo "ERROR:unknown-opcode:$opcode" >&2
            exit 1
            ;;
    esac
done
