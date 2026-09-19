#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

# Closed-vocabulary privileged package helper for Solstone platform installer.
# Invoked via sudo -n by the non-root installer when package-route mutation is required.

set -efu

# Default production paths (immutable unless explicitly overridden via CLI flags under test seam)
LOCK_DIR="/run/lock/solstone-platform"
ETC_ROOT="/etc"
FAKE_PKG_DB=""
FAKE_ROOT=""
FAKE_JOURNAL_LAUNCHER=""

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
        --fake-root)
            FAKE_ROOT="$2"
            shift 2
            ;;
        --fake-journal-launcher)
            FAKE_JOURNAL_LAUNCHER="$2"
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
            if ! awk '
                NR == 1 && NF == 4 && ($1 == "INSTALLED" || $1 == "UNCONFIGURED") && $2 ~ /^[A-Za-z0-9._-]+$/ && $3 ~ /^[^[:space:]]+$/ && $4 ~ /^[A-Za-z0-9._-]+$/ { valid = 1; next }
                { invalid = 1 }
                END { exit !(valid && !invalid) }
            ' "$db_file"; then
                echo "ERROR:query-malformed"
                return 1
            fi
            cat "$db_file"
            return 0
        fi
        echo "ABSENT"
        return 0
    fi

    if [ "$pkg_type" = "deb" ]; then
        if ! command -v dpkg-query >/dev/null 2>&1; then
            echo "ERROR:query-unavailable"
            return 1
        fi
        if query_raw=$(dpkg-query -W -f='${Status}\t${Package}\t${Version}\t${Architecture}\n' "$pkg_name" 2>/dev/null); then
            :
        else
            query_status=$?
            if [ "$query_status" -eq 1 ]; then
                echo "ABSENT"
                return 0
            fi
            echo "ERROR:query-failed"
            return 1
        fi
        if ! query_out=$(printf '%s\n' "$query_raw" | awk -F '\t' -v wanted="$pkg_name" '
            NR != 1 || NF != 4 { bad = 1; next }
            $2 != wanted || $3 == "" || $4 !~ /^[A-Za-z0-9._-]+$/ { bad = 1; next }
            $1 == "install ok installed" { print "INSTALLED " $2 " " $3 " " $4; seen = 1; next }
            $1 == "install ok unpacked" || $1 == "install ok half-configured" || $1 == "install ok triggers-awaited" || $1 == "install ok triggers-pending" { print "UNCONFIGURED " $2 " " $3 " " $4; seen = 1; next }
            $1 == "deinstall ok config-files" { print "CONFIG_FILES " $2 " " $3 " " $4; seen = 1; next }
            { bad = 1 }
            END { exit !(seen && !bad) }
        '); then
            echo "ERROR:query-malformed"
            return 1
        fi
        printf '%s\n' "$query_out"
    elif [ "$pkg_type" = "rpm" ]; then
        if ! command -v rpm >/dev/null 2>&1; then
            echo "ERROR:query-unavailable"
            return 1
        fi
        if query_raw=$(rpm -q --queryformat '%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n' "$pkg_name" 2>/dev/null); then
            :
        else
            query_status=$?
            if [ "$query_status" -eq 1 ]; then
                echo "ABSENT"
                return 0
            fi
            echo "ERROR:query-failed"
            return 1
        fi
        if ! query_out=$(printf '%s\n' "$query_raw" | awk -F '\t' -v wanted="$pkg_name" '
            NR != 1 || NF != 3 { bad = 1; next }
            $1 != wanted || $2 == "" || $3 !~ /^[A-Za-z0-9._-]+$/ { bad = 1; next }
            { print "INSTALLED " $1 " " $2 " " $3; seen = 1 }
            END { exit !(seen && !bad) }
        '); then
            echo "ERROR:query-malformed"
            return 1
        fi
        printf '%s\n' "$query_out"
    else
        echo "ERROR:unsupported-pkg-type"
        return 1
    fi
}

parse_archive_identity() {
    # Package placeholders belong to dpkg-deb, not the shell.
    # shellcheck disable=SC2016
    case "$1" in
        deb) dpkg-deb --show --showformat='${Package} ${Version} ${Architecture}\n' "$2" ;;
        rpm) rpm -qp --queryformat '%{NAME} %{VERSION}-%{RELEASE} %{ARCH}\n' "$2" ;;
        *) return 1 ;;
    esac
}

handle_install_pkg() {
    pkg_type="$1"
    archive_path="$2"
    validate_path "$archive_path"

    if [ ! -f "$archive_path" ]; then
        echo "ERROR:archive-missing" >&2
        exit 1
    fi

    if ! pkg_identity=$(parse_archive_identity "$pkg_type" "$archive_path"); then
        echo "ERROR:install-failed"
        return 1
    fi
    # Deliberate whitespace split of three identity fields; pathname expansion is disabled.
    # shellcheck disable=SC2086
    set -- $pkg_identity
    if [ $# -ne 3 ]; then
        echo "ERROR:install-failed"
        return 1
    fi
    pkg_name="$1"
    pkg_version="$2"
    pkg_arch="$3"
    validate_name "$pkg_name"

    if [ -n "$FAKE_PKG_DB" ] && [ -d "$FAKE_PKG_DB" ]; then
        mkdir -p "${FAKE_PKG_DB}/${pkg_type}"
        printf 'INSTALLED %s %s %s\n' "$pkg_name" "$pkg_version" "$pkg_arch" > "${FAKE_PKG_DB}/${pkg_type}/${pkg_name}"
        if [ "$pkg_name" = "solstone-journal" ] && [ -n "$FAKE_ROOT" ]; then
            if [ -z "$FAKE_JOURNAL_LAUNCHER" ] || [ ! -x "$FAKE_JOURNAL_LAUNCHER" ]; then
                echo "ERROR:install-failed"
                return 1
            fi
            fake_journal_dir="${FAKE_ROOT}/usr/bin"
            if [ -L "$FAKE_ROOT" ] || [ -L "$fake_journal_dir" ] || [ -L "${fake_journal_dir}/journal" ]; then
                echo "ERROR:install-failed"
                return 1
            fi
            mkdir -p "$fake_journal_dir"
            cp "$FAKE_JOURNAL_LAUNCHER" "${fake_journal_dir}/journal"
            chmod 0755 "${fake_journal_dir}/journal"
        fi
        echo "$archive_path" >> "${FAKE_PKG_DB}/install.log"
        if ! installed_identity=$(handle_query_pkg "$pkg_type" "$pkg_name"); then
            echo "ERROR:install-failed"
            return 1
        fi
        if [ "$installed_identity" != "INSTALLED $pkg_name $pkg_version $pkg_arch" ]; then
            echo "ERROR:install-failed"
            return 1
        fi
        echo "OK"
        return 0
    fi

    if [ "$pkg_type" = "deb" ]; then
        if ! DEBIAN_FRONTEND=noninteractive apt-get install -y --reinstall --no-remove "$archive_path" >&2; then
            echo "ERROR:install-failed"
            return 1
        fi
    elif [ "$pkg_type" = "rpm" ]; then
        rpm_action=install
        existing_identity=$(handle_query_pkg rpm "$pkg_name") || return 1
        if [ "$existing_identity" = "INSTALLED $pkg_name $pkg_version $pkg_arch" ]; then
            rpm_action=reinstall
        fi
        # The installer verified this local artifact; repository dependency checks remain enabled.
        if ! dnf -y --setopt=localpkg_gpgcheck=False "$rpm_action" "$archive_path" >&2; then
            echo "ERROR:install-failed"
            return 1
        fi
    else
        echo "ERROR:unsupported-pkg-type"
        return 1
    fi
    if ! installed_identity=$(handle_query_pkg "$pkg_type" "$pkg_name"); then
        echo "ERROR:install-failed"
        return 1
    fi
    if [ "$installed_identity" != "INSTALLED $pkg_name $pkg_version $pkg_arch" ]; then
        echo "ERROR:install-failed"
        return 1
    fi
    echo "OK"
}

handle_remove_pkg() {
    pkg_type="$1"
    pkg_name="$2"
    validate_name "$pkg_name"

    if [ -n "$FAKE_PKG_DB" ] && [ -d "$FAKE_PKG_DB" ]; then
        if ! rm -f "${FAKE_PKG_DB}/${pkg_type}/${pkg_name}" \
            || ! printf '%s\n' "$pkg_name" >> "${FAKE_PKG_DB}/remove.log"; then
            echo "ERROR:remove-failed"
            return 1
        fi
        if [ "$pkg_name" = "solstone-journal" ] && [ -n "$FAKE_ROOT" ]; then
            if ! rm -f "${FAKE_ROOT}/usr/bin/journal"; then
                echo "ERROR:remove-failed"
                return 1
            fi
        fi
        echo "OK"
        return 0
    fi

    if [ "$pkg_type" = "deb" ]; then
        if ! dpkg -r "$pkg_name" >&2; then echo "ERROR:remove-failed"; return 1; fi
        echo "OK"
    elif [ "$pkg_type" = "rpm" ]; then
        if ! rpm -e "$pkg_name" >&2; then echo "ERROR:remove-failed"; return 1; fi
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

    receipt_dir=$(dirname "$RECEIPT_FILE")
    if [ -L "$receipt_dir" ] || [ -L "$RECEIPT_FILE" ] || [ -d "$RECEIPT_FILE" ] || { [ -e "$RECEIPT_FILE" ] && [ ! -f "$RECEIPT_FILE" ]; }; then
        echo "ERROR:receipt-dest-invalid"
        return 1
    fi
    if ! mkdir -p "$receipt_dir"; then
        echo "ERROR:receipt-dest-invalid"
        return 1
    fi
    if ! chmod 0755 "$receipt_dir"; then
        echo "ERROR:receipt-dest-invalid"
        return 1
    fi

    tmp_file="${RECEIPT_FILE}.tmp.$$"
    if ! head -c "$len" > "$tmp_file"; then
        rm -f "$tmp_file"
        echo "ERROR:receipt-truncated"
        return 1
    fi
    actual_len=$(wc -c < "$tmp_file" | tr -d '[:space:]')
    if [ "$actual_len" != "$len" ]; then
        rm -f "$tmp_file"
        echo "ERROR:receipt-truncated"
        return 1
    fi
    if ! chmod 0644 "$tmp_file" || ! mv -T "$tmp_file" "$RECEIPT_FILE"; then
        rm -f "$tmp_file"
        echo "ERROR:receipt-dest-invalid"
        return 1
    fi
    echo "OK"
}

handle_read_receipt() {
    if [ ! -e "$RECEIPT_FILE" ] && [ ! -L "$RECEIPT_FILE" ]; then
        return 0
    fi
    if [ -L "$RECEIPT_FILE" ] || [ ! -f "$RECEIPT_FILE" ] || ! cat "$RECEIPT_FILE"; then
        echo "ERROR:receipt-read-failed"
        return 1
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
