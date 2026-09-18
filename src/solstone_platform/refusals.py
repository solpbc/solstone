# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Named refusal vocabulary and exception definitions."""

from typing import Optional


class Refusal(Exception):
    """Base exception for all named platform refusals."""

    def __init__(self, name: str, detail: Optional[str] = None):
        self.name = name
        self.detail = detail
        if detail:
            super().__init__(f"{name}: {detail}")
        else:
            super().__init__(name)


# Frozen named refusal strings
DUPLICATE_KEY = "duplicate-key"
SCHEMA_INVALID = "schema-invalid"
RELEASE_COHERENCE = "release-coherence"
SIGNATURE_PIN_MISMATCH = "signature-pin-mismatch"
UNSAFE_FILENAME = "unsafe-filename"
URL_USERINFO = "url-userinfo"
URL_FRAGMENT = "url-fragment"
URL_INSECURE = "url-insecure"
LANE_INVALID = "lane-invalid"
VERSION_INVALID = "version-invalid"
DESKTOP_AARCH64 = "desktop-aarch64"
INCOMPLETE_VARIANTS = "incomplete-variants"

ARCHIVE_ABSOLUTE_PATH = "archive-absolute-path"
ARCHIVE_PARENT_TRAVERSAL = "archive-parent-traversal"
ARCHIVE_SYMLINK_ESCAPE = "archive-symlink-escape"
ARCHIVE_HARDLINK_ESCAPE = "archive-hardlink-escape"
ARCHIVE_SYMLINK_THEN_CHILD = "archive-symlink-then-child"
ARCHIVE_DUPLICATE_MEMBER = "archive-duplicate-member"
ARCHIVE_DEVICE = "archive-device"
ARCHIVE_FIFO = "archive-fifo"
ARCHIVE_PACKAGE_SCRIPT = "archive-package-script"
ARCHIVE_PACKAGE_TRIGGER = "archive-package-trigger"

PRODUCTION_UNAVAILABLE = "production-unavailable"
PASSPHRASE_SOURCE_INVALID = "passphrase-source-invalid"
FIXTURE_KEY_REFUSED = "fixture-key-refused"
PIN_MISMATCH = "pin-mismatch"
PUBLISH_INDETERMINATE = "publish-indeterminate"
ROLLBACK_REFUSED = "rollback-refused"
SAME_VERSION_DIFFERENT_BYTES = "same-version-different-bytes"
GENERATION_FAILED = "generation-failed"
STALE_LIST = "stale-list"

HTTP_3XX = "http-3xx"
HTTP_403 = "http-403"
HTTP_404 = "http-404"
HTTP_412 = "http-412"
HTTP_429 = "http-429"
HTTP_5XX = "http-5xx"
HTTP_TIMEOUT = "http-timeout"
HTTP_TRUNCATED_2XX = "http-truncated-2xx"
HTTP_ETAG_MALFORMED = "http-etag-malformed"
HTTP_LOST_BEFORE_COMMIT = "http-lost-before-commit"
HTTP_LOST_AFTER_COMMIT = "http-lost-after-commit"

SOURCE_CHANGED_DURING_CAPTURE = "source-changed-during-capture"
CAPTURE_SYMLINK_TRAVERSAL = "capture-symlink-traversal"
CAPTURE_SPECIAL_FILE = "capture-special-file"
CAPTURE_HARDLINK_ALIAS = "capture-hardlink-alias"
CAPTURE_PATH_ESCAPE = "capture-path-escape"
CAPTURE_DUPLICATE_ENTRY = "capture-duplicate-entry"
CAPTURE_FILE_DISAPPEARED = "capture-file-disappeared"
CAPTURE_DEPTH_EXCEEDED = "capture-depth-exceeded"
CAPTURE_ENTRY_LIMIT_EXCEEDED = "capture-entry-limit-exceeded"
CAPTURE_SIZE_LIMIT_EXCEEDED = "capture-size-limit-exceeded"
CAPTURE_READ_INTERRUPTED = "capture-read-interrupted"

ALL_REFUSALS = {
    DUPLICATE_KEY,
    SCHEMA_INVALID,
    RELEASE_COHERENCE,
    SIGNATURE_PIN_MISMATCH,
    UNSAFE_FILENAME,
    URL_USERINFO,
    URL_FRAGMENT,
    URL_INSECURE,
    LANE_INVALID,
    VERSION_INVALID,
    DESKTOP_AARCH64,
    INCOMPLETE_VARIANTS,
    ARCHIVE_ABSOLUTE_PATH,
    ARCHIVE_PARENT_TRAVERSAL,
    ARCHIVE_SYMLINK_ESCAPE,
    ARCHIVE_HARDLINK_ESCAPE,
    ARCHIVE_SYMLINK_THEN_CHILD,
    ARCHIVE_DUPLICATE_MEMBER,
    ARCHIVE_DEVICE,
    ARCHIVE_FIFO,
    ARCHIVE_PACKAGE_SCRIPT,
    ARCHIVE_PACKAGE_TRIGGER,
    PRODUCTION_UNAVAILABLE,
    PASSPHRASE_SOURCE_INVALID,
    FIXTURE_KEY_REFUSED,
    PIN_MISMATCH,
    PUBLISH_INDETERMINATE,
    ROLLBACK_REFUSED,
    SAME_VERSION_DIFFERENT_BYTES,
    GENERATION_FAILED,
    STALE_LIST,
    HTTP_3XX,
    HTTP_403,
    HTTP_404,
    HTTP_412,
    HTTP_429,
    HTTP_5XX,
    HTTP_TIMEOUT,
    HTTP_TRUNCATED_2XX,
    HTTP_ETAG_MALFORMED,
    HTTP_LOST_BEFORE_COMMIT,
    HTTP_LOST_AFTER_COMMIT,
    SOURCE_CHANGED_DURING_CAPTURE,
    CAPTURE_SYMLINK_TRAVERSAL,
    CAPTURE_SPECIAL_FILE,
    CAPTURE_HARDLINK_ALIAS,
    CAPTURE_PATH_ESCAPE,
    CAPTURE_DUPLICATE_ENTRY,
    CAPTURE_FILE_DISAPPEARED,
    CAPTURE_DEPTH_EXCEEDED,
    CAPTURE_ENTRY_LIMIT_EXCEEDED,
    CAPTURE_SIZE_LIMIT_EXCEEDED,
    CAPTURE_READ_INTERRUPTED,
}
