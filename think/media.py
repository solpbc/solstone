# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Media format registry - single source of truth for extensions, MIME types, and kind."""

FORMATS = [
    (".flac", "audio/flac", "audio"),
    (".opus", "audio/opus", "audio"),
    (".ogg", "audio/ogg", "audio"),
    (".m4a", "audio/mp4", "audio"),
    (".mp3", "audio/mpeg", "audio"),
    (".wav", "audio/wav", "audio"),
    (".webm", "video/webm", "video"),
    (".mp4", "video/mp4", "video"),
    (".mov", "video/quicktime", "video"),
]

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    ext for ext, _, kind in FORMATS if kind == "audio"
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    ext for ext, _, kind in FORMATS if kind == "video"
)
MEDIA_EXTENSIONS: frozenset[str] = frozenset(ext for ext, _, _ in FORMATS)
MIME_TYPES: dict[str, str] = {ext: mime for ext, mime, _ in FORMATS}
