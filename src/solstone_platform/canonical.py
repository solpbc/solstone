# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Duplicate-key rejecting parser and RFC 8785 JSON Canonicalization Scheme (JCS) serializer."""

import json
from typing import Any

from solstone_platform.refusals import DUPLICATE_KEY, SCHEMA_INVALID, Refusal


MAX_CANONICAL_DEPTH = 64
MAX_CANONICAL_INTEGER_DIGITS = 20


def _rejecting_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refusal(DUPLICATE_KEY, f"duplicate key '{key}'")
        result[key] = value
    return result


def parse_json_strict(data: bytes | str) -> Any:
    """Parse JSON while rejecting duplicate keys at any depth."""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as err:
            raise Refusal(SCHEMA_INVALID, f"invalid UTF-8: {err}") from err
    else:
        text = data

    def reject_constant(value: str) -> Any:
        raise Refusal(SCHEMA_INVALID, f"non-finite JSON number is not allowed: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_rejecting_pairs_hook,
            parse_constant=reject_constant,
        )
    except Refusal:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as err:
        raise Refusal(SCHEMA_INVALID, f"malformed JSON: {err}") from err


def _utf16_sort_key(value: str) -> bytes:
    try:
        return value.encode("utf-16-be")
    except UnicodeError as err:
        raise Refusal(SCHEMA_INVALID, "canonical JSON strings must not contain lone surrogates") from err


def _canonicalize_value(val: Any, *, depth: int, active: set[int]) -> str:
    if depth > MAX_CANONICAL_DEPTH:
        raise Refusal(SCHEMA_INVALID, f"canonical JSON nesting exceeds {MAX_CANONICAL_DEPTH}")
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int) and not isinstance(val, bool):
        try:
            rendered = str(val)
        except ValueError as err:
            raise Refusal(SCHEMA_INVALID, "canonical JSON integer exceeds supported conversion limit") from err
        digits = rendered[1:] if rendered.startswith("-") else rendered
        if len(digits) > MAX_CANONICAL_INTEGER_DIGITS:
            raise Refusal(
                SCHEMA_INVALID,
                f"canonical JSON integer exceeds {MAX_CANONICAL_INTEGER_DIGITS} digits",
            )
        return rendered
    if isinstance(val, float):
        raise Refusal(SCHEMA_INVALID, "floating-point numbers are not allowed in canonical platform manifests")
    if isinstance(val, str):
        _utf16_sort_key(val)
        try:
            return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError, UnicodeError) as err:
            raise Refusal(SCHEMA_INVALID, f"invalid canonical JSON string: {err}") from err
    if isinstance(val, list):
        identity = id(val)
        if identity in active:
            raise Refusal(SCHEMA_INVALID, "cyclic list in canonical JSON input")
        active.add(identity)
        try:
            items = [_canonicalize_value(item, depth=depth + 1, active=active) for item in val]
            return "[" + ",".join(items) + "]"
        finally:
            active.remove(identity)
    if isinstance(val, dict):
        identity = id(val)
        if identity in active:
            raise Refusal(SCHEMA_INVALID, "cyclic object in canonical JSON input")
        if not all(isinstance(key, str) for key in val):
            raise Refusal(SCHEMA_INVALID, "canonical JSON object keys must be strings")
        active.add(identity)
        try:
            sorted_keys = sorted(val.keys(), key=_utf16_sort_key)
            entries = []
            for key in sorted_keys:
                _utf16_sort_key(key)
                escaped_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                rendered = _canonicalize_value(val[key], depth=depth + 1, active=active)
                entries.append(f"{escaped_key}:{rendered}")
            return "{" + ",".join(entries) + "}"
        finally:
            active.remove(identity)
    raise Refusal(SCHEMA_INVALID, f"unsupported type in canonicalization: {type(val)}")


def canonical_json_bytes(obj: Any) -> bytes:
    """Produce RFC 8785 canonical UTF-8 JSON bytes with round-trip verification."""
    try:
        canonical_str = _canonicalize_value(obj, depth=0, active=set())
        canonical_bytes = canonical_str.encode("utf-8")
    except Refusal:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError) as err:
        raise Refusal(SCHEMA_INVALID, f"canonical JSON serialization failed: {err}") from err

    # Round-trip verification: reparse and re-canonicalize must produce identical bytes
    reparsed = parse_json_strict(canonical_bytes)
    re_canonical_bytes = _canonicalize_value(reparsed, depth=0, active=set()).encode("utf-8")
    if canonical_bytes != re_canonical_bytes:
        raise Refusal(SCHEMA_INVALID, "canonical JSON round-trip mismatch")

    return canonical_bytes
