# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Duplicate-key rejecting parser and RFC 8785 JSON Canonicalization Scheme (JCS) serializer."""

import json
from typing import Any

from solstone_platform.refusals import DUPLICATE_KEY, SCHEMA_INVALID, Refusal


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

    try:
        return json.loads(text, object_pairs_hook=_rejecting_pairs_hook)
    except json.JSONDecodeError as err:
        raise Refusal(SCHEMA_INVALID, f"malformed JSON: {err}") from err


def _canonicalize_value(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        raise Refusal(SCHEMA_INVALID, "floating-point numbers are not allowed in canonical platform manifests")
    if isinstance(val, str):
        # RFC 8785 string escaping using json.dumps with ensure_ascii=False
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    if isinstance(val, list):
        items = [_canonicalize_value(item) for item in val]
        return "[" + ",".join(items) + "]"
    if isinstance(val, dict):
        # Lexicographical key sort by UTF-16 code units (equivalent to Unicode code point sort for non-surrogate)
        sorted_keys = sorted(val.keys())
        entries = []
        for k in sorted_keys:
            escaped_key = json.dumps(k, ensure_ascii=False, separators=(",", ":"))
            entries.append(f"{escaped_key}:{_canonicalize_value(val[k])}")
        return "{" + ",".join(entries) + "}"
    raise Refusal(SCHEMA_INVALID, f"unsupported type in canonicalization: {type(val)}")


def canonical_json_bytes(obj: Any) -> bytes:
    """Produce RFC 8785 canonical UTF-8 JSON bytes with round-trip verification."""
    canonical_str = _canonicalize_value(obj)
    canonical_bytes = canonical_str.encode("utf-8")

    # Round-trip verification: reparse and re-canonicalize must produce identical bytes
    reparsed = parse_json_strict(canonical_bytes)
    re_canonical_bytes = _canonicalize_value(reparsed).encode("utf-8")
    if canonical_bytes != re_canonical_bytes:
        raise Refusal(SCHEMA_INVALID, "canonical JSON round-trip mismatch")

    return canonical_bytes
