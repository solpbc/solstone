# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Build-time rendering for the installer's portable Fluent patterns."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from collections.abc import Callable

from solstone_platform.refusals import Refusal

ENGLISH_CATALOG = Path("i18n/catalog/source/en-US.ftl")
TRANSLATIONS_DIR = Path("i18n/catalog/translations")
SOURCE_LOCK = Path("i18n/catalog/source-lock.json")
LOCALE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
ID = r"installer-[a-z0-9]+(?:-[a-z0-9]+)*"
DEFINITION = re.compile(rf"({ID}) = (.+)")
PLACEABLE = re.compile(r'\{ (?:\$arg([1-9][0-9]*)|"((?:[^"\\]|\\["\\]|\\u[0-9a-fA-F]{4})*)") \}')
BRACE = re.compile(r"[{}]")
ESCAPE = re.compile(r'\\(["\\]|u[0-9a-fA-F]{4})')
# Each reference names its source context: shell single (s), double (d),
# parameter default (b), trap (t), or printf single/double (q/p). The other
# contexts are AWK (a), help (h), JSON printf (j), JSON notice (n), and
# the prerequisite subject whose prefix chooses grammar (g).
REFERENCE = re.compile(rf"@\{{({ID}):([abdghjnpqst])(?: (\[.*?\]))?\}}@")


def invalid(detail: str) -> Refusal:
    return Refusal("installer-catalog-invalid", detail)


def parse_pattern(value: str, location: str) -> list[str | int]:
    pieces: list[str | int] = []
    index = 0
    while index < len(value):
        brace = BRACE.search(value, index)
        if brace is None:
            pieces.append(value[index:])
            break
        if brace.start() > index:
            pieces.append(value[index:brace.start()])
        match = PLACEABLE.match(value, brace.start())
        if match is None:
            raise invalid(f"{location}: unsupported Fluent placeable")
        argument, literal = match.groups()
        if argument:
            pieces.append(int(argument))
        else:
            pieces.append(ESCAPE.sub(
                lambda m: chr(int(m[1][1:], 16)) if m[1].startswith("u") else m[1],
                literal,
            ))
        index = match.end()
    return pieces


def read_catalog(path: Path) -> dict[str, list[str | int]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as err:
        raise invalid(f"cannot read {path}: {err}") from err
    catalog = {}
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        location = f"{path}:{number}"
        match = DEFINITION.fullmatch(line)
        if match is None:
            raise invalid(f"{location}: expected a Fluent message definition")
        key, value = match.groups()
        if key in catalog:
            raise invalid(f"{location}: duplicate message {key}")
        catalog[key] = parse_pattern(value, location)
    return catalog


_PSEUDO = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "ÅƁÇÐËƑĠĦÏĴĶĻḾÑÖƤǪŔŠŦÜṼŴẊŸŽåƀçðëƒġħïĵķļḿñöƥǫŕšŧüṽŵẋÿž",
)


def pseudo(text: str) -> str:
    vowels = "ÅËÏÖÜåëïöü"
    return "".join(char * 2 if char in vowels else char for char in text.translate(_PSEUDO))


def double_quoted(text: str) -> str:
    return re.sub(r'(["$\\' + chr(96) + r'])', r'\\\1', text)


def single_quoted(text: str) -> str:
    return text.replace("'", """'"'"'""")


def json_content(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def printf_content(text: str, quote: Callable[[str], str]) -> str:
    # The original format escapes, such as \n, are synthesized here.
    # A percent sign in prose is always literal.
    return r"\n".join(quote(line.replace("\\", "\\\\").replace("%", "%%")) for line in text.split("\n"))


def escape(text: str, context: str, key: str) -> str:
    if any((ord(char) < 32 and char not in "\n\t") or 0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise invalid(f"{key}: unsupported control character")
    if context not in {"p", "q"} and "\n" in text:
        raise invalid(f"{key}: a message in this context must fit on one line")
    if context == "h":
        return text
    if context == "b":
        return double_quoted(text).replace("}", r"\}")
    if context == "a":
        if "duplicate-key" in text:
            raise invalid(f"{key}: contains the manifest parser's duplicate-key token")
        return json_content(text)
    if context == "j":
        return printf_content(json_content(text), single_quoted)
    # These notices are interpolated directly into a JSON string. Their
    # existing bindings are architecture and component tokens or commands.
    if context == "n" and any(char in text for char in '"\\\t'):
        raise invalid(f"{key}: unsupported character in a JSON notice")
    if context == "p":
        return printf_content(text, double_quoted)
    if context == "q":
        return printf_content(text, single_quoted)
    if context == "s":
        return single_quoted(text)
    if context == "t":
        return single_quoted(double_quoted(text))
    return double_quoted(text)


def render_installer_strings(
    source: str, catalog: dict[str, list[str | int]], *, pseudo_locale: bool, require_all: bool,
) -> str:
    used = set()

    def replace(match: re.Match) -> str:
        key, context, raw_args = match.groups()
        if key not in catalog:
            raise invalid(f"missing message {key}")
        try:
            args = json.loads(raw_args) if raw_args else []
        except ValueError as err:
            raise invalid(f"{key}: invalid source arguments") from err
        pieces = catalog[key]
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise invalid(f"{key}: source arguments must be strings")
        if sorted(piece for piece in pieces if isinstance(piece, int)) != list(range(1, len(args) + 1)):
            raise invalid(f"{key}: message must use each source argument once")
        used.add(key)
        rendered = "".join(
            args[piece - 1] if isinstance(piece, int)
            else escape(pseudo(piece) if pseudo_locale else piece, context, key)
            for piece in pieces
        )
        if context == "h" and rendered == "SOLSTONE_HELP":
            raise invalid(f"{key}: help text matches the heredoc delimiter")
        if context == "g" and not rendered.startswith("Neither"):
            raise invalid(f"{key}: missing the shell's Neither prefix")
        return rendered

    result = REFERENCE.sub(replace, source)
    if "@{installer-" in result:
        raise invalid("malformed installer message reference")
    if require_all and used != catalog.keys():
        raise invalid(f"unreferenced messages: {', '.join(sorted(catalog.keys() - used))}")
    return result


def validate_locale_tag(locale: str) -> str:
    if not LOCALE_TAG.fullmatch(locale):
        raise invalid(f"invalid locale tag: {locale}")
    return locale


def read_raw_messages(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as err:
        raise invalid(f"cannot read {path}: {err}") from err
    messages: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        location = f"{path}:{number}"
        match = DEFINITION.fullmatch(line)
        if match is None:
            raise invalid(f"{location}: expected a Fluent message definition")
        key, value = match.groups()
        if key in messages:
            raise invalid(f"{location}: duplicate message {key}")
        parse_pattern(value, location)
        messages[key] = value
    return messages


def hash_message_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_locale_catalog(
    repo_root: Path, locale: str | None = None
) -> dict[str, list[str | int]]:
    english = read_catalog(repo_root / ENGLISH_CATALOG)
    if locale is None or locale == "en-US":
        return english
    validate_locale_tag(locale)
    translation_path = repo_root / TRANSLATIONS_DIR / f"{locale}.ftl"
    if not translation_path.is_file():
        return english
    translated = read_catalog(translation_path)
    unknown = sorted(translated.keys() - english.keys())
    if unknown:
        raise invalid(f"{translation_path}: unknown message IDs: {', '.join(unknown)}")
    merged = dict(english)
    for key, pieces in translated.items():
        en_args = sorted(p for p in english[key] if isinstance(p, int))
        tr_args = sorted(p for p in pieces if isinstance(p, int))
        if tr_args != en_args:
            raise invalid(f"{translation_path}: {key}: argument mismatch vs en-US")
        merged[key] = pieces
    return merged


def load_source_lock(lock_path: Path) -> dict:
    if not lock_path.is_file():
        return {"schema_version": 1, "source_locale": "en-US", "messages": {}}
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("messages"), dict):
        raise invalid(f"invalid source-lock schema: {lock_path}")
    return data


def update_source_lock(
    repo_root: Path,
    *,
    acknowledge_locale: str | None = None,
    ids: set[str] | None = None,
) -> dict:
    en_messages = read_raw_messages(repo_root / ENGLISH_CATALOG)
    lock_path = repo_root / SOURCE_LOCK
    existing = load_source_lock(lock_path)
    existing_msgs = existing.get("messages", {})

    tr_messages: dict[str, str] = {}
    if acknowledge_locale is not None:
        validate_locale_tag(acknowledge_locale)
        if acknowledge_locale == "en-US":
            raise invalid("cannot acknowledge source locale en-US")
        tr_path = repo_root / TRANSLATIONS_DIR / f"{acknowledge_locale}.ftl"
        if tr_path.is_file():
            tr_messages = read_raw_messages(tr_path)
            unknown = sorted(tr_messages.keys() - en_messages.keys())
            if unknown:
                raise invalid(f"{tr_path}: unknown message IDs: {', '.join(unknown)}")
        if ids is not None:
            bad = sorted(ids - set(tr_messages.keys()))
            if bad:
                raise invalid(
                    f"cannot acknowledge {acknowledge_locale} IDs absent from {tr_path}: {', '.join(bad)}"
                )
        ack_targets = set(tr_messages.keys()) if ids is None else ids
    else:
        ack_targets = set()

    out_messages: dict[str, dict] = {}
    for key in sorted(en_messages):
        src_hash = hash_message_value(en_messages[key])
        prev = existing_msgs.get(key, {})
        prev_tr = dict(prev.get("translations", {})) if isinstance(prev, dict) else {}
        if acknowledge_locale is not None and key in ack_targets:
            prev_tr[acknowledge_locale] = src_hash
        out_messages[key] = {
            "source": src_hash,
            "translations": dict(sorted(prev_tr.items())),
        }

    result = {
        "schema_version": 1,
        "source_locale": "en-US",
        "messages": out_messages,
    }
    lock_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def locale_status(repo_root: Path, locale: str, ids: set[str] | None = None) -> dict:
    validate_locale_tag(locale)
    en_messages = read_raw_messages(repo_root / ENGLISH_CATALOG)
    lock_data = load_source_lock(repo_root / SOURCE_LOCK)
    lock_msgs = lock_data.get("messages", {})
    tr_path = repo_root / TRANSLATIONS_DIR / f"{locale}.ftl"
    tr_messages = read_raw_messages(tr_path) if tr_path.is_file() else {}
    target_ids = sorted(en_messages.keys() if ids is None else (ids & en_messages.keys()))

    fresh: list[str] = []
    stale: list[str] = []
    missing: list[str] = []
    for key in target_ids:
        src_hash = hash_message_value(en_messages[key])
        if key not in tr_messages:
            missing.append(key)
            continue
        ack = lock_msgs.get(key, {}).get("translations", {}).get(locale)
        if ack == src_hash:
            fresh.append(key)
        else:
            stale.append(key)
    return {
        "locale": locale,
        "fresh": fresh,
        "stale": stale,
        "missing": missing,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage installer string catalog locks and draft status.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_lock = sub.add_parser("lock", help="Refresh i18n/catalog/source-lock.json")
    p_lock.add_argument("--root", type=Path, default=Path("."))
    p_lock.add_argument("--acknowledge", default=None)
    p_lock.add_argument("--ids-file", type=Path, default=None)

    p_status = sub.add_parser("status", help="Inspect draft locale freshness")
    p_status.add_argument("--root", type=Path, default=Path("."))
    p_status.add_argument("--locale", required=True)
    p_status.add_argument("--ids-file", type=Path, default=None)
    p_status.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    ids: set[str] | None = None
    if args.ids_file is not None:
        ids = {
            line.strip()
            for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    if args.command == "lock":
        update_source_lock(root, acknowledge_locale=args.acknowledge, ids=ids)
        return 0
    report = locale_status(root, args.locale, ids=ids)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"{report['locale']}: fresh={len(report['fresh'])} "
            f"stale={len(report['stale'])} missing={len(report['missing'])}"
        )
    return 1 if report["stale"] else 0


if __name__ == "__main__":
    sys.exit(_main())
