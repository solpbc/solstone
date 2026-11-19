# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Build-time rendering for the installer's portable Fluent patterns."""

import json
from pathlib import Path
import re
from collections.abc import Callable

from solstone_platform.refusals import Refusal

ENGLISH_CATALOG = Path("i18n/catalog/source/en-US.ftl")
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
