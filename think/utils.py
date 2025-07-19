import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from timefhuman import timefhuman

DATE_RE = re.compile(r"\d{8}")

# Colors used for topic visualization in the dream app
CATEGORY_COLORS = [
    "#007bff",
    "#28a745",
    "#17a2b8",
    "#ffc107",
    "#6f42c1",
    "#fd7e14",
    "#e83e8c",
    "#6c757d",
    "#20c997",
    "#ff5722",
    "#9c27b0",
    "#795548",
]


def day_path(day: str) -> str:
    """Return absolute path for *day* from ``JOURNAL_PATH`` environment variable."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")
    if not DATE_RE.fullmatch(day):
        raise ValueError("day must be in YYYYMMDD format")
    return os.path.join(journal, day)


def _append_task_log(dir_path: str | Path, message: str) -> None:
    """Append ``message`` to ``task_log.txt`` inside ``dir_path``."""
    path = Path(dir_path) / "task_log.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())}\t{message}\n")
    except Exception:
        pass


def day_log(day: str, message: str) -> None:
    """Convenience wrapper to log message for ``day``."""
    _append_task_log(day_path(day), message)


def journal_log(message: str) -> None:
    """Append ``message`` to the journal's ``task_log.txt``."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        _append_task_log(journal, message)


def setup_cli(parser: argparse.ArgumentParser, *, parse_known: bool = False):
    """Parse command line arguments and configure logging.

    The parser will be extended with ``-v``/``--verbose`` and ``-d``/``--debug`` flags. Environment
    variables from ``.env`` are loaded and ``JOURNAL_PATH`` is validated. The
    parsed arguments are returned. If ``parse_known`` is ``True`` a tuple of
    ``(args, extra)`` is returned using :func:`argparse.ArgumentParser.parse_known_args`.
    """

    load_dotenv()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    if parse_known:
        args, extra = parser.parse_known_args()
    else:
        args = parser.parse_args()
        extra = None

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level)

    journal = os.getenv("JOURNAL_PATH")
    if not journal or not os.path.isdir(journal):
        parser.error("JOURNAL_PATH not set or invalid")

    return (args, extra) if parse_known else args


def get_topics() -> dict[str, dict[str, object]]:
    """Return available topics with metadata.

    Each key is the topic name and the value contains the ``path`` to the
    ``.txt`` file, the assigned ``color`` from :data:`CATEGORY_COLORS`, and the
    file ``mtime``.
    """

    topics_dir = Path(__file__).parent / "topics"
    topics: dict[str, dict[str, object]] = {}
    for idx, txt_path in enumerate(sorted(topics_dir.glob("*.txt"))):
        name = txt_path.stem
        color = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
        mtime = int(txt_path.stat().st_mtime)
        topics[name] = {
            "path": str(txt_path),
            "color": color,
            "mtime": mtime,
        }
    return topics


def parse_time_range(text: str) -> Optional[tuple[str, str, str]]:
    """Return ``(day, start, end)`` from a natural language time range.

    Parameters
    ----------
    text:
        Natural language description of a time range.

    Returns
    -------
    tuple[str, str, str] | None
        ``(day, start, end)`` if a single range within one day was detected.
        ``day`` is ``YYYYMMDD`` and ``start``/``end`` are ``HHMMSS``. ``None``
        if parsing fails.
    """

    try:
        result = timefhuman(text)
    except Exception as exc:  # pragma: no cover - unexpected library failure
        logging.info("timefhuman failed for %s: %s", text, exc)
        return None

    logging.debug("timefhuman(%s) -> %r", text, result)

    if len(result) != 1:
        logging.info("timefhuman did not return a single expression for %s", text)
        return None

    range_item = result[0]
    if not isinstance(range_item, tuple) or len(range_item) != 2:
        logging.info("Expected a range from %s but got %r", text, range_item)
        return None

    start_dt, end_dt = range_item
    if start_dt.date() != end_dt.date():
        logging.info("Range must be within a single day: %s -> %s", start_dt, end_dt)
        return None

    day = start_dt.strftime("%Y%m%d")
    start = start_dt.strftime("%H%M%S")
    end = end_dt.strftime("%H%M%S")
    return day, start, end
