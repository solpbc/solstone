import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

DATE_RE = re.compile(r"\d{8}")


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
