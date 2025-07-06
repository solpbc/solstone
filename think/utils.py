import os
import re

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
