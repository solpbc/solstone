"""Utilities for indexing ponder outputs and occurrences."""

import glob
import json
import os
import re
import sqlite3
from typing import Dict, List, Tuple

import nltk
import sqlite_utils
from dotenv import load_dotenv

from think.utils import journal_log

from .entities import find_day_dirs, load_cache, save_cache, scan_entities

INDEX_DIR = "index"

nltk.download("punkt", quiet=True)


# Sentence indexing helpers -------------------------------------------------

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "ponder")
PROMPT_BASENAMES = [
    os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(PROMPT_DIR, "*.txt"))
]
PONDER_BASENAMES = [f"ponder_{b}" for b in PROMPT_BASENAMES]


def split_sentences(text: str) -> List[str]:
    """Return a list of cleaned sentences from markdown text."""
    cleaned = re.sub(r"^[*-]\s*", "", text, flags=re.MULTILINE)
    sentences = [s.strip() for s in nltk.sent_tokenize(cleaned) if s.strip()]
    return sentences


# Ponder indexing -----------------------------------------------------------


def get_index(journal: str) -> Tuple[sqlite3.Connection, str]:
    """Return SQLite connection for sentence and occurrence indexes."""
    db_dir = os.path.join(journal, INDEX_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "ponders.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS sentences USING fts5(
            sentence, path UNINDEXED, day UNINDEXED, ponder UNINDEXED, position UNINDEXED
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS occurrences USING fts5(
            title, summary, details,
            path UNINDEXED, day UNINDEXED, idx UNINDEXED,
            type UNINDEXED, source UNINDEXED,
            start UNINDEXED, end UNINDEXED,
            work UNINDEXED, participants UNINDEXED
        )
        """
    )
    return conn, db_path


def find_ponder_files(journal: str, exts: Tuple[str, ...] | None = None) -> Dict[str, str]:
    """Map relative ponder file path to full path filtered by ``exts``."""
    files: Dict[str, str] = {}
    exts = exts or (".md", ".json")
    for day, day_path in find_day_dirs(journal).items():
        for name in os.listdir(day_path):
            base, ext = os.path.splitext(name)
            if ext in exts and base in PONDER_BASENAMES:
                rel = os.path.join(day, name)
                files[rel] = os.path.join(day_path, name)
    return files


def _scan_files(
    conn: sqlite3.Connection,
    files: Dict[str, str],
    cache_map: Dict[str, int],
    delete_sql: str,
    index_func,
    verbose: bool = False,
) -> bool:
    changed = False
    total = len(files)
    for idx, (rel, path) in enumerate(files.items(), 1):
        if total:
            print(f"[{idx}/{total}] {rel}")
        mtime = int(os.path.getmtime(path))
        if cache_map.get(rel) != mtime:
            conn.execute(delete_sql, (rel,))
            index_func(conn, rel, path, verbose)
            conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel, mtime))
            cache_map[rel] = mtime
            changed = True

    for rel in list(cache_map.keys()):
        if rel not in files:
            conn.execute(delete_sql, (rel,))
            conn.execute("DELETE FROM files WHERE path=?", (rel,))
            del cache_map[rel]
            changed = True

    return changed


def _index_sentences(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = split_sentences(text)
    if verbose:
        print(f"  indexed {len(sentences)} sentences")
    day, ponder = rel.split(os.sep, 1)
    for pos, sentence in enumerate(sentences):
        conn.execute(
            ("INSERT INTO sentences(sentence, path, day, ponder, position) VALUES (?, ?, ?, ?, ?)"),
            (sentence, rel, day, ponder, pos),
        )


def _index_occurrences(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    occs = data.get("occurrences", []) if isinstance(data, dict) else data
    if verbose:
        print(f"  indexed {len(occs)} occurrences")
    day = rel.split(os.sep, 1)[0]
    for idx, occ in enumerate(occs):
        title = occ.get("title", "")
        summary = occ.get("summary", occ.get("subject", ""))
        details = occ.get("details")
        if not isinstance(details, str):
            try:
                details = json.dumps(details, ensure_ascii=False)
            except Exception:
                details = str(details)
        participants = occ.get("participants")
        if isinstance(participants, list):
            participants = ", ".join(participants)
        conn.execute(
            (
                "INSERT INTO occurrences(title, summary, details, path, day, idx, type, source, start, end, work, participants) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                title,
                summary,
                details or "",
                rel,
                day,
                idx,
                occ.get("type", ""),
                occ.get("source", ""),
                occ.get("start", ""),
                occ.get("end", ""),
                str(occ.get("work", "")),
                participants or "",
            ),
        )


def scan_ponders(journal: str, cache: Dict[str, dict], verbose: bool = False) -> bool:
    """Index sentences from ponder markdown files."""
    conn, _ = get_index(journal)
    p_cache = cache.setdefault("ponders", {})
    files = find_ponder_files(journal, (".md",))
    if files:
        print(f"\nIndexing {len(files)} ponder files...")
    changed = _scan_files(
        conn, files, p_cache, "DELETE FROM sentences WHERE path=?", _index_sentences, verbose
    )
    if changed:
        conn.commit()
    conn.close()
    return changed


def scan_occurrences(journal: str, cache: Dict[str, dict], verbose: bool = False) -> bool:
    """Index occurrence JSON files."""
    conn, _ = get_index(journal)
    o_cache = cache.setdefault("occurrences", {})
    files = find_ponder_files(journal, (".json",))
    if files:
        print(f"\nIndexing {len(files)} occurrence files...")
    changed = _scan_files(
        conn, files, o_cache, "DELETE FROM occurrences WHERE path=?", _index_occurrences, verbose
    )
    if changed:
        conn.commit()
    conn.close()
    return changed


def search_ponders(journal: str, query: str, n_results: int = 5) -> List[Dict[str, str]]:
    """Search the ponder sentence index and return results."""
    conn, _ = get_index(journal)
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)
    cursor = conn.execute(
        f"""
        SELECT sentence, path, day, ponder, position, bm25(sentences) as rank
        FROM sentences WHERE sentences MATCH {quoted} ORDER BY rank LIMIT ?
        """,
        (n_results,),
    )
    results = []
    for sentence, path, day, ponder, pos, rank in cursor.fetchall():
        results.append(
            {
                "id": path,
                "text": sentence,
                "metadata": {
                    "day": day,
                    "ponder": ponder,
                    "path": path,
                    "index": pos,
                },
                "score": rank,
            }
        )
    conn.close()
    return results


def search_occurrences(journal: str, query: str, n_results: int = 5) -> List[Dict[str, str]]:
    """Search the occurrences index and return results."""
    conn, _ = get_index(journal)
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)
    cursor = conn.execute(
        f"""
        SELECT title, summary, details, path, day, idx, type, source, start, end, work, participants, bm25(occurrences) as rank
        FROM occurrences WHERE occurrences MATCH {quoted} ORDER BY rank LIMIT ?
        """,
        (n_results,),
    )
    results = []
    for row in cursor.fetchall():
        (
            title,
            summary,
            details,
            path,
            day,
            idx,
            o_type,
            source,
            start,
            end,
            work,
            participants,
            rank,
        ) = row
        text = title or summary or details
        results.append(
            {
                "id": f"{path}:{idx}",
                "text": text,
                "metadata": {
                    "day": day,
                    "path": path,
                    "index": idx,
                    "type": o_type,
                    "start": start,
                    "end": end,
                    "source": source,
                    "work": work,
                    "participants": participants,
                },
                "score": rank,
            }
        )
    conn.close()
    return results


def _display_search_results(results: List[Dict[str, str]]) -> None:
    """Display search results in a consistent format."""
    for idx, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        snippet = r["text"]
        print(f"{idx}. {meta.get('day')} {meta.get('ponder')}: {snippet}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Index ponder markdown and occurrence files")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan journal and update the index before searching",
    )
    parser.add_argument(
        "-q",
        "--query",
        nargs="?",
        const="",
        help="Run query (interactive mode if no query provided)",
    )

    args = parser.parse_args()

    # Require either --rescan or -q
    if not args.rescan and args.query is None:
        parser.print_help()
        return

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        parser.error("JOURNAL_PATH not set")

    cache = load_cache(journal)
    if args.rescan:
        changed = scan_entities(journal, cache)
        changed |= scan_ponders(journal, cache, verbose=True)
        changed |= scan_occurrences(journal, cache, verbose=True)
        if changed:
            save_cache(journal, cache)
        journal_log("indexer rescan ok")

    # Handle query argument
    if args.query is not None:
        if args.query:
            # Single query mode - run query and exit
            results = search_ponders(journal, args.query, 5)
            _display_search_results(results)
        else:
            # Interactive mode
            while True:
                try:
                    query = input("search> ").strip()
                except EOFError:
                    break
                if not query:
                    break
                results = search_ponders(journal, query, 5)
                _display_search_results(results)


if __name__ == "__main__":
    main()
