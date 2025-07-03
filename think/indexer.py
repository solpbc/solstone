"""Utilities for indexing ponder markdown files."""

import glob
import os
import re
import sqlite3
from typing import Dict, List, Tuple

import nltk

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


def get_ponder_index(journal: str) -> Tuple[sqlite3.Connection, str]:
    """Return SQLite connection for the ponder sentence index."""
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
    return conn, db_path


def find_ponder_files(journal: str) -> Dict[str, str]:
    """Map relative ponder file path to full path."""
    files: Dict[str, str] = {}
    for day, day_path in find_day_dirs(journal).items():
        for name in os.listdir(day_path):
            base, ext = os.path.splitext(name)
            if ext in {".md", ".json"} and base in PONDER_BASENAMES:
                rel = os.path.join(day, name)
                files[rel] = os.path.join(day_path, name)
    return files


def scan_ponders(journal: str, cache: Dict[str, dict], verbose: bool = False) -> bool:
    """Index sentences from ponder markdown files into SQLite."""
    conn, _ = get_ponder_index(journal)
    p_cache = cache.setdefault("ponders", {})
    files = find_ponder_files(journal)
    total = len(files)
    if total:
        print(f"\nIndexing {total} ponder files...")
    changed = False

    for idx, (rel, path) in enumerate(files.items(), 1):
        if total:
            print(f"[{idx}/{total}] {rel}")
        mtime = int(os.path.getmtime(path))
        if p_cache.get(rel) != mtime:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            conn.execute("DELETE FROM sentences WHERE path=?", (rel,))
            sentences = split_sentences(text)
            if verbose:
                print(f"  indexed {len(sentences)} sentences")
            day, ponder = rel.split(os.sep, 1)
            for pos, sentence in enumerate(sentences):
                conn.execute(
                    (
                        "INSERT INTO sentences(sentence, path, day, ponder, position) "
                        "VALUES (?, ?, ?, ?, ?)"
                    ),
                    (sentence, rel, day, ponder, pos),
                )
            conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel, mtime))
            p_cache[rel] = mtime
            changed = True

    for rel in list(p_cache.keys()):
        if rel not in files:
            conn.execute("DELETE FROM sentences WHERE path=?", (rel,))
            conn.execute("DELETE FROM files WHERE path=?", (rel,))
            del p_cache[rel]
            changed = True

    if changed:
        conn.commit()
    conn.close()
    return changed


def search_ponders(journal: str, query: str, n_results: int = 5) -> List[Dict[str, str]]:
    """Search the ponder sentence index and return results."""
    conn, _ = get_ponder_index(journal)
    cursor = conn.execute(
        """
        SELECT sentence, path, day, ponder, position, bm25(sentences) as rank
        FROM sentences WHERE sentences MATCH ? ORDER BY rank LIMIT ?
        """,
        (query, n_results),
    )
    results = []
    for sentence, path, day, ponder, pos, rank in cursor.fetchall():
        results.append(
            {
                "id": path,
                "text": sentence,
                "metadata": {"day": day, "ponder": ponder, "index": pos},
                "score": rank,
            }
        )
    conn.close()
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Index ponder markdown files")
    parser.add_argument("journal", help="Path to the journal directory")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan journal and update the index before searching",
    )
    args = parser.parse_args()

    cache = load_cache(args.journal)
    if args.rescan:
        changed = scan_entities(args.journal, cache)
        changed |= scan_ponders(args.journal, cache, verbose=True)
        if changed:
            save_cache(args.journal, cache)

    while True:
        try:
            query = input("search> ").strip()
        except EOFError:
            break
        if not query:
            break
        results = search_ponders(args.journal, query, 5)
        for idx, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            snippet = r["text"]
            print(f"{idx}. {meta.get('day')} {meta.get('ponder')}: {snippet}")


if __name__ == "__main__":
    main()
