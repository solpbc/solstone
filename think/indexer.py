import json
import os
import re
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from usearch.index import Index

TOP_KEY = "__top__"
INDEX_DIR = "index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
_EMBED_DIM = _EMBEDDER.get_sentence_embedding_dimension()


class SemanticChunker:
    """Chunk text based on semantic similarity between sentence groups."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        self.model = SentenceTransformer(model_name)

    def chunk_by_semantic_similarity(self, text: str, threshold: float = 0.7) -> List[str]:
        sentences = nltk.sent_tokenize(text)

        sentence_groups = []
        for i, _ in enumerate(sentences):
            context = sentences[max(0, i - 1) : min(len(sentences), i + 2)]
            sentence_groups.append(" ".join(context))

        if not sentence_groups:
            return []

        embeddings = self.model.encode(sentence_groups)

        distances: List[float] = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distances.append(1 - similarity)

        breakpoints = [0]
        if distances:
            breakpoint_percentile = np.percentile(distances, 95)
            for i, dist in enumerate(distances):
                if dist > breakpoint_percentile:
                    breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        chunks = []
        for i in range(len(breakpoints) - 1):
            chunk = " ".join(sentences[breakpoints[i] : breakpoints[i + 1]])
            chunks.append(chunk)

        return chunks


_CHUNKER = SemanticChunker(EMBED_MODEL_NAME)

DATE_RE = re.compile(r"\d{8}")
ITEM_RE = re.compile(r"^\s*[-*]\s*(.*)")


def find_day_dirs(journal: str) -> Dict[str, str]:
    days = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def parse_entity_line(line: str) -> Optional[Tuple[str, str, str]]:
    cleaned = line.replace("**", "")
    match = ITEM_RE.match(cleaned)
    if not match:
        return None

    text = match.group(1).strip()
    if ":" not in text:
        return None

    etype, rest = text.split(":", 1)
    rest = rest.strip()
    if " - " in rest:
        name, desc = rest.split(" - ", 1)
    else:
        name, desc = rest, ""

    return etype.strip(), name.strip(), desc.strip()


def parse_entities(path: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    valid_types = {"Person", "Company", "Project", "Tool"}

    file_path = os.path.join(path, "entities.md")
    if not os.path.isfile(file_path):
        return items

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not ITEM_RE.match(line.replace("**", "")):
                continue
            parsed = parse_entity_line(line)
            if not parsed:
                continue
            etype, name, desc = parsed
            if etype not in valid_types:
                continue
            items.append((etype, name, desc))

    return items


def load_cache(journal: str) -> Dict[str, dict]:
    cache_path = os.path.join(journal, "indexer.json")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(journal: str, cache: Dict[str, dict]) -> None:
    cache_path = os.path.join(journal, "indexer.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def build_entities(cache: Dict[str, dict]) -> Dict[str, Dict[str, dict]]:
    index: Dict[str, Dict[str, dict]] = {}
    for key, info in cache.items():
        is_top = info.get("top", False)
        for etype, name, desc in info.get("entries", []):
            type_map = index.setdefault(etype, {})
            entry = type_map.setdefault(name, {"dates": [], "descriptions": {}, "top": False})
            if is_top:
                entry["top"] = True
                if desc:
                    entry["top_desc"] = desc
            else:
                if key not in entry["dates"]:
                    entry["dates"].append(key)
                if desc:
                    entry["descriptions"][key] = desc

    for type_map in index.values():
        for info in type_map.values():
            if info.get("top"):
                info["primary"] = info.get("top_desc", "")
            elif info["descriptions"]:
                latest = max(info["descriptions"].keys())
                info["primary"] = info["descriptions"].get(latest, "")
            else:
                info["primary"] = ""
            info.pop("top_desc", None)

    return index


def scan_entities(journal: str, cache) -> Dict[str, Dict[str, dict]]:
    days = find_day_dirs(journal)
    changed = False

    # handle top entities file in parent directory
    top_path = os.path.join(journal, "entities.md")
    if os.path.isfile(top_path):
        mtime = int(os.path.getmtime(top_path))
        info = cache.get(TOP_KEY)
        if info is None or info.get("mtime") != mtime:
            cache[TOP_KEY] = {
                "file": os.path.relpath(top_path, journal),
                "mtime": mtime,
                "entries": parse_entities(journal),
                "top": True,
            }
            changed = True
    elif TOP_KEY in cache:
        del cache[TOP_KEY]
        changed = True

    # remove days no longer present (ignore top key)
    for day in list(cache.keys()):
        if day == TOP_KEY:
            continue
        if day not in days:
            del cache[day]
            changed = True

    for day, path in days.items():
        md_path = os.path.join(path, "entities.md")
        if not os.path.isfile(md_path):
            if day in cache:
                del cache[day]
                changed = True
            continue

        mtime = int(os.path.getmtime(md_path))
        day_info = cache.get(day)
        if day_info is None or day_info.get("mtime") != mtime:
            entries = parse_entities(path)
            cache[day] = {
                "file": os.path.relpath(md_path, journal),
                "mtime": mtime,
                "entries": entries,
            }
            changed = True

    return changed


def get_entities(journal: str) -> Dict[str, Dict[str, dict]]:
    cache = load_cache(journal)
    if scan_entities(journal, cache):
        save_cache(journal, cache)

    return build_entities(cache)


def get_ponder_index(journal: str) -> Tuple[Index, dict, str, str]:
    """Return the USearch index and metadata for ponder files."""
    db_path = os.path.join(journal, INDEX_DIR)
    os.makedirs(db_path, exist_ok=True)
    index_path = os.path.join(db_path, "ponders.usearch")
    meta_path = os.path.join(db_path, "ponders_meta.json")

    if os.path.isfile(index_path):
        index = Index.restore(index_path, view=False)
    else:
        index = Index(ndim=_EMBED_DIM, metric="cos", dtype="f32")

    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"paths": {}, "info": {}, "next_id": 1}

    return index, meta, index_path, meta_path


def find_ponder_files(journal: str) -> Dict[str, str]:
    """Map relative ponder file path to full path."""
    files: Dict[str, str] = {}
    for day, day_path in find_day_dirs(journal).items():
        for name in os.listdir(day_path):
            if name.startswith("ponder_") and name.endswith(".md"):
                rel = os.path.join(day, name)
                files[rel] = os.path.join(day_path, name)
    return files


def scan_ponders(journal: str, cache: Dict[str, dict]) -> bool:
    """Index ponder markdown files into USearch in semantic chunks if they changed."""
    index, meta, index_path, meta_path = get_ponder_index(journal)
    p_cache = cache.setdefault("ponders", {})
    files = find_ponder_files(journal)
    total = len(files)
    if total:
        print(f"\nIndexing {total} ponder files...")
    changed = False

    next_id = meta.get("next_id", 1)
    path_map = meta.setdefault("paths", {})
    info_map = meta.setdefault("info", {})

    for idx, (rel, path) in enumerate(files.items(), 1):
        if total:
            print(f"[{idx}/{total}] {rel}")
        mtime = int(os.path.getmtime(path))
        if p_cache.get(rel) != mtime:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # remove old entries for this file
            keys = path_map.get(rel, [])
            for k in keys:
                index.remove(k)
                info_map.pop(str(k), None)

            chunks = _CHUNKER.chunk_by_semantic_similarity(text)
            print(f"  chunked into {len(chunks)} segments")
            embeddings = _EMBEDDER.encode(chunks)
            keys = []
            for i, emb in enumerate(embeddings):
                key = next_id
                next_id += 1
                index.add(key, emb)
                info_map[str(key)] = {
                    "rel": rel,
                    "day": os.path.basename(os.path.dirname(path)),
                    "ponder": os.path.basename(path),
                    "chunk": i,
                    "text": chunks[i],
                }
                keys.append(key)
            path_map[rel] = keys
            p_cache[rel] = mtime
            changed = True

    for rel in list(p_cache.keys()):
        if rel not in files:
            keys = path_map.pop(rel, [])
            for k in keys:
                index.remove(k)
                info_map.pop(str(k), None)
            del p_cache[rel]
            changed = True

    meta["next_id"] = next_id
    if changed:
        print("Saving updated index...")
        index.save(index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return changed


def search_ponders(journal: str, query: str, n_results: int = 5) -> List[Dict[str, str]]:
    """Search the ponder chunk index and return results."""
    index, meta, _, _ = get_ponder_index(journal)
    embedding = _EMBEDDER.encode(query)
    matches = index.search(embedding, n_results)
    info_map = meta.get("info", {})
    results = []
    for key, dist in zip(matches.keys, matches.distances):
        entry = info_map.get(str(int(key)))
        if not entry:
            continue
        results.append(
            {
                "id": entry["rel"],
                "text": entry.get("text", ""),
                "metadata": {
                    "day": entry["day"],
                    "ponder": entry["ponder"],
                    "chunk": entry.get("chunk", 0),
                },
                "distance": float(dist),
            }
        )
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
        changed |= scan_ponders(args.journal, cache)
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
            snippet = r["text"].splitlines()[0][:80]
            print(f"{idx}. {meta.get('day')} {meta.get('ponder')}: {snippet}")


if __name__ == "__main__":
    main()
