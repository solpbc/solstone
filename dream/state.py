from typing import Any, Dict, List

# Cached Gemini chat history. Each entry is a mapping with
# ``role`` ("user" or "bot") and ``text`` keys.
chat_history: List[Dict[str, str]] = []

journal_root: str = ""
entities_index: Dict[str, Dict[str, dict]] = {}
occurrences_index: Dict[str, List[Dict[str, Any]]] = {}
