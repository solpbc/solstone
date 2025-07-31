from typing import Any, Dict, List

# Currently selected backend name ("google", "openai" or "anthropic").
# Each message starts fresh with no persistent session.
chat_backend: str = "google"

journal_root: str = ""
occurrences_index: Dict[str, List[Dict[str, Any]]] = {}
