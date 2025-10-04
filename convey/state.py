from typing import Any, Dict, List, Optional

# Currently selected backend name ("google", "openai" or "anthropic").
# Each message starts fresh with no persistent session.
chat_backend: str = "google"

journal_root: str = ""
occurrences_index: Dict[str, List[Dict[str, Any]]] = {}
occurrences_index_date: Optional[str] = None  # Track when index was built
occurrences_index_days: set = set()  # Track which days are in the index
