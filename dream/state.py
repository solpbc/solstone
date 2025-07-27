from typing import Any, Dict, List, Optional

from think.agents import BaseAgentSession

# Cached chat agent session. ``BaseAgentSession.history`` stores
# the role/content pairs for the conversation.
chat_agent: Optional[BaseAgentSession] = None
# Currently selected backend name. ``send_message`` updates this when
# a new backend is requested.
chat_backend: str = "google"

journal_root: str = ""
entities_index: Dict[str, Dict[str, dict]] = {}
occurrences_index: Dict[str, List[Dict[str, Any]]] = {}
