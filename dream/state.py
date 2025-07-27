from typing import Any, Dict, List, Optional

from think.google import AgentSession

# Cached Gemini chat agent session. ``AgentSession.history`` stores
# the role/content pairs for the conversation.
chat_agent: Optional[AgentSession] = None

journal_root: str = ""
entities_index: Dict[str, Dict[str, dict]] = {}
occurrences_index: Dict[str, List[Dict[str, Any]]] = {}
