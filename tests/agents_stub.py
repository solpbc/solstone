import sys
import types
from types import SimpleNamespace


def install_agents_stub():
    """Install a dummy `agents` package with the pieces used by `think.openai`."""
    agents_stub = types.ModuleType("agents")
    agents_mcp_stub = types.ModuleType("agents.mcp")
    agents_mcp_server_stub = types.ModuleType("agents.mcp.server")
    agents_items_stub = types.ModuleType("agents.items")
    agents_run_stub = types.ModuleType("agents.run")
    agents_model_settings_stub = types.ModuleType("agents.model_settings")

    last_kwargs: dict[str, dict] = {}

    class DummyAgent:
        def __init__(self, *_, **kwargs):
            last_kwargs.update(kwargs)

    class StreamResult:
        def __init__(self, events=None):
            self.events = events or []
            self.final_output = "ok"

        async def stream_events(self):
            for event in self.events:
                yield event

    class DummyRunner:
        called = False
        events_to_stream = []

        @staticmethod
        def run_streamed(agent, input, session=None, run_config=None, max_turns=None):  # noqa: D417
            DummyRunner.called = True
            return StreamResult(DummyRunner.events_to_stream)

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_run_stub.RunConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    agents_model_settings_stub.ModelSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    agents_stub.RunConfig = agents_run_stub.RunConfig
    agents_stub.ModelSettings = agents_model_settings_stub.ModelSettings
    agents_stub.set_default_openai_key = lambda key: None

    class DummySession:
        def __init__(self, *_, **__):
            pass

    agents_stub.SQLiteSession = DummySession

    class DummyConversationSession:
        def __init__(self, conversation_id=None):
            self.conversation_id = conversation_id

    agents_stub.OpenAIConversationsSession = DummyConversationSession

    class DummyMCPServer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    agents_mcp_server_stub.MCPServerStreamableHttp = lambda **kwargs: DummyMCPServer()

    agents_items_stub.MessageOutputItem = SimpleNamespace
    agents_items_stub.ToolCallItem = SimpleNamespace
    agents_items_stub.ToolCallOutputItem = SimpleNamespace
    agents_items_stub.ReasoningItem = SimpleNamespace

    for key in list(sys.modules.keys()):
        if key == "agents" or key.startswith("agents."):
            sys.modules.pop(key)
    sys.modules["agents"] = agents_stub
    sys.modules["agents.mcp"] = agents_mcp_stub
    sys.modules["agents.mcp.server"] = agents_mcp_server_stub
    sys.modules["agents.items"] = agents_items_stub
    sys.modules["agents.run"] = agents_run_stub
    sys.modules["agents.model_settings"] = agents_model_settings_stub
    sys.modules.pop("think.openai", None)

    return last_kwargs, DummyRunner
