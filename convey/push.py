import json
from typing import List

from flask_sock import Sock
from simple_websocket import ConnectionClosed


class PushServer:
    """Simple event WebSocket that broadcasts all events to all clients."""

    def __init__(self, path: str = "/ws/events") -> None:
        self.path = path
        self.clients: List[object] = []

    def register(self, sock: Sock) -> None:
        @sock.route(self.path, endpoint="push_ws")
        def _handler(ws) -> None:
            self.clients.append(ws)
            try:
                while ws.connected:
                    # Just keep connection alive, no need for subscriptions
                    ws.receive(timeout=1)
            except ConnectionClosed:
                pass
            finally:
                if ws in self.clients:
                    self.clients.remove(ws)

    def start(self) -> None:  # Backwards compatibility
        pass

    def push(self, event: dict) -> None:
        """Push event to all connected clients."""
        msg = json.dumps(event)
        for ws in list(self.clients):
            try:
                ws.send(msg)
            except ConnectionClosed:
                if ws in self.clients:
                    self.clients.remove(ws)


push_server = PushServer()
