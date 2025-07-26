import json
from typing import List

from flask_sock import Sock
from simple_websocket import ConnectionClosed


class PushServer:
    """Simple global event WebSocket for the dream app."""

    def __init__(self, path: str = "/ws/events") -> None:
        self.path = path
        self.clients: List[object] = []

    def register(self, sock: Sock) -> None:
        @sock.route(self.path, endpoint="push_ws")
        def _handler(ws) -> None:
            self.clients.append(ws)
            try:
                while ws.connected:
                    ws.receive(timeout=1)
            except ConnectionClosed:
                pass
            finally:
                if ws in self.clients:
                    self.clients.remove(ws)

    def start(self) -> None:  # Backwards compatibility
        pass

    def push(self, event: dict) -> None:
        msg = json.dumps(event)
        for ws in list(self.clients):
            try:
                ws.send(msg)
            except ConnectionClosed:
                if ws in self.clients:
                    self.clients.remove(ws)


push_server = PushServer()
