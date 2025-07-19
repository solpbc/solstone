import asyncio
import json
import threading
from typing import List, Tuple

import websockets


class PushServer:
    """Simple global event websocket for the dream app."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8766) -> None:
        self.host = host
        self.port = port
        self.loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self.clients: List[
            Tuple[asyncio.AbstractEventLoop, websockets.WebSocketServerProtocol]
        ] = []

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self) -> None:
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)

        async def start_server() -> None:
            server = await websockets.serve(self._handler, self.host, self.port)
            await server.wait_closed()

        self.loop.run_until_complete(start_server())

    async def _handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        assert self.loop is not None
        self.clients.append((self.loop, ws))
        try:
            await ws.wait_closed()
        finally:
            if (self.loop, ws) in self.clients:
                self.clients.remove((self.loop, ws))

    def push(self, event: dict) -> None:
        msg = json.dumps(event)
        for loop, ws in list(self.clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
            except Exception:
                if (loop, ws) in self.clients:
                    self.clients.remove((loop, ws))


push_server = PushServer()
