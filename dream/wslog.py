import asyncio
import logging
import threading
from collections import defaultdict
from typing import DefaultDict, Set

import websockets


class WSLogServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: DefaultDict[str, Set[websockets.WebSocketServerProtocol]] = defaultdict(set)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self) -> None:
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)

        async def handler(ws: websockets.WebSocketServerProtocol, path: str) -> None:
            job_id = path.lstrip("/")
            self.clients[job_id].add(ws)
            try:
                await ws.wait_closed()
            finally:
                self.clients[job_id].discard(ws)

        async def run_server() -> None:
            server = await websockets.serve(handler, self.host, self.port)
            await server.wait_closed()

        self.loop.run_until_complete(run_server())

    def broadcast(self, job_id: str, message: str) -> None:
        if not self.loop or job_id not in self.clients:
            return
        for ws in list(self.clients[job_id]):
            asyncio.run_coroutine_threadsafe(ws.send(message), self.loop)


class WSLogHandler(logging.Handler):
    def __init__(self, server: WSLogServer, job_id: str) -> None:
        super().__init__()
        self.server = server
        self.job_id = job_id

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.server.broadcast(self.job_id, msg)


def capture_logs(server: WSLogServer, job_id: str):
    handler = WSLogHandler(server, job_id)
    logger = logging.getLogger()
    logger.addHandler(handler)

    class _Ctx:
        def __enter__(self):
            return handler

        def __exit__(self, exc_type, exc, tb):
            logger.removeHandler(handler)

    return _Ctx()


ws_server = WSLogServer()
