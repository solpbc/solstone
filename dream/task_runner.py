import asyncio
import contextlib
import glob
import json
import os
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import websockets

from think import entity_roll

from . import state
from .views.entities import reload_entities


def _run_command(cmd: list[str], logger: Callable[[str, str], None]) -> int:
    print("[task] run:", " ".join(cmd))
    env = os.environ.copy()
    if state.journal_root:
        env["JOURNAL_PATH"] = state.journal_root
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout and proc.stderr

    def _reader(stream: asyncio.StreamReader, typ: str) -> None:
        for line in stream:
            logger(typ, line.rstrip())

    t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()
    print(f"[task] done: {' '.join(cmd)} (exit {proc.returncode})")
    return proc.returncode


class _LineLogger:
    def __init__(self, typ: str, logger: Callable[[str, str], None]):
        self.typ = typ
        self.logger = logger
        self.buf = ""

    def write(self, data: str) -> int:
        self.buf += data
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line:
                self.logger(self.typ, line)
        return len(data)

    def flush(self) -> None:
        if self.buf:
            self.logger(self.typ, self.buf)
            self.buf = ""


def run_task(
    name: str, day: Optional[str] = None, logger: Optional[Callable[[str, str], None]] = None
) -> int:
    logger = logger or (lambda t, m: None)
    out_logger = _LineLogger("stdout", logger)
    err_logger = _LineLogger("stderr", logger)
    start = time.monotonic()
    info = f"{name} {day}" if day else name
    print(f"[task] start: {info}")

    with contextlib.redirect_stdout(out_logger), contextlib.redirect_stderr(err_logger):
        try:
            if name == "reindex":
                code = _run_command(
                    [
                        sys.executable,
                        "-m",
                        "think.indexer",
                        "--rescan",
                    ],
                    logger,
                )
            elif name == "summary":
                code = _run_command(["journal-stats"], logger)
            elif name == "reload_entities":
                code = _run_command(
                    [
                        sys.executable,
                        "-m",
                        "think.entities",
                        "--rescan",
                    ],
                    logger,
                )
                reload_entities()
            elif name == "hear_repair":
                if not day:
                    raise ValueError("day required")
                code = _run_command(["gemini-transcribe", "--repair", day], logger)
            elif name == "see_repair":
                if not day:
                    raise ValueError("day required")
                code = _run_command(["screen-describe", "--repair", day], logger)
            elif name == "ponder":
                if not day:
                    raise ValueError("day required")
                think_dir = os.path.dirname(entity_roll.__file__)
                prompts = sorted(glob.glob(os.path.join(think_dir, "ponder", "*.txt")))
                code = 0
                for prompt in prompts:
                    code = _run_command(["ponder", day, "-f", prompt, "-p"], logger)
                    if code != 0:
                        break
            elif name == "entity":
                if not day:
                    raise ValueError("day required")
                code = _run_command(
                    [
                        "entity-roll",
                        "--day",
                        day,
                        "--force",
                    ],
                    logger,
                )
            elif name == "reduce":
                if not day:
                    raise ValueError("day required")
                code = _run_command(["reduce-screen", day], logger)
            elif name == "process_day":
                if not day:
                    raise ValueError("day required")
                code = _run_command(["process-day", "--day", day, "--repair"], logger)
            else:
                logger("stderr", f"Unknown task: {name}")
                code = 1
        except Exception as e:  # pragma: no cover - for unexpected errors
            logger("stderr", str(e))
            code = 1
        finally:
            out_logger.flush()
            err_logger.flush()
    print(f"[task] end: {info} ({time.monotonic() - start:.1f}s)")
    return code


class TaskRunner:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.loop: asyncio.AbstractEventLoop | None = None
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

        async def start_server():
            server = await websockets.serve(lambda ws: self._handler(ws, ""), self.host, self.port)
            await server.wait_closed()

        self.loop.run_until_complete(start_server())

    async def _handler(self, ws: websockets.WebSocketServerProtocol, path: str) -> None:
        try:
            msg = await ws.recv()
            req = json.loads(msg)
            task = req.get("task")
            day = req.get("day")
        except Exception as e:  # pragma: no cover - handshake errors
            await ws.send(json.dumps({"type": "stderr", "text": str(e)}))
            await ws.close()
            return

        def _log(typ: str, text: str) -> None:
            if not self.loop:
                return
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps({"type": typ, "text": text})), self.loop
            )

        def _runner() -> None:
            code = run_task(task, day, _log)
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    ws.send(json.dumps({"type": "exit", "code": code})), self.loop
                )

        threading.Thread(target=_runner, daemon=True).start()
        await ws.wait_closed()


task_runner = TaskRunner()
