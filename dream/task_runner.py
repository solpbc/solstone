import asyncio
import contextlib
import glob
import json
import os
import subprocess
import threading
from typing import Callable, Optional

import websockets

from . import state
from think import entity_roll
from think.indexer import (
    load_cache,
    save_cache,
    scan_entities,
    scan_occurrences,
    scan_ponders,
)
from think.journal_stats import JournalStats
from think.reduce_screen import reduce_day
from .views.entities import reload_entities


def _run_command(cmd: list[str], logger: Callable[[str, str], None]) -> int:
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


def run_task(name: str, day: Optional[str] = None, logger: Optional[Callable[[str, str], None]] = None) -> int:
    logger = logger or (lambda t, m: None)
    out_logger = _LineLogger("stdout", logger)
    err_logger = _LineLogger("stderr", logger)

    with contextlib.redirect_stdout(out_logger), contextlib.redirect_stderr(err_logger):
        try:
            if name == "reindex":
                journal = state.journal_root
                cache = load_cache(journal)
                changed = False
                changed |= scan_entities(journal, cache)
                changed |= scan_ponders(journal, cache)
                changed |= scan_occurrences(journal, cache)
                if changed:
                    save_cache(journal, cache)
                code = 0
            elif name == "summary":
                js = JournalStats()
                js.scan(state.journal_root)
                js.save_markdown(state.journal_root)
                code = 0
            elif name == "reload_entities":
                reload_entities()
                code = 0
            elif name == "repairs":
                if not day:
                    raise ValueError("day required")
                code = _run_command(["gemini-transcribe", "--repair", day], logger)
                if code == 0:
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
                day_dirs = entity_roll.find_day_dirs(state.journal_root)
                entity_roll.process_day(day, day_dirs, True)
                code = 0
            elif name == "reduce":
                if not day:
                    raise ValueError("day required")
                reduce_day(day)
                code = 0
            else:
                logger("stderr", f"Unknown task: {name}")
                code = 1
        except Exception as e:  # pragma: no cover - for unexpected errors
            logger("stderr", str(e))
            code = 1
        finally:
            out_logger.flush()
            err_logger.flush()
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
        self.loop.run_until_complete(websockets.serve(self._handler, self.host, self.port))
        self.loop.run_forever()

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
