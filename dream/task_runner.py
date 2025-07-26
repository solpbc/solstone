import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

from flask_sock import Sock
from simple_websocket import ConnectionClosed

from . import state
from .tasks import task_manager
from .views.entities import reload_entities


def _run_command(
    cmd: list[str],
    logger: Callable[[str, str], None],
    stop: Optional[threading.Event] = None,
) -> tuple[int, str]:
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

    def _reader(stream, typ: str) -> None:
        for line in stream:
            logger(typ, line.rstrip())

    t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()
    stop = stop or threading.Event()
    while proc.poll() is None:
        if stop.is_set():
            proc.kill()
            break
        time.sleep(0.1)
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
    name: str,
    day: Optional[str] = None,
    logger: Optional[Callable[[str, str], None]] = None,
    *,
    force: bool = False,
    stop: Optional[threading.Event] = None,
) -> int:
    logger = logger or (lambda t, m: None)
    out_logger = _LineLogger("stdout", logger)
    err_logger = _LineLogger("stderr", logger)
    start = time.monotonic()
    info = f"{name} {day}" if day else name
    print(f"[task] start: {info}")
    use_stop = stop is not None
    stop = stop or threading.Event()

    commands: list[str] = []
    with contextlib.redirect_stdout(out_logger), contextlib.redirect_stderr(err_logger):
        try:
            if name == "reindex":
                code = 0
                for idx_name in ["summaries", "events", "transcripts"]:
                    args = [
                        sys.executable,
                        "-m",
                        "think.indexer",
                        "--index",
                        idx_name,
                        "--rescan",
                        "--verbose",
                    ]
                    commands.append(" ".join(args))
                    code = (
                        _run_command(args, logger, stop)
                        if use_stop
                        else _run_command(args, logger)
                    )
                    if code != 0:
                        break
            elif name == "summary":
                args = ["think-journal-stats", "--verbose"]
                commands.append(" ".join(args))
                code = (
                    _run_command(args, logger, stop)
                    if use_stop
                    else _run_command(args, logger)
                )
            elif name == "reload_entities":
                args = [
                    sys.executable,
                    "-m",
                    "think.entities",
                    "--rescan",
                    "--verbose",
                ]
                commands.append(" ".join(args))
                code = (
                    _run_command(args, logger, stop)
                    if use_stop
                    else _run_command(args, logger)
                )
                reload_entities()
            elif name == "hear_repair":
                if not day:
                    raise ValueError("day required")
                args = ["hear-transcribe", "--repair", day, "-v"]
                commands.append(" ".join(args))
                code = (
                    _run_command(args, logger, stop)
                    if use_stop
                    else _run_command(args, logger)
                )
            elif name == "see_repair":
                if not day:
                    raise ValueError("day required")
                args = ["see-describe", "--repair", day, "-v"]
                commands.append(" ".join(args))
                code = (
                    _run_command(args, logger, stop)
                    if use_stop
                    else _run_command(args, logger)
                )
            elif name == "ponder":
                if not day:
                    raise ValueError("day required")
                from think.utils import get_topics

                prompts = [info["path"] for info in get_topics().values()]
                prompts.sort()
                code = 0
                for prompt in prompts:
                    cmd = [
                        "think-ponder",
                        day,
                        "-f",
                        prompt,
                        "-p",
                        "--verbose",
                    ]
                    if force:
                        cmd.append("--force")
                    commands.append(" ".join(cmd))
                    code = (
                        _run_command(cmd, logger, stop)
                        if use_stop
                        else _run_command(cmd, logger)
                    )
                    if code != 0:
                        break
            elif name == "entity":
                if not day:
                    raise ValueError("day required")
                args = [
                    "think-entity-roll",
                    "--day",
                    day,
                    "--force",
                    "--verbose",
                ]
                commands.append(" ".join(args))
                code = (
                    _run_command(args, logger, stop)
                    if use_stop
                    else _run_command(args, logger)
                )
            elif name == "reduce":
                if not day:
                    raise ValueError("day required")
                cmd = ["see-reduce", day, "--verbose"]
                if force:
                    cmd.append("--force")
                commands.append(" ".join(cmd))
                code = (
                    _run_command(cmd, logger, stop)
                    if use_stop
                    else _run_command(cmd, logger)
                )
            elif name == "importer":
                if not day:
                    raise ValueError("file and timestamp required")
                try:
                    path, ts = day.split("|", 1)
                except ValueError:
                    raise ValueError("invalid importer args")
                cmd = ["think-importer", path, ts, "--verbose"]
                commands.append(" ".join(cmd))
                code = (
                    _run_command(cmd, logger, stop)
                    if use_stop
                    else _run_command(cmd, logger)
                )
                from think.utils import importer_log

                status = "ok" if code == 0 else f"fail({code})"
                importer_log(f"{os.path.basename(path)} {ts} {status}")
            elif name == "process_day":
                if not day:
                    raise ValueError("day required")
                cmd = [
                    "think-process-day",
                    "--day",
                    day,
                    "--verbose",
                ]
                if force:
                    cmd.append("--force")
                commands.append(" ".join(cmd))
                code = (
                    _run_command(cmd, logger, stop)
                    if use_stop
                    else _run_command(cmd, logger)
                )
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
    return code, " && ".join(commands)


class TaskRunner:
    def __init__(self, path: str = "/ws/tasks") -> None:
        self.path = path
        self.stops: dict[str, threading.Event] = {}

    def register(self, sock: Sock) -> None:
        @sock.route(self.path, endpoint="tasks_ws")
        def _handler(ws) -> None:
            self._handler(ws)

    def start(self) -> None:  # Backwards compatibility
        pass

    def _handler(self, ws) -> None:
        try:
            msg = ws.receive()
            req = json.loads(msg)
        except Exception as e:  # pragma: no cover - handshake errors
            try:
                ws.send(json.dumps({"type": "stderr", "text": str(e)}))
            except ConnectionClosed:
                pass
            ws.close()
            return

        if "attach" in req:
            tid = req.get("attach")
            task = task_manager.tasks.get(tid)
            if not task:
                try:
                    ws.send(json.dumps({"type": "stderr", "text": "unknown task"}))
                except ConnectionClosed:
                    pass
                ws.close()
                return
            for line in task.log:
                try:
                    ws.send(json.dumps({"type": "stdout", "text": line}))
                except ConnectionClosed:
                    break
            task.watchers.append(ws)
            while ws.connected:
                ws.receive(timeout=1)
            with task_manager.lock:
                if ws in task.watchers:
                    task.watchers.remove(ws)
            return

        if "kill" in req:
            tid = req.get("kill")
            stop = self.stops.get(tid)
            if stop:
                stop.set()
                task_manager.kill_task(tid)
            ws.close()
            return

        task = req.get("task")
        day = req.get("day")
        force = bool(req.get("force"))
        src = req.get("src", "")
        t = task_manager.create_task(task, day, src)
        stop = threading.Event()
        self.stops[t.id] = stop
        try:
            ws.send(json.dumps({"type": "id", "id": t.id}))
        except ConnectionClosed:
            return  # Client disconnected before task started

        def _log(typ: str, text: str) -> None:
            task_manager.append_log(t.id, typ, text)
            try:
                ws.send(json.dumps({"type": typ, "text": text}))
            except ConnectionClosed:
                pass  # WebSocket closed, log is still recorded in task_manager

        def _runner() -> None:
            code, cmd_str = run_task(task, day, _log, force=force, stop=stop)
            task_manager.finish_task(t.id, code, cmd_str)
            try:
                ws.send(json.dumps({"type": "exit", "code": code}))
            except ConnectionClosed:
                pass  # WebSocket closed, task is still marked as finished

        threading.Thread(target=_runner, daemon=True).start()
        while ws.connected:
            ws.receive(timeout=1)
        self.stops.pop(t.id, None)


task_runner = TaskRunner()
