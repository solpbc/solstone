from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import state


@dataclass
class Task:
    id: str
    name: str
    day: Optional[str]
    start: float
    initiator: str = ""
    end: Optional[float] = None
    exit_code: Optional[int] = None
    killed: bool = False
    log: List[str] = field(default_factory=list)
    watchers: List[tuple[asyncio.AbstractEventLoop, object]] = field(
        default_factory=list, repr=False
    )

    @property
    def duration(self) -> float:
        end = self.end or time.time()
        return end - self.start


class TaskManager:
    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()

    # tasks dir path
    def _tasks_dir(self) -> str:
        if not state.journal_root:
            return ""
        path = os.path.join(state.journal_root, "tasks")
        os.makedirs(path, exist_ok=True)
        return path

    def create_task(self, name: str, day: Optional[str], initiator: str) -> Task:
        tid = f"{int(time.time()*1000)}"
        task = Task(id=tid, name=name, day=day, start=time.time(), initiator=initiator)
        with self.lock:
            self.tasks[tid] = task
        return task

    def list_tasks(self) -> List[Dict[str, object]]:
        with self.lock:
            items = list(self.tasks.values())
        out: List[Dict[str, object]] = []
        for t in sorted(items, key=lambda x: x.start, reverse=True):
            out.append(
                {
                    "id": t.id,
                    "name": t.name,
                    "day": t.day,
                    "start": t.start,
                    "end": t.end,
                    "exit_code": t.exit_code,
                    "killed": t.killed,
                    "initiator": t.initiator,
                    "lines": len(t.log),
                }
            )
        return out

    def append_log(self, tid: str, typ: str, text: str) -> None:
        with self.lock:
            task = self.tasks.get(tid)
            if not task:
                return
            task.log.append(text)
            watchers = list(task.watchers)
        for loop, ws in watchers:
            try:
                asyncio.run_coroutine_threadsafe(
                    ws.send(json.dumps({"type": typ, "text": text})), loop
                )
            except Exception:
                pass

    def finish_task(self, tid: str, code: int) -> None:
        with self.lock:
            task = self.tasks.get(tid)
            if not task:
                return
            task.end = time.time()
            task.exit_code = code
            task.watchers.clear()
            self._save_task(task)

    def kill_task(self, tid: str) -> None:
        with self.lock:
            task = self.tasks.get(tid)
            if not task:
                return
            task.killed = True
            task.end = time.time()
            task.watchers.clear()
            self._save_task(task)

    def _save_task(self, task: Task) -> None:
        path = self._tasks_dir()
        if not path:
            return
        fname = os.path.join(path, f"{task.id}.json")
        data = {
            "id": task.id,
            "name": task.name,
            "day": task.day,
            "start": task.start,
            "end": task.end,
            "exit_code": task.exit_code,
            "killed": task.killed,
            "initiator": task.initiator,
            "log": task.log,
        }
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load_cached(self) -> None:
        path = self._tasks_dir()
        if not path:
            return
        for name in os.listdir(path):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(path, name), "r", encoding="utf-8") as f:
                    data = json.load(f)
                t = Task(
                    id=data.get("id", name[:-5]),
                    name=data.get("name", ""),
                    day=data.get("day"),
                    start=float(data.get("start", time.time())),
                    initiator=data.get("initiator", ""),
                    end=data.get("end"),
                    exit_code=data.get("exit_code"),
                    killed=bool(data.get("killed")),
                    log=data.get("log", []),
                )
                self.tasks[t.id] = t
            except Exception:
                continue

    def clear_old(self, days: int = 7) -> int:
        cutoff = time.time() - days * 86400
        removed = 0
        path = self._tasks_dir()
        if not path:
            return 0
        for name in os.listdir(path):
            if not name.endswith(".json"):
                continue
            full = os.path.join(path, name)
            try:
                stat = os.stat(full)
            except FileNotFoundError:
                continue
            if stat.st_mtime < cutoff:
                try:
                    os.remove(full)
                    removed += 1
                except Exception:
                    pass
        return removed


task_manager = TaskManager()
