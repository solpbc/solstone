# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Persistent voice brain for session instructions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from think.utils import get_config, get_journal
from think.voice.config import get_brain_model

logger = logging.getLogger(__name__)

BRAIN_REFRESH_MAX_AGE_SECONDS = 6 * 3600
SESSION_FILENAME = "voice-brain-session"

_INIT_PROMPT_TEMPLATE = """You are preparing the current voice-session instruction for {agent_name}, the spoken identity of this solstone journal.

Your task right now is to read the current journal state and produce exactly one fresh instruction for an OpenAI Realtime voice session. The instruction must sound like spoken English. Keep it concise, natural, and useful in conversation. No markdown. No bullets. No XML outside the required wrapper tags.

Voice style rules:
- Write for speech, not reading.
- Keep the voice model oriented toward short spoken turns, usually 2 to 4 sentences unless the user clearly asks for more.
- Prefer concrete wording over abstract wording.
- If context is missing, the instruction should say to answer honestly and briefly rather than guessing.

Terminology covenant:
- Use the words observer and listen when referring to the live sensing system.
- Never use the words keeper, assistant, record, or capture.

Before you write the instruction, ingest the current context:
- Read the identity material under journal/identity/ and treat {agent_name} as the canonical spoken name.
- Read today's journal summary and today's segment-level summaries if they exist.
- Read the active entities that matter right now.
- Read the open commitments.
- Read today's calendar and anticipated activities.
- Read the latest briefing in journal/identity/briefing.md if it is for today.

Then write one system instruction that does all of the following:
- Establish who {agent_name} is and how the voice should speak.
- Anchor the voice in today's real context.
- Name the most important people, commitments, and upcoming events if they are present.
- Tell the voice to stay concise, spoken, and honest about missing information.
- Preserve the terminology covenant above.

Output only this wrapper and the instruction inside it:
<voice_instruction>
...
</voice_instruction>
"""

_REFRESH_PROMPT_TEMPLATE = """Read the current journal state again and refresh the voice-session instruction for {agent_name}.

Repeat the same steps as the startup pass:
- read the identity material
- read today's journal summary and segment-level summaries if they exist
- read active entities, open commitments, today's calendar, and today's briefing if present
- keep the terminology covenant in force

Output only the refreshed instruction between <voice_instruction> tags.
"""


@dataclass
class BrainState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready_event: threading.Event = field(default_factory=threading.Event)
    start_future: Future[tuple[str, str]] | None = None
    refresh_future: Future[tuple[str, str]] | None = None
    last_error: str | None = None


_BRAIN_STATE = BrainState()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _session_file(journal: str | Path | None = None) -> Path:
    journal_root = Path(journal) if journal is not None else Path(get_journal())
    return journal_root / "health" / SESSION_FILENAME


def extract_instruction(text: str) -> str | None:
    start_tag = "<voice_instruction>"
    end_tag = "</voice_instruction>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start == -1 or end == -1:
        return None
    return text[start + len(start_tag) : end].strip() or None


def _agent_name() -> str:
    config = get_config()
    agent = config.get("agent")
    if isinstance(agent, dict):
        value = agent.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "sol"


def _build_init_prompt() -> str:
    return _INIT_PROMPT_TEMPLATE.format(agent_name=_agent_name())


def _build_refresh_prompt() -> str:
    return _REFRESH_PROMPT_TEMPLATE.format(agent_name=_agent_name())


def _claude_cmd() -> str:
    executable = shutil.which("claude")
    if executable:
        return executable
    raise RuntimeError("claude CLI not available")


async def _run_claude(
    message: str,
    extra_args: list[str],
    *,
    timeout: float,
) -> tuple[str, str]:
    cmd = [
        _claude_cmd(),
        "-p",
        message,
        "--model",
        get_brain_model(),
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        *extra_args,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(_repo_root()),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise

    if proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip() or (
            f"claude exited {proc.returncode}"
        )
        raise RuntimeError(message)

    payload = json.loads(stdout.decode("utf-8"))
    result = payload.get("result")
    session_id = payload.get("session_id")
    if not isinstance(result, str) or not isinstance(session_id, str):
        raise RuntimeError("claude response missing result or session_id")
    return result, session_id


def load_session_id(journal: str | Path | None = None) -> str | None:
    try:
        session_id = _session_file(journal).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return session_id or None


def _save_session_id(session_id: str, journal: str | Path | None = None) -> None:
    path = _session_file(journal)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(session_id, encoding="utf-8")
    os.replace(tmp, path)


def _touch_session_file(journal: str | Path | None = None) -> None:
    path = _session_file(journal)
    if path.exists():
        path.touch()


def brain_age_seconds(app: Any) -> int | None:
    refreshed_at = getattr(app, "voice_brain_refreshed_at", None)
    if not isinstance(refreshed_at, (int, float)):
        return None
    return int(max(0, time.time() - refreshed_at))


def brain_is_stale(
    app: Any, *, max_age_seconds: int = BRAIN_REFRESH_MAX_AGE_SECONDS
) -> bool:
    age = brain_age_seconds(app)
    return age is not None and age > max_age_seconds


async def start_brain(journal: str | Path | None = None) -> tuple[str, str]:
    logger.info("voice brain starting")
    text, session_id = await _run_claude(
        _build_init_prompt(),
        ["-n", "voice-brain"],
        timeout=300,
    )
    instruction = extract_instruction(text)
    if instruction is None:
        raise RuntimeError("voice instruction missing")
    _save_session_id(session_id, journal)
    return session_id, instruction


async def refresh_brain(
    session_id: str,
    journal: str | Path | None = None,
) -> str:
    logger.info("voice brain refreshing")
    text, _ = await _run_claude(
        _build_refresh_prompt(),
        ["--resume", session_id],
        timeout=300,
    )
    instruction = extract_instruction(text)
    if instruction is None:
        raise RuntimeError("voice instruction missing")
    _touch_session_file(journal)
    return instruction


async def ask_brain(session_id: str, question: str) -> str:
    logger.info("voice brain answering follow-up")
    text, _ = await _run_claude(
        question,
        ["--resume", session_id],
        timeout=120,
    )
    return text


async def _start_or_resume_brain(app: Any) -> tuple[str, str]:
    journal_root = getattr(app, "voice_journal_root", None)
    session_id = load_session_id(journal_root)
    if session_id:
        try:
            instruction = await refresh_brain(session_id, journal_root)
            return session_id, instruction
        except Exception:
            logger.warning(
                "voice brain resume failed; starting new session", exc_info=True
            )
    return await start_brain(journal_root)


async def _refresh_existing_brain(app: Any) -> tuple[str, str]:
    journal_root = getattr(app, "voice_journal_root", None)
    session_id = getattr(app, "voice_brain_session", None) or load_session_id(
        journal_root
    )
    if not isinstance(session_id, str) or not session_id.strip():
        return await _start_or_resume_brain(app)
    instruction = await refresh_brain(session_id, journal_root)
    return session_id, instruction


def _apply_brain_result(app: Any, session_id: str, instruction: str) -> None:
    app.voice_brain_session = session_id
    app.voice_brain_instruction = instruction
    app.voice_brain_refreshed_at = time.time()
    _BRAIN_STATE.ready_event.set()
    _BRAIN_STATE.last_error = None


def _complete_future(
    app: Any,
    attr_name: str,
    future: Future[tuple[str, str]],
) -> None:
    try:
        session_id, instruction = future.result()
    except Exception as exc:
        logger.exception("voice brain task failed")
        with _BRAIN_STATE.lock:
            setattr(_BRAIN_STATE, attr_name, None)
            _BRAIN_STATE.last_error = str(exc)
            if not getattr(app, "voice_brain_instruction", ""):
                _BRAIN_STATE.ready_event.clear()
        return

    with _BRAIN_STATE.lock:
        setattr(_BRAIN_STATE, attr_name, None)
        _apply_brain_result(app, session_id, instruction)


def _runtime_loop():
    from think.voice.runtime import get_runtime_state

    runtime = get_runtime_state()
    if runtime.loop is None:
        raise RuntimeError("voice runtime unavailable")
    return runtime.loop


def schedule_start(app: Any) -> Future[tuple[str, str]]:
    if getattr(app, "voice_brain_instruction", ""):
        _BRAIN_STATE.ready_event.set()
    with _BRAIN_STATE.lock:
        existing = _BRAIN_STATE.start_future
        if existing is not None and not existing.done():
            return existing
        future = asyncio.run_coroutine_threadsafe(
            _start_or_resume_brain(app), _runtime_loop()
        )
        _BRAIN_STATE.start_future = future
        future.add_done_callback(
            lambda done: _complete_future(app, "start_future", done)
        )
        return future


def schedule_refresh(app: Any, *, force: bool = False) -> Future[tuple[str, str]]:
    if (
        not force
        and getattr(app, "voice_brain_instruction", "")
        and not brain_is_stale(app)
    ):
        return schedule_start(app)
    with _BRAIN_STATE.lock:
        existing = _BRAIN_STATE.refresh_future
        if existing is not None and not existing.done():
            return existing
        if (
            _BRAIN_STATE.start_future is not None
            and not _BRAIN_STATE.start_future.done()
        ):
            return _BRAIN_STATE.start_future
        future = asyncio.run_coroutine_threadsafe(
            _refresh_existing_brain(app),
            _runtime_loop(),
        )
        _BRAIN_STATE.refresh_future = future
        future.add_done_callback(
            lambda done: _complete_future(app, "refresh_future", done)
        )
        return future


def wait_until_ready(app: Any, timeout: float) -> bool:
    if getattr(app, "voice_brain_instruction", ""):
        _BRAIN_STATE.ready_event.set()
        return True
    schedule_start(app)
    ready = _BRAIN_STATE.ready_event.wait(timeout)
    return ready and bool(getattr(app, "voice_brain_instruction", ""))


def clear_brain_state() -> None:
    with _BRAIN_STATE.lock:
        futures = [
            _BRAIN_STATE.start_future,
            _BRAIN_STATE.refresh_future,
        ]
        _BRAIN_STATE.start_future = None
        _BRAIN_STATE.refresh_future = None
        _BRAIN_STATE.last_error = None
        _BRAIN_STATE.ready_event.clear()
    for future in futures:
        if future is not None and not future.done():
            future.cancel()


__all__ = [
    "BRAIN_REFRESH_MAX_AGE_SECONDS",
    "ask_brain",
    "brain_age_seconds",
    "brain_is_stale",
    "clear_brain_state",
    "extract_instruction",
    "load_session_id",
    "refresh_brain",
    "schedule_refresh",
    "schedule_start",
    "start_brain",
    "wait_until_ready",
]
