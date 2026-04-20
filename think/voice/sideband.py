# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""OpenAI Realtime sideband loop for voice sessions."""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any

from openai import AsyncOpenAI

from think.voice.config import get_openai_api_key, get_voice_model
from think.voice.tools import dispatch_tool_call

logger = logging.getLogger(__name__)


async def _sideband_loop(conn: Any, call_id: str, app: Any) -> None:
    async for event in conn:
        if getattr(event, "type", None) != "response.function_call_arguments.done":
            continue
        output = await dispatch_tool_call(
            getattr(event, "name", ""),
            getattr(event, "arguments", ""),
            call_id,
            app,
        )
        await conn.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": getattr(event, "call_id", ""),
                "output": output,
            }
        )
        await conn.response.create()


async def _run_sideband(call_id: str, app: Any) -> None:
    openai_key = get_openai_api_key()
    if openai_key is None:
        raise RuntimeError("voice unavailable — openai key not configured")
    client = AsyncOpenAI(api_key=openai_key)
    logger.info("voice sideband starting call_id=%s", call_id)
    try:
        async with client.realtime.connect(
            call_id=call_id,
            model=get_voice_model(),
        ) as conn:
            await _sideband_loop(conn, call_id, app)
    except Exception:
        logger.exception("voice sideband failed call_id=%s", call_id)


def register_voice_task(app: Any, future: Future[Any]) -> None:
    app.voice_tasks.add(future)
    future.add_done_callback(app.voice_tasks.discard)


__all__ = ["_run_sideband", "_sideband_loop", "register_voice_task"]
