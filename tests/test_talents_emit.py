# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json

from think.talents import JSONEventWriter


class _BrokenPipeStdout:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        raise BrokenPipeError()


def test_json_event_writer_emit_broken_pipe_keeps_sidecar(monkeypatch, tmp_path):
    stdout = _BrokenPipeStdout()
    monkeypatch.setattr("sys.stdout", stdout)
    sidecar = tmp_path / "events.jsonl"
    writer = JSONEventWriter(str(sidecar))

    try:
        writer.emit({"event": "error", "error": "pipe closed"})
        assert writer._pipe_dead is True
        first_write_count = len(stdout.writes)

        writer.emit({"event": "finish", "result": "sidecar only"})
        assert len(stdout.writes) == first_write_count
    finally:
        writer.close()

    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert rows == [
        {"event": "error", "error": "pipe closed"},
        {"event": "finish", "result": "sidecar only"},
    ]
