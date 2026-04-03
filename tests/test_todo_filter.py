# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unit tests for the todo detector pre-filter hook."""

from apps.todos.muse.todo_filter import pre_process


class TestTodoFilter:
    def test_empty_transcript_skips(self):
        result = pre_process({"transcript": ""})
        assert result == {"skip_reason": "no commitment signals in transcript"}

    def test_missing_transcript_skips(self):
        result = pre_process({})
        assert result == {"skip_reason": "no commitment signals in transcript"}

    def test_none_transcript_skips(self):
        result = pre_process({"transcript": None})
        assert result == {"skip_reason": "no commitment signals in transcript"}

    def test_whitespace_transcript_skips(self):
        result = pre_process({"transcript": "   \n  "})
        assert result == {"skip_reason": "no commitment signals in transcript"}

    def test_no_signals_skips(self):
        result = pre_process({"transcript": "just writing some python code today"})
        assert result == {"skip_reason": "no commitment signals in transcript"}

    def test_action_commitment_proceeds(self):
        result = pre_process({"transcript": "I'll send that email tomorrow"})
        assert result is None

    def test_follow_up_proceeds(self):
        result = pre_process({"transcript": "need to follow up with the team"})
        assert result is None

    def test_reminder_proceeds(self):
        result = pre_process({"transcript": "remind me to check the logs"})
        assert result is None

    def test_deadline_proceeds(self):
        result = pre_process({"transcript": "need to finish this by Monday"})
        assert result is None

    def test_explicit_marker_proceeds(self):
        result = pre_process({"transcript": "TODO: fix the auth flow"})
        assert result is None

    def test_task_creation_proceeds(self):
        result = pre_process({"transcript": "add to my list: buy groceries"})
        assert result is None

    def test_case_insensitive(self):
        result = pre_process({"transcript": "i'll handle it"})
        assert result is None

    def test_we_should_proceeds(self):
        result = pre_process({"transcript": "we should schedule a meeting"})
        assert result is None

    def test_circle_back_proceeds(self):
        result = pre_process({"transcript": "let's circle back on this"})
        assert result is None

    def test_dont_forget_proceeds(self):
        result = pre_process({"transcript": "don't forget to update the docs"})
        assert result is None

    def test_action_items_proceeds(self):
        result = pre_process({"transcript": "here are the action items from today"})
        assert result is None

    def test_next_steps_proceeds(self):
        result = pre_process({"transcript": "the next steps are to review the PR"})
        assert result is None
