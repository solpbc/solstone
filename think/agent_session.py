from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["BaseAgentSession"]


class BaseAgentSession(ABC):
    """Abstract base class for LLM agent sessions."""

    @abstractmethod
    async def __aenter__(self) -> "BaseAgentSession":
        """Enter the session context."""

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the session context."""

    @property
    @abstractmethod
    def history(self) -> list[dict[str, str]]:
        """Return the chat history as ``role``/``content`` pairs."""

    @abstractmethod
    def add_history(self, role: str, text: str) -> None:
        """Queue a prior message for the next run."""

    @abstractmethod
    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through the agent and return the response text."""
