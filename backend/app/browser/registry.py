"""Registry of live browser sessions, keyed by task id.

Why this exists
---------------
LangGraph checkpoints its state so a run can be paused for human approval and
resumed later. That requires the state to be serialisable. A live Chromium
connection is not, so the graph stores only a `task_id` string and every node
that needs the browser asks this registry for the session.

The registry is also where session *lifetime* is decided: a session outlives a
single graph invocation (so an approval pause does not lose the page you were
on) and is only torn down when the task ends or is cancelled.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from app.browser.session import BrowserSession
from app.config import Settings

logger = logging.getLogger(__name__)


class SessionRegistry:
    """Maps task ids to live :class:`BrowserSession` objects."""

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        # Guards against two concurrent steps of the same task each deciding
        # that no session exists and launching a second browser.
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, task_id: str) -> asyncio.Lock:
        return self._locks.setdefault(task_id, asyncio.Lock())

    async def acquire(
        self, task_id: str, settings: Settings, profile: str | None = None
    ) -> BrowserSession:
        """Return the session for `task_id`, launching one if needed."""
        async with self._lock_for(task_id):
            session = self._sessions.get(task_id)
            if session is not None:
                return session
            session = await BrowserSession.start(settings, profile)
            self._sessions[task_id] = session
            return session

    def get(self, task_id: str) -> BrowserSession | None:
        """Return the session for `task_id` without creating one."""
        return self._sessions.get(task_id)

    async def release(self, task_id: str) -> None:
        """Close and forget the session for `task_id`."""
        session = self._sessions.pop(task_id, None)
        self._locks.pop(task_id, None)
        if session is not None:
            await session.close()

    async def release_all(self) -> None:
        """Close every session. Called on application shutdown."""
        for task_id in list(self._sessions):
            try:
                await self.release(task_id)
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.exception("Failed to release browser session for task %s", task_id)

    @property
    def active_task_ids(self) -> list[str]:
        return list(self._sessions)


@lru_cache
def get_session_registry() -> SessionRegistry:
    """Process-wide registry singleton."""
    return SessionRegistry()
