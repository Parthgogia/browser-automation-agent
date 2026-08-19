"""Data access for tasks, events and tool calls.

Every method here tolerates the database being absent, and does so by *working
anyway*: when Postgres is unreachable the repository keeps a bounded in-memory
mirror of recent tasks instead. That is what turns "forgot to start Docker"
into "history is not kept between restarts" rather than into a UI that 404s on
the task it is currently running.

Persistence is a feature of this system, not a prerequisite for it, and no call
site should have to remember that.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select

from app.db.models import Task, TaskEvent, TaskStatus, ToolCallRecord
from app.db.session import Database
from app.events.types import AgentEvent

logger = logging.getLogger(__name__)

#: Tool results can be very long (a full page of extracted text). Store enough
#: to audit the call, not the entire payload.
_MAX_STORED_RESULT = 4_000

#: Tasks retained by the in-memory fallback. Bounded so that a long-running
#: server without a database cannot leak memory indefinitely.
_FALLBACK_LIMIT = 100


class TaskRepository:
    """Reads and writes the task history.

    Backed by Postgres when it is available and by an in-memory mirror when it
    is not. Which one is in play is decided in one place -- `self._db.session()`
    yields `None` when there is no database this run.
    """

    def __init__(self, database: Database) -> None:
        self._db = database
        #: task_id -> detached Task, used only when Postgres is unavailable.
        self._fallback: dict[str, Task] = {}
        self._fallback_calls: dict[str, list[ToolCallRecord]] = {}

    def _remember(self, task: Task) -> None:
        """Add a task to the in-memory mirror, evicting the oldest if full."""
        if len(self._fallback) >= _FALLBACK_LIMIT:
            oldest = next(iter(self._fallback))
            self._fallback.pop(oldest, None)
            self._fallback_calls.pop(oldest, None)
        self._fallback[task.id] = task

    # -------------------------------------------------------------- writes --

    async def create_task(self, task_id: str, goal: str, profile: str) -> Task | None:
        task = Task(
            id=task_id,
            goal=goal,
            profile=profile,
            status=TaskStatus.PENDING,
            # Column defaults are applied by the INSERT, so a detached instance
            # needs them set explicitly to be usable as a fallback record.
            plan=[],
            steps_taken=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        async with self._db.session() as session:
            if session is None:
                self._remember(task)
                return task
            session.add(task)
            return task

    async def update_task(self, task_id: str, **fields: Any) -> None:
        """Patch a task row. Unknown field names are ignored."""
        async with self._db.session() as session:
            task = (
                self._fallback.get(task_id)
                if session is None
                else await session.get(Task, task_id)
            )
            if task is None:
                return
            for key, value in fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            status = fields.get("status")
            if status is not None and TaskStatus(status).is_terminal:
                task.finished_at = datetime.now(UTC)

    async def record_event(self, event: AgentEvent) -> None:
        """Persist one narration event.

        Failures are logged and swallowed: losing a log line must never take
        down a task that is otherwise going fine.
        """
        try:
            async with self._db.session() as session:
                if session is None:
                    return
                session.add(
                    TaskEvent(
                        id=event.id,
                        task_id=event.task_id,
                        type=str(event.type),
                        message=event.message[:4000],
                        data=_jsonable(event.data),
                        step=event.step,
                        created_at=event.created_at,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not persist event %s: %s", event.type, exc)

    async def record_tool_call(
        self,
        *,
        task_id: str,
        step: int,
        tool_name: str,
        arguments: dict[str, Any],
        ok: bool,
        result: str,
        duration_ms: int,
        risk: int = 0,
        approved: bool | None = None,
        page_url: str | None = None,
        screenshot_path: str | None = None,
    ) -> None:
        try:
            async with self._db.session() as session:
                if session is None:
                    self._fallback_calls.setdefault(task_id, []).append(
                        ToolCallRecord(
                            task_id=task_id,
                            step=step,
                            tool_name=tool_name,
                            arguments=_jsonable(arguments),
                            ok=ok,
                            result=result[:_MAX_STORED_RESULT],
                            duration_ms=duration_ms,
                            risk=risk,
                            approved=approved,
                            page_url=page_url,
                            screenshot_path=screenshot_path,
                            created_at=datetime.now(UTC),
                        )
                    )
                    return
                session.add(
                    ToolCallRecord(
                        task_id=task_id,
                        step=step,
                        tool_name=tool_name,
                        arguments=_jsonable(arguments),
                        ok=ok,
                        result=result[:_MAX_STORED_RESULT],
                        duration_ms=duration_ms,
                        risk=risk,
                        approved=approved,
                        page_url=page_url,
                        screenshot_path=screenshot_path,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not persist tool call %s: %s", tool_name, exc)

    # --------------------------------------------------------------- reads --

    async def get_task(self, task_id: str) -> Task | None:
        async with self._db.session() as session:
            if session is None:
                return self._fallback.get(task_id)
            return await session.get(Task, task_id)

    async def list_tasks(self, limit: int = 50, offset: int = 0) -> list[Task]:
        async with self._db.session() as session:
            if session is None:
                newest = sorted(
                    self._fallback.values(), key=lambda task: task.created_at, reverse=True
                )
                return newest[offset : offset + limit]
            result = await session.execute(
                select(Task).order_by(desc(Task.created_at)).limit(limit).offset(offset)
            )
            return list(result.scalars().all())

    async def list_events(self, task_id: str, limit: int = 1000) -> list[TaskEvent]:
        async with self._db.session() as session:
            if session is None:
                return []
            result = await session.execute(
                select(TaskEvent)
                .where(TaskEvent.task_id == task_id)
                .order_by(TaskEvent.created_at)
                .limit(limit)
            )
            return list(result.scalars().all())

    async def list_tool_calls(self, task_id: str) -> list[ToolCallRecord]:
        async with self._db.session() as session:
            if session is None:
                return list(self._fallback_calls.get(task_id, []))
            result = await session.execute(
                select(ToolCallRecord)
                .where(ToolCallRecord.task_id == task_id)
                .order_by(ToolCallRecord.step)
            )
            return list(result.scalars().all())


def _jsonable(value: Any) -> Any:
    """Coerce a payload into something the JSON column will accept.

    Tool arguments come from the model and event payloads come from all over
    the codebase, so neither is guaranteed to be JSON-clean. Anything exotic is
    stringified rather than allowed to blow up the insert.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
