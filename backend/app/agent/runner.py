"""Task orchestration: starting, pausing, resuming and cancelling runs.

The graph knows how to think; this knows how to *run*. It owns the background
asyncio tasks, maps a task id to a LangGraph thread, and -- the part that takes
the most care -- distinguishes a run that has **finished** from one that has
merely **paused** at an approval gate. The two look similar from the outside
(the coroutine returns either way) but mean opposite things for the browser:
a finished run releases its session, a paused one must keep the page exactly as
the user last saw it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.runtime import AgentRuntime
from app.agent.state import initial_state
from app.db.models import TaskStatus
from app.events import EventType

logger = logging.getLogger(__name__)


class TaskNotRunning(RuntimeError):
    """Raised when an operation targets a task that is not in flight."""


class AgentRunner:
    """Starts and supervises agent runs."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        self._graph = build_graph(runtime)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        #: Set for tasks currently parked at an approval interrupt.
        self._awaiting: set[str] = set()

    # -------------------------------------------------------------- public --

    async def start(
        self, goal: str, *, profile: str | None = None, task_id: str | None = None
    ) -> str:
        """Begin a run in the background and return its task id."""
        task_id = task_id or uuid.uuid4().hex
        profile = profile or self._runtime.settings.browser_default_profile

        await self._runtime.repository.create_task(task_id, goal, profile)

        state = initial_state(task_id, goal, profile)
        self._tasks[task_id] = asyncio.create_task(
            self._run(task_id, state), name=f"agent-{task_id}"
        )
        return task_id

    async def resume(self, task_id: str, approved: bool) -> None:
        """Answer a pending approval and let the run continue."""
        if task_id not in self._awaiting:
            raise TaskNotRunning(f"Task {task_id} is not waiting for approval.")
        self._awaiting.discard(task_id)
        self._tasks[task_id] = asyncio.create_task(
            self._run(task_id, Command(resume={"approved": approved})),
            name=f"agent-{task_id}-resume",
        )

    async def cancel(self, task_id: str) -> bool:
        """Stop a run and release its browser. Returns False if not running."""
        task = self._tasks.get(task_id)
        if task is None or task.done():
            self._awaiting.discard(task_id)
            await self._release(task_id)
            return False

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self._awaiting.discard(task_id)
        self._runtime.emit(task_id, EventType.TASK_CANCELLED, "Cancelled by the user")
        await self._runtime.repository.update_task(task_id, status=TaskStatus.CANCELLED)
        await self._release(task_id)
        return True

    def is_running(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        return task is not None and not task.done()

    def is_awaiting_approval(self, task_id: str) -> bool:
        return task_id in self._awaiting

    async def shutdown(self) -> None:
        """Cancel every in-flight run. Called on application shutdown."""
        for task_id in list(self._tasks):
            await self.cancel(task_id)

    # ------------------------------------------------------------ internal --

    async def _run(self, task_id: str, payload: object) -> None:
        """Drive the graph until it finishes or parks on an interrupt."""
        config = {
            "configurable": {"thread_id": task_id},
            # A generous ceiling: every agent step is several graph nodes, and
            # the real limits are the step and failure budgets in the router.
            "recursion_limit": self._runtime.settings.agent_max_steps * 8 + 50,
        }

        try:
            final = await self._graph.ainvoke(payload, config=config)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a crash must still be reported
            logger.exception("Task %s crashed", task_id)
            self._runtime.emit(task_id, EventType.TASK_FAILED, f"The run crashed: {exc}")
            await self._runtime.repository.update_task(
                task_id, status=TaskStatus.FAILED, error=str(exc)
            )
            await self._release(task_id)
            return

        # LangGraph reports a pause by putting `__interrupt__` in the returned
        # state. The coroutine has ended, but the *run* has not.
        if final.get("__interrupt__"):
            self._awaiting.add(task_id)
            logger.info("Task %s is waiting for approval", task_id)
            return

        await self._release(task_id)

    async def _release(self, task_id: str) -> None:
        """Close the browser for a run that is genuinely over."""
        try:
            await self._runtime.sessions.release(task_id)
        except Exception:  # noqa: BLE001 - teardown is best effort
            logger.exception("Failed to release browser for task %s", task_id)
