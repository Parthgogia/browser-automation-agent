"""Execution node: actually do the thing.

Every effect this agent has on the world passes through this one function,
which is what makes the audit trail complete. It dispatches the call, times it,
records it, and -- crucially -- feeds the outcome back into the transcript in
the model's own conversational format, so the next decision is made with full
knowledge of what just happened.

Failures are data, not exceptions. A failed action increments a counter that
`reflect` watches; three failures in a row means the current approach is wrong
and the agent should rethink rather than retry harder.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.db.models import TaskStatus
from app.events import EventType
from app.llm.base import Message, ToolCall

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]


def make_execute_node(runtime: AgentRuntime) -> NodeFn:
    """Build the tool-execution node."""

    async def execute(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0)
        call: ToolCall | None = state.get("pending_call")
        risk = state.get("pending_risk") or {}

        if call is None:  # defensive: routing should never send us here
            return {"pending_call": None}

        context = runtime.tool_context(task_id, state["goal"], state.get("profile", "default"))
        started = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                runtime.tools.execute(call.name, call.arguments, context),
                timeout=runtime.settings.agent_step_timeout_s,
            )
        except TimeoutError:
            from app.tools.base import ToolResult

            result = ToolResult.failure(
                f"{call.name} took longer than "
                f"{runtime.settings.agent_step_timeout_s}s and was abandoned. "
                "The page may be stuck; try reloading or a different approach."
            )

        duration_ms = int((time.perf_counter() - started) * 1000)

        runtime.emit(
            task_id,
            EventType.TOOL_RESULT,
            result.message[:1000],
            step=step,
            tool=call.name,
            ok=result.ok,
            duration_ms=duration_ms,
        )
        await runtime.repository.record_tool_call(
            task_id=task_id,
            step=step,
            tool_name=call.name,
            arguments=call.arguments,
            ok=result.ok,
            result=result.message,
            duration_ms=duration_ms,
            risk=int(risk.get("risk", 0)),
            approved=state.get("approval_granted"),
            page_url=_current_url(runtime, task_id),
        )

        update: dict = {
            "pending_call": None,
            "pending_risk": None,
            "approval_granted": None,
            "messages": [Message.tool_result(call, result.render())],
            "consecutive_failures": (
                0 if result.ok else state.get("consecutive_failures", 0) + 1
            ),
            "total_failures": state.get("total_failures", 0) + (0 if result.ok else 1),
        }

        if result.terminal:
            update.update(_terminal_update(call, result))

        await runtime.repository.update_task(task_id, steps_taken=step)
        return update

    return execute


def _terminal_update(call: ToolCall, result) -> dict:
    """State changes for the two tools that end a run."""
    if call.name == "ask_user":
        return {
            "status": TaskStatus.AWAITING_INPUT,
            "question": result.data.get("question", result.message),
            "result": result.message,
            "success": None,
        }
    return {
        "status": TaskStatus.COMPLETED if result.ok else TaskStatus.FAILED,
        "result": result.data.get("summary", result.message),
        "success": result.ok,
    }


def _current_url(runtime: AgentRuntime, task_id: str) -> str:
    session = runtime.sessions.get(task_id)
    observation = session.last_observation if session else None
    return observation.url if observation else ""
