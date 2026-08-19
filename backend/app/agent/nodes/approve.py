"""Human-in-the-loop gate.

Two nodes live here, because they are two halves of one idea -- deciding
whether the agent is allowed to proceed:

``approve``
    Pauses the entire graph on a proposed action and waits for a person. This
    is a real pause, not a poll: LangGraph's `interrupt` checkpoints the run,
    unwinds, and the API returns. Minutes later the user clicks Approve, the
    graph resumes from this exact point, and `interrupt` returns their answer.
    The browser is untouched throughout, because the session lives in the
    registry rather than in graph state.

``refuse``
    Turns a blocked or rejected action into a message the model can read and
    work around, instead of an exception that ends the run. An agent told
    "no, and here is why" usually finds another route; one that crashes cannot.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langgraph.types import interrupt

from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.db.models import TaskStatus
from app.events import EventType
from app.llm.base import Message, ToolCall

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]


def make_approve_node(runtime: AgentRuntime) -> NodeFn:
    """Build the approval-gate node."""

    async def approve(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0)
        call: ToolCall | None = state.get("pending_call")
        risk = state.get("pending_risk") or {}

        if call is None:  # defensive: routing should never send us here
            return {"approval_granted": False}

        payload = {
            "task_id": task_id,
            "step": step,
            "tool": call.name,
            "arguments": call.arguments,
            "reason": risk.get("reason", "This action may be irreversible."),
            "page_url": _current_url(runtime, task_id),
        }

        runtime.emit(
            task_id,
            EventType.APPROVAL_REQUIRED,
            payload["reason"],
            step=step,
            # `task_id` and `step` are already positional/keyword arguments of
            # `emit`; passing them again through **payload would collide.
            **{k: v for k, v in payload.items() if k not in {"task_id", "step"}},
        )
        await runtime.repository.update_task(task_id, status=TaskStatus.AWAITING_APPROVAL)

        # Execution stops here. Everything below runs only after the user
        # answers and the graph is resumed with a value.
        answer = interrupt(payload)
        granted = _interpret(answer)

        runtime.emit(
            task_id,
            EventType.APPROVAL_RESOLVED,
            "Approved by the user" if granted else "Rejected by the user",
            step=step,
            approved=granted,
            tool=call.name,
        )
        await runtime.repository.update_task(task_id, status=TaskStatus.RUNNING)

        return {"approval_granted": granted, "status": TaskStatus.RUNNING}

    return approve


def make_refuse_node(runtime: AgentRuntime) -> NodeFn:
    """Build the node that reports a blocked or rejected action to the model."""

    async def refuse(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0)
        call: ToolCall | None = state.get("pending_call")
        risk = state.get("pending_risk") or {}

        if call is None:
            return {"pending_call": None, "pending_risk": None}

        rejected_by_user = state.get("approval_granted") is False
        if rejected_by_user:
            reason = (
                "The user declined this action. Do not attempt it again. "
                "Either find a different way to reach the goal, or call finish "
                "and explain what you were unable to do."
            )
        else:
            reason = (
                f"This action is not permitted: {risk.get('reason', 'blocked by policy')} "
                "Find another way, or use ask_user if only the user can do it."
            )

        runtime.emit(
            task_id,
            EventType.LOG,
            f"Refused {call.name}: {reason}",
            step=step,
            tool=call.name,
            rejected_by_user=rejected_by_user,
        )
        await runtime.repository.record_tool_call(
            task_id=task_id,
            step=step,
            tool_name=call.name,
            arguments=call.arguments,
            ok=False,
            result=reason,
            duration_ms=0,
            risk=int(risk.get("risk", 0)),
            approved=False,
            page_url=_current_url(runtime, task_id),
        )

        return {
            "pending_call": None,
            "pending_risk": None,
            "approval_granted": None,
            "messages": [Message.tool_result(call, reason)],
        }

    return refuse


def _interpret(answer: object) -> bool:
    """Read a resume value as a yes or a no.

    Accepts several shapes because the value comes from an HTTP client:
    ``True``, ``"approve"``, ``{"approved": true}`` all mean yes. Anything
    unrecognised means no -- defaulting an ambiguous answer to "go ahead and
    spend the money" is the wrong failure mode.
    """
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, dict):
        for key in ("approved", "approve", "granted", "decision"):
            if key in answer:
                return _interpret(answer[key])
        return False
    if isinstance(answer, str):
        return answer.strip().lower() in {"yes", "y", "true", "approve", "approved", "ok"}
    return False


def _current_url(runtime: AgentRuntime, task_id: str) -> str:
    """URL the browser is on, for display in the approval prompt."""
    session = runtime.sessions.get(task_id)
    observation = session.last_observation if session else None
    return observation.url if observation else ""
