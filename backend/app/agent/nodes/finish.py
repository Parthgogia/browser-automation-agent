"""Completion node: settle the run and tell the user what happened.

Reached from four different directions -- the model called `finish`, it called
`ask_user`, the step budget ran out, or something failed hard -- so its job is
to normalise all of them into one final answer and one terminal status.

It is also where a successful run writes back to long-term memory, which is the
only point in the system where the agent learns anything that outlives the run.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.db.models import TaskStatus
from app.events import EventType

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]


def make_finish_node(runtime: AgentRuntime) -> NodeFn:
    """Build the completion node."""

    async def finish(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0)
        status = state.get("status", TaskStatus.COMPLETED)

        # Reaching this node without a status set means the loop ended on its
        # own terms: the step budget, not a decision by the model.
        if status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.AWAITING_INPUT,
            TaskStatus.CANCELLED,
        }:
            status = TaskStatus.FAILED

        result = state.get("result") or state.get("error") or _budget_message(state, runtime)
        success = state.get("success")

        if status is TaskStatus.AWAITING_INPUT:
            runtime.emit(
                task_id,
                EventType.TASK_COMPLETED,
                result,
                step=step,
                awaiting_input=True,
                question=state.get("question"),
            )
        elif status is TaskStatus.COMPLETED and success:
            runtime.emit(task_id, EventType.TASK_COMPLETED, result, step=step, success=True)
            await _learn(runtime, state)
        else:
            runtime.emit(
                task_id,
                EventType.TASK_FAILED,
                result,
                step=step,
                success=False,
                error=state.get("error"),
            )

        await runtime.repository.update_task(
            task_id,
            status=status,
            result_summary=result,
            success=success,
            error=state.get("error"),
            pending_question=state.get("question"),
            steps_taken=step,
        )

        return {"status": status, "result": result, "success": success}

    return finish


def _budget_message(state: AgentState, runtime: AgentRuntime) -> str:
    """Explain a run that ran out of budget without finishing.

    Which ceiling was hit tells the user something different each time, so
    the message names it rather than saying 'the agent gave up'.
    """
    steps = state.get("step", 0)
    if state.get("total_failures", 0) >= runtime.settings.agent_max_failures:
        return (
            f"Gave up after {steps} steps: too many actions failed "
            f"({state.get('total_failures', 0)} in total). The site may be blocking "
            "automation, or the task may need a login the agent does not have."
        )
    if state.get("reflections", 0) >= runtime.settings.agent_max_reflections:
        return (
            f"Gave up after {steps} steps: the agent rethought its approach "
            f"{state.get('reflections', 0)} times without making progress."
        )
    return (
        f"Stopped after {steps} steps without completing the task "
        f"(the limit is {runtime.settings.agent_max_steps}). "
        "Try narrowing the request, or raise AGENT_MAX_STEPS."
    )


async def _learn(runtime: AgentRuntime, state: AgentState) -> None:
    """Record the outcome of a successful run as a memory.

    Only outcomes, and only on success. Storing failures would teach the agent
    its own mistakes; storing page contents would fill memory with noise that
    drowns out the handful of facts that actually matter.
    """
    if runtime.memory is None or not runtime.memory.available:
        return
    summary = (state.get("result") or "").strip()
    if not summary:
        return
    try:
        await runtime.memory.add(
            f"Task '{state['goal']}' was completed. Outcome: {summary[:400]}",
            category="task_outcome",
            task_id=state["task_id"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not store task outcome: %s", exc)
