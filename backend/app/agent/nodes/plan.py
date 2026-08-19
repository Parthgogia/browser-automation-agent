"""Planning node: turn a sentence into a short, concrete plan.

Why plan at all, when the act loop could just start clicking? Two reasons that
show up immediately in practice:

* **Multi-source tasks fall apart without one.** "Compare MacBook prices on
  Amazon and Flipkart" needs the agent to remember there is a second site after
  it has finished with the first. A plan in the transcript is what carries that.
* **It surfaces misunderstandings in one cheap call** rather than after fifteen
  browser steps in the wrong direction.

The plan is guidance, not a program. `reflect` rewrites it whenever reality
turns out to differ.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.parsing import parse_json_object, parse_string_list
from app.agent.prompts import PLANNER_PROMPT, SYSTEM_PROMPT, render_task_message
from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.db.models import TaskStatus
from app.events import EventType
from app.llm.base import Message

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]


def make_plan_node(runtime: AgentRuntime) -> NodeFn:
    """Build the planning node."""

    async def plan(state: AgentState) -> dict:
        task_id, goal = state["task_id"], state["goal"]

        runtime.emit(task_id, EventType.TASK_STARTED, goal, step=0)
        await runtime.repository.update_task(task_id, status=TaskStatus.PLANNING)

        memories = ""
        if runtime.memory is not None and runtime.memory.available:
            try:
                memories = await runtime.memory.recall_for_goal(goal)
            except Exception as exc:  # noqa: BLE001 - memory is never critical
                logger.debug("Memory recall failed: %s", exc)

        understanding, steps = await _generate_plan(runtime, goal, memories)

        runtime.emit(
            task_id,
            EventType.PLAN_CREATED,
            understanding or f"Planning: {goal}",
            step=0,
            plan=steps,
            understanding=understanding,
        )
        await runtime.repository.update_task(
            task_id, status=TaskStatus.RUNNING, plan=steps
        )

        return {
            "understanding": understanding,
            "plan": steps,
            "memories": memories,
            "status": TaskStatus.RUNNING,
            "messages": [
                Message.system(SYSTEM_PROMPT),
                Message.user(render_task_message(goal, understanding, steps, memories)),
            ],
        }

    return plan


async def _generate_plan(
    runtime: AgentRuntime, goal: str, memories: str
) -> tuple[str, list[str]]:
    """Ask the model for a plan, degrading gracefully if it does not comply."""
    memory_block = f"\n{memories}\n" if memories else ""
    prompt = PLANNER_PROMPT.format(goal=goal, memories=memory_block)

    try:
        response = await runtime.llm.complete(
            [Message.user(prompt)],
            # The fast model is enough for a five-line plan and keeps the
            # first response quick, which matters for perceived latency.
            model=runtime.settings.llm_fast_model,
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        # A failed plan is not a failed task: the act loop can work from the
        # goal alone. Losing the run here would be a much worse outcome.
        logger.warning("Planning call failed (%s); continuing without a plan", exc)
        return "", []

    parsed = parse_json_object(response.text) or {}
    understanding = str(parsed.get("understanding", "")).strip()
    steps = parse_string_list(parsed.get("steps"))

    if not steps:
        logger.debug("Planner returned no usable steps; proceeding with the goal alone")
    return understanding, steps
