"""The agent graph.

    START
      |
      v
    plan ---------> observe <---------------+------------------+
                       |                    |                  |
                       v                    |                  |
                    decide                  |                  |
                       |                    |                  |
        +--------------+--------------+     |                  |
        |              |              |     |                  |
     approve        refuse         execute  |                  |
        |              |              |     |                  |
   granted? --no-------+              +-----+ (continue)       |
        |                             |                        |
       yes                            +--> reflect ------------+
        |                             |
        +-----------> execute         +--> finish --> END

Everything here is a routing decision, and the routing *is* the agent's
character. Three choices are worth calling out:

* **Observe before every decision, never after.** The model always reasons over
  a page rendering taken moments earlier, so element indices are fresh. The
  alternative -- act, then observe, then decide from a stale index -- is the
  single most common source of "element not found" loops.
* **Refusals return to `decide`, not to `observe`.** A blocked action did not
  touch the page, so re-reading it would burn a step and a screenshot to learn
  nothing.
* **Reflection sits between execution and the next observation**, so a revised
  plan is in the transcript *before* the model next looks at the page.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    make_approve_node,
    make_decide_node,
    make_execute_node,
    make_finish_node,
    make_observe_node,
    make_plan_node,
    make_reflect_node,
    make_refuse_node,
    should_reflect,
)
from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.db.models import TaskStatus
from app.safety.policy import RiskLevel

logger = logging.getLogger(__name__)

#: Statuses that mean the run is over and should route straight to `finish`.
_TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.AWAITING_INPUT,
    "completed",
    "failed",
    "cancelled",
    "awaiting_input",
}


def build_graph(runtime: AgentRuntime, *, checkpointer=None):
    """Compile the agent graph against `runtime`.

    A checkpointer is required for the approval interrupt to work: `interrupt`
    saves the run so it can be resumed from the same point after the human
    answers. `InMemorySaver` is the default, which means pending approvals do
    not survive a server restart -- fine for a single-process local tool, and
    swappable for `langgraph-checkpoint-postgres` when it needs to be durable.
    """
    graph = StateGraph(AgentState)

    graph.add_node("plan", make_plan_node(runtime))
    graph.add_node("observe", make_observe_node(runtime))
    graph.add_node("decide", make_decide_node(runtime))
    graph.add_node("approve", make_approve_node(runtime))
    graph.add_node("refuse", make_refuse_node(runtime))
    graph.add_node("execute", make_execute_node(runtime))
    graph.add_node("reflect", make_reflect_node(runtime))
    graph.add_node("finish", make_finish_node(runtime))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "observe")
    graph.add_edge("observe", "decide")

    graph.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"approve": "approve", "refuse": "refuse", "execute": "execute", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "approve",
        _route_after_approval,
        {"execute": "execute", "refuse": "refuse"},
    )
    # A refusal leaves the page untouched, so the model can choose again
    # immediately without paying for another observation.
    graph.add_edge("refuse", "decide")

    graph.add_conditional_edges(
        "execute",
        _route_after_execute(runtime),
        {"finish": "finish", "reflect": "reflect", "observe": "observe"},
    )
    graph.add_edge("reflect", "observe")
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# ------------------------------------------------------------------ routing --


def _route_after_decide(
    state: AgentState,
) -> Literal["approve", "refuse", "execute", "finish"]:
    """Where a proposed action goes: to a human, to the bin, or to the browser."""
    if state.get("status") in _TERMINAL_STATUSES:
        return "finish"

    call = state.get("pending_call")
    if call is None:
        return "finish"

    risk = (state.get("pending_risk") or {}).get("risk", int(RiskLevel.SAFE))
    if risk >= int(RiskLevel.BLOCKED):
        return "refuse"
    if risk >= int(RiskLevel.CONFIRM):
        return "approve"
    return "execute"


def _route_after_approval(state: AgentState) -> Literal["execute", "refuse"]:
    """Honour the human's answer."""
    return "execute" if state.get("approval_granted") else "refuse"


def _route_after_execute(runtime: AgentRuntime):
    """Build the post-execution router.

    Closes over the runtime because the step ceiling is configuration, and a
    router that reads it from settings each time stays correct when a test
    lowers the limit.
    """

    def route(state: AgentState) -> Literal["finish", "reflect", "observe"]:
        if state.get("status") in _TERMINAL_STATUSES:
            return "finish"
        if state.get("step", 0) >= runtime.settings.agent_max_steps:
            logger.info("Step budget exhausted for task %s", state.get("task_id"))
            return "finish"
        # Two independent give-up conditions. Neither can use
        # `consecutive_failures`, which reflection resets by design.
        if state.get("total_failures", 0) >= runtime.settings.agent_max_failures:
            logger.info("Failure budget exhausted for task %s", state.get("task_id"))
            return "finish"
        if state.get("reflections", 0) >= runtime.settings.agent_max_reflections:
            logger.info("Reflection budget exhausted for task %s", state.get("task_id"))
            return "finish"
        if should_reflect(state):
            return "reflect"
        return "observe"

    return route
