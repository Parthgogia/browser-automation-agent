"""The state object that flows through the LangGraph.

One rule governs what may live here: **it must describe the run, not own the
resources.** The graph is checkpointed so a run can pause at an approval gate
and resume minutes later, possibly in a different request. A live Chromium
connection or an open database session cannot survive that, so the state holds
a `task_id` and the nodes look live resources up by it.

What the state *does* own is the conversation. `messages` is the agent's
working memory for this run -- the plan, every observation, every tool call and
every result, in order. It is the single most important field here: the quality
of the agent is largely the quality of what ends up in this list.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from app.llm.base import Message, ToolCall


def append_messages(
    existing: list[Message] | None, incoming: list[Message] | None
) -> list[Message]:
    """Reducer for the `messages` channel: nodes append, never replace.

    LangGraph merges each node's returned state through these reducers, so a
    node that emits ``{"messages": [msg]}`` adds one turn rather than
    clobbering the transcript.
    """
    return (existing or []) + (incoming or [])


class AgentState(TypedDict, total=False):
    """Everything the graph knows about a run in progress."""

    # ---- identity, fixed for the whole run -------------------------------
    task_id: str
    goal: str
    profile: str

    # ---- planning --------------------------------------------------------
    #: The model's restatement of what the user actually wants. Cheap to
    #: produce and a useful early signal that the request was misread.
    understanding: str
    #: Human-readable step list. Guidance for the model, not a program: the
    #: agent is free to deviate, and `reflect` rewrites this when reality
    #: diverges from the plan.
    plan: list[str]
    #: Relevant long-term memories, rendered once at planning time.
    memories: str

    # ---- conversation ----------------------------------------------------
    messages: Annotated[list[Message], append_messages]

    # ---- loop control ----------------------------------------------------
    step: int
    #: Failures since the last success. Drives reflection.
    consecutive_failures: int
    #: Failures over the whole run. Never reset; drives the give-up ceiling,
    #: which `consecutive_failures` cannot because reflection clears it.
    total_failures: int
    #: How many times the agent has stopped to rethink.
    reflections: int
    #: Set when the model has chosen an action that has not yet run.
    pending_call: ToolCall | None
    #: Safety verdict for `pending_call`, as {"risk": int, "reason": str}.
    pending_risk: dict[str, Any] | None
    #: Filled by the approval gate on resume.
    approval_granted: bool | None

    # ---- outcome ---------------------------------------------------------
    status: str
    result: str | None
    success: bool | None
    #: Set when the run ended by asking the user a question.
    question: str | None
    error: str | None


def initial_state(task_id: str, goal: str, profile: str) -> AgentState:
    """A fresh state for a new run."""
    return AgentState(
        task_id=task_id,
        goal=goal,
        profile=profile,
        understanding="",
        plan=[],
        memories="",
        messages=[],
        step=0,
        consecutive_failures=0,
        total_failures=0,
        reflections=0,
        pending_call=None,
        pending_risk=None,
        approval_granted=None,
        status="pending",
        result=None,
        success=None,
        question=None,
        error=None,
    )
