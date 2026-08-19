"""Reflection node: stop and rethink when the current approach is not working.

This is the difference between an agent that recovers and one that loops. Left
alone, a model that has failed to click something three times will try a fourth
time -- each failure looks locally like bad luck. Reflection breaks that by
pulling the model out of the step-by-step frame and asking a different
question: given everything so far, is this approach viable at all?

It runs only when the agent is actually in trouble (see `should_reflect`),
because it costs a full model call and interrupting a run that is going fine
makes it worse, not better.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.parsing import parse_json_object, parse_string_list
from app.agent.prompts import REFLECTION_PROMPT, render_reflection_message
from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.events import EventType
from app.llm.base import Message

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]

#: Consecutive failures before we stop and rethink. Two is deliberate: one
#: failure is normal noise on the web, two in a row is a pattern.
REFLECT_AFTER_FAILURES = 2

#: How many recent transcript entries the reflection prompt is shown.
_RECENT_WINDOW = 8


def should_reflect(state: AgentState) -> bool:
    """Is the run in enough trouble to justify a reflection call?"""
    return state.get("consecutive_failures", 0) >= REFLECT_AFTER_FAILURES


def _spent(state: AgentState) -> dict:
    """State change common to every exit from reflection.

    The consecutive-failure streak resets because the agent has just been
    given new direction, and `reflections` increments so that the give-up
    ceiling is reachable no matter which path reflection took.
    """
    return {
        "consecutive_failures": 0,
        "reflections": state.get("reflections", 0) + 1,
    }


def make_reflect_node(runtime: AgentRuntime) -> NodeFn:
    """Build the reflection node."""

    async def reflect(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0)
        failures = state.get("consecutive_failures", 0)

        prompt = REFLECTION_PROMPT.format(
            goal=state["goal"],
            plan=_render_plan(state.get("plan", [])),
            recent=_render_recent(state.get("messages", [])),
            failures=failures,
        )

        try:
            response = await runtime.llm.complete(
                [Message.user(prompt)],
                model=runtime.settings.llm_fast_model,
                temperature=0.4,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a run on reflection
            logger.warning("Reflection call failed: %s", exc)
            return _spent(state)

        parsed = parse_json_object(response.text) or {}
        assessment = str(parsed.get("assessment", "")).strip()
        advice = str(parsed.get("advice", "")).strip()
        steps = parse_string_list(parsed.get("steps"))

        if not (assessment or advice):
            # Nothing usable came back. Still counts as an attempt, so the
            # give-up ceiling can be reached instead of reflecting forever.
            return _spent(state)

        runtime.emit(
            task_id,
            EventType.PLAN_REVISED,
            assessment or "Revised approach",
            step=step,
            plan=steps,
            advice=advice,
        )
        if steps:
            await runtime.repository.update_task(task_id, plan=steps)

        return {
            **_spent(state),
            "plan": steps or state.get("plan", []),
            "messages": [Message.user(render_reflection_message(assessment, advice))],
        }

    return reflect


def _render_plan(plan: list[str]) -> str:
    if not plan:
        return "(no plan was made)"
    return "\n".join(f"{n}. {step}" for n, step in enumerate(plan, start=1))


def _render_recent(messages: list[Message]) -> str:
    """Compact rendering of recent activity, without the page dumps."""
    lines: list[str] = []
    for message in messages[-_RECENT_WINDOW:]:
        if message.role == "assistant":
            for call in message.tool_calls:
                lines.append(f"Tried: {call.name}({call.arguments})")
            if message.content and not message.tool_calls:
                lines.append(f"Thought: {message.content[:200]}")
        elif message.role == "tool":
            lines.append(f"Result: {message.content[:300]}")
    return "\n".join(lines) or "(nothing yet)"
