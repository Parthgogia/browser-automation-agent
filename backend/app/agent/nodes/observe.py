"""Observation node: look at the page and put what we see into the transcript.

This runs before every decision, and it is where the perception strategy is
actually applied: build the numbered DOM index, always capture a screenshot for
the human watching, and attach that screenshot to the model's prompt only when
the text rendering looks inadequate (see `AgentRuntime.should_attach_screenshot`).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.prompts import render_observation_message
from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.events import EventType
from app.llm.base import Message

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]


def make_observe_node(runtime: AgentRuntime) -> NodeFn:
    """Build the observation node."""

    async def observe(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0) + 1

        session = await runtime.browser_for(task_id, state.get("profile"))

        try:
            observation = await session.observe()
        except Exception as exc:  # noqa: BLE001 - a broken page must not end the run
            logger.warning("Observation failed: %s", exc)
            runtime.emit(task_id, EventType.ERROR, f"Could not read the page: {exc}", step=step)
            return {
                "step": step,
                "messages": [
                    Message.user(
                        f"[Step {step}] The page could not be read ({exc}). "
                        "Try reloading, or navigating somewhere else."
                    )
                ],
            }

        # Always persist a screenshot: it is the live preview the human watches,
        # independent of whether the model is shown it.
        screenshot_url = None
        if observation.screenshot_bytes:
            screenshot_url = runtime.save_screenshot(
                task_id, step, observation.screenshot_bytes
            )
            observation.screenshot_path = screenshot_url
            runtime.emit(
                task_id,
                EventType.SCREENSHOT,
                observation.title or observation.url,
                step=step,
                url=screenshot_url,
                page_url=observation.url,
            )

        runtime.emit(
            task_id,
            EventType.OBSERVATION,
            f"{observation.title or 'Untitled'} - {observation.url}",
            step=step,
            url=observation.url,
            title=observation.title,
            element_count=len(observation.elements),
            screenshot=screenshot_url,
        )

        attach_image = runtime.should_attach_screenshot(
            observation, consecutive_failures=state.get("consecutive_failures", 0)
        )
        images = (
            [observation.screenshot_bytes]
            if attach_image and observation.screenshot_bytes
            else []
        )

        text = render_observation_message(
            observation, step, runtime.settings.agent_max_steps
        )
        if images:
            text += (
                "\n\nA screenshot of the page is attached. The numbers drawn on it "
                "are the same element numbers listed above."
            )

        return {"step": step, "messages": [Message.user(text, images=images)]}

    return observe
