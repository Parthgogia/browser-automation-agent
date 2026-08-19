"""Decision node: the model chooses the next action.

This is the only place the agent's reasoning model runs against the full
transcript. Two details are load-bearing:

* **Tool use is forced.** Free text here is always a dead end -- the loop has no
  way to act on "I think I should search for laptops". Requiring a function
  call turns "I should search" into `web_search(query=...)`.
* **The transcript is trimmed, not truncated.** Old page renderings are the
  bulk of the context and the least useful part of it: what the page looked
  like nine steps ago is irrelevant, while *what the agent did* nine steps ago
  is essential. `_trim` drops the former and keeps the latter.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.events import EventType
from app.llm.base import LLMError, Message, ToolCall

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]

#: How many recent page renderings to keep in full. Beyond this, an observation
#: is replaced by a one-line placeholder.
_FULL_OBSERVATIONS = 2

#: Images are by far the heaviest part of the context; only the newest is kept.
_FULL_IMAGES = 1


def make_decide_node(runtime: AgentRuntime) -> NodeFn:
    """Build the decision node."""

    async def decide(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0)

        messages = _trim(state.get("messages", []))
        tools = runtime.tools.specs()

        try:
            response = await runtime.llm.complete(
                messages, tools=tools, force_tool_use=True
            )
        except LLMError as exc:
            logger.error("Reasoning call failed: %s", exc)
            runtime.emit(task_id, EventType.ERROR, str(exc), step=step)
            return {
                "status": "failed",
                "error": f"The language model could not be reached: {exc}",
                "success": False,
            }

        if response.text:
            runtime.emit(task_id, EventType.THOUGHT, response.text, step=step)

        if not response.tool_calls:
            # `force_tool_use` makes this rare, but a model can still answer
            # with prose. Treat it as an answer rather than discarding it.
            logger.info("Model returned no tool call; treating its text as the answer")
            return {
                "pending_call": None,
                "pending_risk": None,
                "status": "completed",
                "result": response.text or "The agent stopped without a result.",
                "success": bool(response.text),
                "messages": [Message.assistant(response.text)],
            }

        # One action per step. A model that proposes three at once has not seen
        # the result of the first, so the rest are speculation.
        call = response.tool_calls[0]
        decision = _classify(runtime, state, call)

        runtime.emit(
            task_id,
            EventType.TOOL_CALL,
            _describe(call),
            step=step,
            tool=call.name,
            arguments=call.arguments,
            risk=int(decision.risk),
            reason=decision.reason,
        )

        return {
            "pending_call": call,
            "pending_risk": {"risk": int(decision.risk), "reason": decision.reason},
            "approval_granted": None,
            "messages": [Message.assistant(response.text, tool_calls=[call])],
        }

    return decide


def _classify(runtime: AgentRuntime, state: AgentState, call: ToolCall):
    """Run the safety policy over a proposed call.

    The element's *label* is what makes classification work -- "Place your
    order" is the signal, not the fact that a click was requested -- so it is
    resolved from the latest observation before evaluating.
    """
    session = runtime.sessions.get(state["task_id"])
    observation = session.last_observation if session else None

    label = ""
    index = call.arguments.get("index")
    if observation is not None and isinstance(index, int):
        element = observation.element_by_index(index)
        if element is not None:
            label = f"{element.label} {element.attrs}"

    return runtime.safety.evaluate(
        call.name,
        call.arguments,
        element_label=label,
        current_url=observation.url if observation else "",
    )


def _describe(call: ToolCall) -> str:
    """One-line human summary of a tool call, for the event stream."""
    if not call.arguments:
        return call.name
    rendered = ", ".join(
        f"{key}={_shorten(value)}" for key, value in call.arguments.items()
    )
    return f"{call.name}({rendered})"


def _shorten(value: object, limit: int = 60) -> str:
    text = str(value)
    return f"{text[:limit]}..." if len(text) > limit else text


def _trim(messages: list[Message]) -> list[Message]:
    """Keep the transcript inside a sensible context budget.

    Walking backwards, the newest observations survive intact; older ones are
    collapsed to a stub that preserves the URL. Assistant turns and tool
    results are always kept in full -- they are the agent's memory of what it
    has already tried, and dropping them causes it to loop.
    """
    observations_kept = 0
    images_kept = 0
    trimmed: list[Message] = []

    for message in reversed(messages):
        is_observation = message.role == "user" and "Page content" in message.content

        if is_observation:
            observations_kept += 1
            if observations_kept > _FULL_OBSERVATIONS:
                trimmed.append(Message.user(_stub(message.content)))
                continue

        if message.images:
            images_kept += 1
            if images_kept > _FULL_IMAGES:
                message = Message(
                    role=message.role,
                    content=message.content,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                    images=[],
                )

        trimmed.append(message)

    return list(reversed(trimmed))


def _stub(content: str) -> str:
    """Collapse an old observation to its identifying header lines."""
    header = [
        line
        for line in content.splitlines()[:4]
        if line.startswith(("URL:", "Title:", "[Step"))
    ]
    return "\n".join(header) + "\n[earlier page rendering omitted]"
