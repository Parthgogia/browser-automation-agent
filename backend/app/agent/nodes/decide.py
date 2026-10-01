"""Decision node: the model chooses the next action.

The decision model does not receive the full transcript. Browser agents make a
model call for every action, so repeatedly sending old DOM renderings, tool
schemas and verbose results is both expensive and a quick way to exhaust a
free provider's input-token quota. Instead each decision gets a compact,
reconstructed working context: goal, current plan, confirmed outcomes, recent
activity and the one current page rendering.

Two details are load-bearing:

* **Tool use is forced.** Free text here is always a dead end -- the loop has no
  way to act on "I think I should search for laptops". Requiring a function
  call turns "I should search" into `web_search(query=...)`.
* **The full transcript remains state, not prompt input.** It is still used for
  reflection, auditability and state recovery. The decision prompt is rebuilt
  from it deterministically, so losing old raw turns does not make the agent
  forget which approaches it has already tried.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.events import EventType
from app.llm.base import LLMError, Message, ToolCall

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]

#: A decision needs the current page, not a collection of stale DOM dumps.
_MAX_CURRENT_OBSERVATION_CHARS = 8_000

#: Earlier results are useful as durable facts, but large extracts and search
#: pages must not grow the prompt without bound.
_MAX_CONFIRMED_FACTS = 6
_MAX_FACT_CHARS = 320
_MAX_RESEARCH_RESULT_CHARS = 2_000
_MAX_ACTIVITY = 5
_MAX_ACTIVITY_CHARS = 360

#: This is intentionally conservative. It retains enough of a plan to steer a
#: multi-site task without allowing a verbose planner response to dominate the
#: next action.
_MAX_PLAN_STEPS = 6
_MAX_PLAN_STEP_CHARS = 240


def make_decide_node(runtime: AgentRuntime) -> NodeFn:
    """Build the decision node."""

    async def decide(state: AgentState) -> dict:
        task_id = state["task_id"]
        step = state.get("step", 0)

        messages = _decision_messages(state)
        tools = runtime.tools.specs(include=_relevant_tool_names(runtime, state))

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


def _decision_messages(state: AgentState) -> list[Message]:
    """Build the bounded prompt used for one browser decision.

    This intentionally emits only normal system/user turns. Provider-specific
    tool-result protocols require every historic function call and response to
    remain paired; rebuilding those protocols would retain precisely the
    repeated context this function is meant to remove.
    """
    messages = state.get("messages", [])
    observation = _latest_observation_message(messages)
    parts = [
        f"Goal: {state['goal']}",
        _render_understanding(state.get("understanding", "")),
        _render_plan(state.get("plan", [])),
        _render_memories(state.get("memories", "")),
        _render_confirmed_facts(messages),
        _render_recent_activity(messages),
        "Current page:\n" + _shorten(observation.content, _MAX_CURRENT_OBSERVATION_CHARS),
        "Choose exactly one next action using the available tools.",
    ]
    return [
        Message.system(SYSTEM_PROMPT),
        Message.user(
            "\n\n".join(part for part in parts if part),
            images=observation.images[:1],
        ),
    ]


def _render_understanding(understanding: str) -> str:
    if not understanding:
        return ""
    return f"Task interpretation: {_shorten(understanding, _MAX_PLAN_STEP_CHARS)}"


def _render_plan(plan: list[str]) -> str:
    if not plan:
        return ""
    steps = [
        f"{number}. {_shorten(step, _MAX_PLAN_STEP_CHARS)}"
        for number, step in enumerate(plan[:_MAX_PLAN_STEPS], start=1)
    ]
    return "Plan:\n" + "\n".join(steps)


def _render_memories(memories: str) -> str:
    if not memories:
        return ""
    return "Relevant saved context:\n" + _shorten(memories, 1_000)


def _render_confirmed_facts(messages: list[Message]) -> str:
    """Keep useful successful outcomes from before the recent activity window."""
    results, research_result = _successful_results(messages)
    older_results = results[:-_MAX_ACTIVITY]
    if not older_results and not research_result:
        return ""
    parts: list[str] = []
    if older_results:
        parts.append(
            "Confirmed earlier outcomes:\n"
            + "\n".join(f"- {result}" for result in older_results[-_MAX_CONFIRMED_FACTS:])
        )
    if research_result:
        parts.append(f"Latest detailed research:\n{research_result}")
    return "\n\n".join(parts)


def _successful_results(messages: list[Message]) -> tuple[list[str], str | None]:
    """Keep one larger research result and compact summaries of other outcomes."""
    results: list[str] = []
    research_result: str | None = None
    last_action = ""
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            last_action = message.tool_calls[-1].name
        elif message.role == "tool" and not message.content.startswith("FAILED:"):
            if last_action in {"extract_text", "web_search"}:
                research_result = _shorten(message.content, _MAX_RESEARCH_RESULT_CHARS)
            else:
                results.append(_shorten(message.content, _MAX_FACT_CHARS))
    return results, research_result


def _render_recent_activity(messages: list[Message]) -> str:
    """Render only the last few attempted actions and their outcomes."""
    lines: list[str] = []
    for message in messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                lines.append(f"Action: {_describe(call)}")
        elif message.role == "tool":
            lines.append(f"Outcome: {_shorten(message.content, _MAX_ACTIVITY_CHARS)}")
    if not lines:
        return "No actions have run yet."
    return "Recent activity:\n" + "\n".join(lines[-(_MAX_ACTIVITY * 2) :])


def _latest_observation_message(messages: list[Message]) -> Message:
    for message in reversed(messages):
        if message.role == "user" and "Page content" in message.content:
            return message
    return Message.user(
        "No page observation is available yet. Use a navigation or search tool."
    )


def _relevant_tool_names(runtime: AgentRuntime, state: AgentState) -> set[str]:
    """Advertise the smallest useful vocabulary for the observed page.

    The agent can still reach every capability when it becomes relevant, but a
    page with no file input should not pay to send an upload-file schema.
    """
    names = {
        "navigate",
        "go_back",
        "reload",
        "web_search",
        "extract_text",
        "wait",
        "finish",
        "ask_user",
        "scroll",
        "scroll_to_text",
        "recall",
    }
    session = runtime.sessions.get(state["task_id"])
    observation = session.last_observation if session else None
    if observation is None:
        return names

    tags = {element.tag.lower() for element in observation.elements}
    if observation.elements:
        names.update({"click", "hover", "press_key"})
    if tags & {"input", "textarea"}:
        names.add("type_text")
    if "select" in tags:
        names.add("select_option")
    if "input" in tags and any(
        "type=\"file\"" in element.attrs for element in observation.elements
    ):
        names.add("upload_file")
    if len(observation.tabs) > 1:
        names.update({"switch_tab", "close_tab"})
    if any(term in state["goal"].lower() for term in ("compare", "versus", " vs ", "both ")):
        names.add("new_tab")
    return names


def _trim(messages: list[Message]) -> list[Message]:
    """Compatibility helper retained for callers that need transcript trimming.

    Decisions now use :func:`_decision_messages`; this helper retains the
    former compacting behaviour for integrations that still need it.
    """
    observations_kept = 0
    images_kept = 0
    trimmed: list[Message] = []

    for message in reversed(messages):
        is_observation = message.role == "user" and "Page content" in message.content
        if is_observation:
            observations_kept += 1
            if observations_kept > 2:
                header = [
                    line
                    for line in message.content.splitlines()[:4]
                    if line.startswith(("URL:", "Title:", "[Step"))
                ]
                trimmed.append(
                    Message.user("\n".join(header) + "\n[earlier page rendering omitted]")
                )
                continue
        if message.images:
            images_kept += 1
            if images_kept > 1:
                message = Message(
                    role=message.role,
                    content=message.content,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
        trimmed.append(message)
    return list(reversed(trimmed))
