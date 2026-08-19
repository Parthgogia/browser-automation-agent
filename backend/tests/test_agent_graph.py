"""End-to-end graph tests, using the mock LLM and a fake browser.

These are the tests that would actually catch a regression in the agent's
behaviour: they run the real graph, the real router, the real safety policy and
the real tool dispatcher, and assert on what the agent *did*.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.nodes.approve import _interpret
from app.agent.state import initial_state
from app.db.models import TaskStatus
from app.events import EventType
from app.llm.base import LLMResponse, ToolCall

TASK_ID = "task-1"


def _config(runtime, thread: str = TASK_ID) -> dict:
    return {
        "configurable": {"thread_id": thread},
        "recursion_limit": runtime.settings.agent_max_steps * 8 + 50,
    }


class ScriptedProvider:
    """An LLM that returns a fixed sequence of tool calls.

    Lets a test state exactly what the model decides, which is the only way to
    exercise a specific graph path deterministically.
    """

    name = "scripted"

    def __init__(self, script: list[ToolCall]) -> None:
        self._script = list(script)
        self.calls = 0

    async def complete(self, messages, *, tools=None, **_kwargs) -> LLMResponse:
        self.calls += 1
        if not tools:  # the planner asks for prose
            return LLMResponse(text='{"understanding": "test", "steps": ["do it"]}')
        if not self._script:
            return LLMResponse(
                tool_calls=[ToolCall(name="finish", arguments={"summary": "ran out of script"})]
            )
        return LLMResponse(text="thinking", tool_calls=[self._script.pop(0)])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    @property
    def embedding_dimensions(self) -> int:
        return 8


# --------------------------------------------------------------- happy path --


async def test_agent_completes_a_task(runtime) -> None:
    """A scripted run navigates, reads and finishes, and reports its summary."""
    runtime.llm = ScriptedProvider(
        [
            ToolCall(name="navigate", arguments={"url": "https://example.com"}),
            ToolCall(name="extract_text", arguments={}),
            ToolCall(
                name="finish",
                arguments={"summary": "The laptop costs 119990.", "success": True},
            ),
        ]
    )
    graph = build_graph(runtime)

    final = await graph.ainvoke(
        initial_state(TASK_ID, "find the laptop price", "test"), config=_config(runtime)
    )

    assert final["status"] == TaskStatus.COMPLETED
    assert final["success"] is True
    assert "119990" in final["result"]


async def test_plan_is_recorded_and_streamed(runtime) -> None:
    """The plan reaches both the state and the event stream."""
    runtime.llm = ScriptedProvider(
        [ToolCall(name="finish", arguments={"summary": "done", "success": True})]
    )
    graph = build_graph(runtime)

    await graph.ainvoke(
        initial_state(TASK_ID, "do something", "test"), config=_config(runtime)
    )

    types = [event.type for event in runtime.events.history(TASK_ID)]
    assert EventType.TASK_STARTED in types
    assert EventType.PLAN_CREATED in types
    assert EventType.TOOL_CALL in types
    assert EventType.TASK_COMPLETED in types


# ------------------------------------------------------------- the mock LLM --


async def test_mock_provider_drives_a_full_run(runtime) -> None:
    """The keyless default path actually completes a task.

    This is the guarantee that `LLM_PROVIDER=mock` is a usable smoke test and
    not a stub: with no key and no network, the agent still searches, reads and
    reports.
    """
    graph = build_graph(runtime)

    final = await graph.ainvoke(
        initial_state(TASK_ID, "find wireless headphones", "test"), config=_config(runtime)
    )

    assert final["status"] in {TaskStatus.COMPLETED, TaskStatus.FAILED}
    assert final["result"]
    tools_used = [
        event.data.get("tool")
        for event in runtime.events.history(TASK_ID)
        if event.type is EventType.TOOL_CALL
    ]
    assert "navigate" in tools_used
    assert "finish" in tools_used


# ------------------------------------------------------- human in the loop --


async def test_risky_action_pauses_for_approval(runtime) -> None:
    """A purchase button stops the graph instead of being clicked."""
    runtime.llm = ScriptedProvider(
        [ToolCall(name="click", arguments={"index": 2})]  # "Place your order"
    )
    graph = build_graph(runtime)

    result = await graph.ainvoke(
        initial_state(TASK_ID, "buy the laptop", "test"), config=_config(runtime)
    )

    assert result.get("__interrupt__"), "the graph should have paused"
    payload = result["__interrupt__"][0].value
    assert payload["tool"] == "click"
    assert "irreversible" in payload["reason"].lower()

    events = [e for e in runtime.events.history(TASK_ID) if e.type is EventType.APPROVAL_REQUIRED]
    assert len(events) == 1


async def test_approval_lets_the_action_through(runtime, fake_session) -> None:
    """Answering yes resumes the run and the click actually happens."""
    runtime.llm = ScriptedProvider(
        [
            ToolCall(name="click", arguments={"index": 2}),
            ToolCall(name="finish", arguments={"summary": "ordered", "success": True}),
        ]
    )
    graph = build_graph(runtime)
    config = _config(runtime)

    await graph.ainvoke(initial_state(TASK_ID, "buy it", "test"), config=config)
    final = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert ("click", (), {}) in fake_session.calls
    assert final["status"] == TaskStatus.COMPLETED


async def test_rejection_is_reported_to_the_model_not_executed(runtime, fake_session) -> None:
    """Answering no keeps the click from happening and tells the model why."""
    runtime.llm = ScriptedProvider(
        [
            ToolCall(name="click", arguments={"index": 2}),
            ToolCall(name="finish", arguments={"summary": "the user declined", "success": False}),
        ]
    )
    graph = build_graph(runtime)
    config = _config(runtime)

    await graph.ainvoke(initial_state(TASK_ID, "buy it", "test"), config=config)
    final = await graph.ainvoke(Command(resume={"approved": False}), config=config)

    assert ("click", (), {}) not in fake_session.calls
    assert final["success"] is False

    # The refusal must reach the transcript, or the model will simply retry.
    tool_messages = [m for m in final["messages"] if m.role == "tool"]
    assert any("declined" in m.content.lower() for m in tool_messages)


@pytest.mark.parametrize("answer", [False, "no", "maybe", {}, {"approved": False}, None])
def test_only_an_explicit_yes_counts_as_approval(answer) -> None:
    """The resume value is parsed defensively.

    Anything that is not clearly affirmative -- a typo, an empty body, a
    missing field -- must read as "no". Defaulting an ambiguous answer to
    "go ahead and spend the money" is the wrong way to fail.
    """
    assert _interpret(answer) is False


@pytest.mark.parametrize("answer", [True, "yes", "Approve", {"approved": True}])
def test_affirmative_answers_are_recognised(answer) -> None:
    assert _interpret(answer) is True


@pytest.mark.parametrize("answer", [False, "no", {}, "maybe"])
async def test_ambiguous_answers_do_not_click(runtime, fake_session, answer) -> None:
    """The same defensiveness, verified through the real graph.

    `None` is excluded only because LangGraph cannot currently resume on a bare
    `None`; the API never produces one, and `_interpret` covers it above.
    """
    runtime.llm = ScriptedProvider(
        [
            ToolCall(name="click", arguments={"index": 2}),
            ToolCall(name="finish", arguments={"summary": "stopped", "success": False}),
        ]
    )
    graph = build_graph(runtime)
    config = _config(runtime, thread=f"task-{answer}")

    await graph.ainvoke(initial_state(TASK_ID, "buy it", "test"), config=config)
    await graph.ainvoke(Command(resume=answer), config=config)

    assert ("click", (), {}) not in fake_session.calls


# ------------------------------------------------------------ blocked paths --


async def test_credential_typing_is_blocked_without_asking(runtime, fake_session) -> None:
    """A password field is refused outright -- there is no approval for it."""
    runtime.llm = ScriptedProvider(
        [
            ToolCall(name="type_text", arguments={"index": 0, "text": "hunter2"}),
            ToolCall(name="finish", arguments={"summary": "cannot log in", "success": False}),
        ]
    )
    # Relabel the input so it reads as a password field.
    fake_session._observation.elements[0].label = "Password"
    fake_session._observation.elements[0].attrs = 'type="password"'

    graph = build_graph(runtime)
    final = await graph.ainvoke(
        initial_state(TASK_ID, "log in", "test"), config=_config(runtime)
    )

    assert not any(call[0] == "type" for call in fake_session.calls)
    assert final["status"] == TaskStatus.FAILED
    # No approval prompt: this is a hard block, not a question for the user.
    assert not [
        e for e in runtime.events.history(TASK_ID) if e.type is EventType.APPROVAL_REQUIRED
    ]


# ------------------------------------------------------------- loop control --


async def test_step_budget_stops_a_runaway_agent(runtime) -> None:
    """An agent that never calls finish is stopped by the step ceiling."""

    class NeverFinishes(ScriptedProvider):
        async def complete(self, messages, *, tools=None, **_kwargs) -> LLMResponse:
            if not tools:
                return LLMResponse(text='{"understanding": "x", "steps": []}')
            return LLMResponse(
                tool_calls=[ToolCall(name="scroll", arguments={"direction": "down"})]
            )

    runtime.llm = NeverFinishes([])
    graph = build_graph(runtime)

    final = await graph.ainvoke(
        initial_state(TASK_ID, "scroll forever", "test"), config=_config(runtime)
    )

    assert final["status"] == TaskStatus.FAILED
    assert final["step"] <= runtime.settings.agent_max_steps
    assert "steps" in final["result"]


async def test_ask_user_ends_the_run_awaiting_input(runtime) -> None:
    """`ask_user` is terminal, and surfaces the question."""
    runtime.llm = ScriptedProvider(
        [ToolCall(name="ask_user", arguments={"question": "Which size do you want?"})]
    )
    graph = build_graph(runtime)

    final = await graph.ainvoke(
        initial_state(TASK_ID, "order a shirt", "test"), config=_config(runtime)
    )

    assert final["status"] == TaskStatus.AWAITING_INPUT
    assert final["question"] == "Which size do you want?"
