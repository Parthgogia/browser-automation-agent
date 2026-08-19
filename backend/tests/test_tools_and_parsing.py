"""Unit tests for the tool layer, model-output parsing and the event bus."""

from __future__ import annotations

import asyncio

import pytest

from app.agent.nodes.decide import _trim
from app.agent.parsing import parse_json_object, parse_string_list
from app.browser.observation import IndexedElement, Observation, ScrollInfo
from app.events import AgentEvent, EventBus, EventType
from app.llm.base import Message, ToolCall
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry, build_registry

# ------------------------------------------------------------- the registry --


def test_every_tool_advertises_a_valid_schema() -> None:
    """A malformed schema is rejected by the provider, not by us -- so check.

    Gemini only accepts an OpenAPI 3.0 subset, and `required` must name real
    properties or the whole request 400s at runtime.
    """
    for spec in build_registry().specs():
        assert spec.description, f"{spec.name} has no description"
        assert spec.parameters["type"] == "object"
        properties = spec.parameters["properties"]
        for name in spec.parameters.get("required", []):
            assert name in properties, f"{spec.name} requires undeclared {name!r}"
        # `_optional` is our own marker and must never reach the provider.
        assert not any("_optional" in prop for prop in properties.values())


async def test_unknown_tool_returns_a_helpful_failure(runtime) -> None:
    """The model must be told what it *could* have called."""
    context = runtime.tool_context("task-1", "goal", "test")
    result = await runtime.tools.execute("teleport", {}, context)

    assert not result.ok
    assert "no tool called 'teleport'" in result.message
    assert "navigate" in result.message


async def test_hallucinated_arguments_are_dropped(runtime) -> None:
    """Models invent arguments; that must not raise a TypeError."""
    context = runtime.tool_context("task-1", "goal", "test")
    result = await runtime.tools.execute(
        "scroll", {"direction": "down", "colour": "blue", "speed": 9}, context
    )
    assert result.ok


async def test_missing_required_arguments_are_reported(runtime) -> None:
    context = runtime.tool_context("task-1", "goal", "test")
    result = await runtime.tools.execute("click", {}, context)

    assert not result.ok
    assert "index" in result.message


async def test_a_raising_tool_becomes_a_failed_result() -> None:
    """An exception inside a tool must not end the run."""
    registry = ToolRegistry()

    @registry.tool("explode", "Always fails", {"type": "object", "properties": {}})
    async def explode(context) -> ToolResult:
        raise ValueError("boom")

    result = await registry.execute("explode", {}, context=None)  # type: ignore[arg-type]
    assert not result.ok
    assert "boom" in result.message


def test_terminal_tools_are_marked() -> None:
    registry = build_registry()
    assert registry.get("finish").terminal
    assert registry.get("ask_user").terminal
    assert not registry.get("click").terminal


# ------------------------------------------------------------------ parsing --


@pytest.mark.parametrize(
    "text",
    [
        '{"steps": ["a", "b"]}',
        '```json\n{"steps": ["a", "b"]}\n```',
        '```\n{"steps": ["a", "b"]}\n```',
        'Here is the plan: {"steps": ["a", "b"]}. Let me know!',
        '  \n {"steps": ["a", "b"]}  \n ',
    ],
)
def test_json_survives_the_usual_model_wrapping(text: str) -> None:
    parsed = parse_json_object(text)
    assert parsed == {"steps": ["a", "b"]}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", "{not json}"])
def test_unparseable_output_returns_none(text: str) -> None:
    assert parse_json_object(text) is None


def test_step_lists_are_cleaned_of_model_formatting() -> None:
    assert parse_string_list(["1. Search", "- Open it", "* Read it", "  "]) == [
        "Search",
        "Open it",
        "Read it",
    ]


def test_a_string_of_steps_is_split() -> None:
    assert parse_string_list("Search\nOpen it\nRead it") == [
        "Search",
        "Open it",
        "Read it",
    ]


# ------------------------------------------------------- transcript trimming --


def _observation_message(step: int) -> Message:
    return Message.user(
        f"[Step {step}]\nURL: https://example.com/{step}\nTitle: Page\n"
        f"Page content (interactive elements are numbered):\n[0]<button>Go</button>"
    )


def test_trimming_keeps_actions_but_collapses_old_pages() -> None:
    """What the agent *did* is load-bearing; what pages looked like is not."""
    messages: list[Message] = [Message.system("system")]
    for step in range(6):
        messages.append(_observation_message(step))
        call = ToolCall(name="click", arguments={"index": step})
        messages.append(Message.assistant("thinking", tool_calls=[call]))
        messages.append(Message.tool_result(call, f"Clicked {step}"))

    trimmed = _trim(messages)

    # Every action and result survives.
    assert len([m for m in trimmed if m.role == "assistant"]) == 6
    assert len([m for m in trimmed if m.role == "tool"]) == 6

    # Only the two newest page renderings are kept in full.
    full = [m for m in trimmed if "interactive elements are numbered" in m.content]
    assert len(full) == 2
    stubs = [m for m in trimmed if "earlier page rendering omitted" in m.content]
    assert len(stubs) == 4
    # A stub still identifies which page it was.
    assert "URL: https://example.com/0" in stubs[0].content


def test_trimming_keeps_only_the_newest_image() -> None:
    """Images dominate the context budget, so all but the latest are dropped."""
    messages = [
        Message.user("first", images=[b"a"]),
        Message.user("second", images=[b"b"]),
        Message.user("third", images=[b"c"]),
    ]
    trimmed = _trim(messages)

    assert [bool(m.images) for m in trimmed] == [False, False, True]
    assert trimmed[-1].images == [b"c"]
    # Text is never lost, only the attachment.
    assert [m.content for m in trimmed] == ["first", "second", "third"]


def test_trimming_preserves_order() -> None:
    messages = [Message.user(str(i)) for i in range(5)]
    assert [m.content for m in _trim(messages)] == ["0", "1", "2", "3", "4"]


# ---------------------------------------------------------------- event bus --


async def test_subscribers_receive_events_and_the_backlog() -> None:
    bus = EventBus()
    bus.publish(AgentEvent(task_id="t", type=EventType.LOG, message="before"))

    async with bus.subscribe("t") as events:
        iterator = events.__aiter__()
        first = await asyncio.wait_for(iterator.__anext__(), timeout=1)
        assert first.message == "before"  # replayed

        bus.publish(AgentEvent(task_id="t", type=EventType.LOG, message="after"))
        second = await asyncio.wait_for(iterator.__anext__(), timeout=1)
        assert second.message == "after"  # live


async def test_events_are_isolated_by_task() -> None:
    bus = EventBus()
    async with bus.subscribe("a") as events:
        bus.publish(AgentEvent(task_id="b", type=EventType.LOG, message="other task"))
        bus.publish(AgentEvent(task_id="a", type=EventType.LOG, message="mine"))

        iterator = events.__aiter__()
        received = await asyncio.wait_for(iterator.__anext__(), timeout=1)
        assert received.message == "mine"


async def test_stream_survives_an_idle_period() -> None:
    """A heartbeat timeout must not kill the stream.

    Regression test. The first implementation wrapped an async generator's
    `__anext__` in `asyncio.wait_for`; on timeout that cancels the generator
    frame, terminating it, so the next read raised `StopAsyncIteration` and the
    WebSocket closed at the first quiet moment -- exactly when the agent was
    busy on a slow page and the user most wanted to watch.
    """
    bus = EventBus()

    async with bus.subscribe("t", replay=False) as events:
        # Two idle periods in a row, each reported as "nothing yet".
        assert await events.next(timeout=0.05) is None
        assert await events.next(timeout=0.05) is None

        # The subscription must still be live afterwards.
        bus.publish(AgentEvent(task_id="t", type=EventType.LOG, message="after idle"))
        received = await events.next(timeout=1)
        assert received is not None
        assert received.message == "after idle"

        # ...and keep working for the next one.
        bus.publish(AgentEvent(task_id="t", type=EventType.LOG, message="and again"))
        again = await events.next(timeout=1)
        assert again is not None and again.message == "and again"


async def test_backlog_is_drained_before_waiting() -> None:
    """Replayed history must arrive without burning a timeout each."""
    bus = EventBus()
    for n in range(3):
        bus.publish(AgentEvent(task_id="t", type=EventType.LOG, message=f"old {n}"))

    async with bus.subscribe("t") as events:
        drained = [await events.next(timeout=0.05) for _ in range(3)]
        assert [event.message for event in drained if event] == ["old 0", "old 1", "old 2"]
        # Nothing left, so the next read times out rather than blocking forever.
        assert await events.next(timeout=0.05) is None


def test_publishing_without_subscribers_is_harmless() -> None:
    """The agent publishes constantly; nobody may be listening."""
    bus = EventBus()
    bus.publish(AgentEvent(task_id="t", type=EventType.LOG, message="into the void"))
    assert len(bus.history("t")) == 1


# --------------------------------------------------------------- rendering --


def test_observation_rendering_carries_what_the_model_needs() -> None:
    observation = Observation(
        url="https://example.com",
        title="Example",
        content="[0]<button>Go</button>",
        elements=[
            IndexedElement(
                index=0, tag="button", label="Go", attrs="", in_viewport=True,
                box=(0, 0, 10, 10), frame_index=0, local_index=0,
            )
        ],
        scroll=ScrollInfo(pixels_above=0, pixels_below=1200),
    )
    rendered = observation.render()

    assert "URL: https://example.com" in rendered
    assert "Title: Example" in rendered
    assert "1200px below" in rendered  # so the model knows to scroll
    assert "[0]<button>Go</button>" in rendered


def test_oversized_renderings_are_truncated_with_a_marker() -> None:
    observation = Observation(url="u", title="t", content="x" * 50_000)
    rendered = observation.render(max_chars=1000)

    assert len(rendered) < 2000
    assert "truncated" in rendered
