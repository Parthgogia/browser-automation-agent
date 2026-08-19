"""A deterministic, keyless LLM stand-in.

This is not a toy. It exists so that the *whole* system -- graph, tool
dispatch, browser control, approval flow, persistence, WebSocket stream and UI
-- can be run and tested end to end before anyone has an API key, and so that
CI can exercise the agent without spending money or depending on the network.

It behaves like a very simple but not entirely stupid agent: it searches the
web for the goal, opens the first result, reads it, and reports back. That path
touches every moving part of the system, which is exactly what you want from a
smoke test.

The tradeoff is honest and worth stating: it cannot *reason*. Give it a task
that needs judgement and it will do the scripted thing anyway. Switch
`LLM_PROVIDER` to `gemini` for real work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct

from app.config import Settings
from app.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)

#: Width of the fake embedding vectors. Matches text-embedding-004 so that
#: switching to a real provider does not force a schema migration.
_EMBEDDING_DIMS = 768

#: Interactive-element lines in an observation: `[12]<input placeholder="...">`
_ELEMENT_RE = re.compile(r"^\[(\d+)]<(\w+)([^>]*)>(.*)</\w+>$", re.MULTILINE)

_SEARCH_HINTS = ("search", "query", "q=", 'name="q"')


class MockProvider(LLMProvider):
    """Scripted provider that drives a plausible search-and-read task."""

    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------ requests --

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        force_tool_use: bool = False,
    ) -> LLMResponse:
        goal = _extract_goal(messages)
        tool_names = {tool.name for tool in (tools or [])}

        # The planner asks for prose (a JSON plan), not tool calls.
        if not tools:
            return LLMResponse(text=_fake_plan(goal))

        return LLMResponse(**_next_action(goal, messages, tool_names))

    # ---------------------------------------------------------- embeddings --

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Hash text into a stable pseudo-random unit vector.

        Not semantically meaningful, but it *is* deterministic and distinct per
        input, which is enough for the storage layer and its tests to be
        exercised honestly.
        """
        return [_hash_embedding(text) for text in texts]

    @property
    def embedding_dimensions(self) -> int:
        return _EMBEDDING_DIMS


# ------------------------------------------------------------- the script --


def _next_action(goal: str, messages: list[Message], tool_names: set[str]) -> dict:
    """Pick the next tool call from what has already been done."""
    performed = [call.name for message in messages for call in message.tool_calls]
    observation = _latest_observation(messages)

    # 1. Nothing has happened yet: go to a search engine.
    if "navigate" not in performed and "navigate" in tool_names:
        return _call(
            "Starting by searching the web for the goal.",
            "navigate",
            {"url": "https://duckduckgo.com/"},
        )

    # 2. On the search page: type the goal into the first search-looking box.
    if "type_text" not in performed and "type_text" in tool_names:
        index = _find_search_box(observation)
        if index is not None:
            return _call(
                f"Searching for {goal!r}.",
                "type_text",
                {"index": index, "text": goal, "press_enter": True},
            )

    # 3. Results are up: read the page.
    if "extract_text" not in performed and "extract_text" in tool_names:
        return _call("Reading the results page.", "extract_text", {})

    # 4. Report what we found and stop.
    summary = _summarise(observation, goal)
    return _call("Reporting the result.", "finish", {"summary": summary, "success": True})


def _call(thought: str, name: str, arguments: dict) -> dict:
    return {"text": thought, "tool_calls": [ToolCall(name=name, arguments=arguments)]}


def _extract_goal(messages: list[Message]) -> str:
    """Recover the user's original request from the conversation."""
    for message in messages:
        if message.role == "user":
            match = re.search(r"(?:Goal|Task):\s*(.+)", message.content)
            if match:
                return match.group(1).strip().splitlines()[0]
            return message.content.strip().splitlines()[0][:200]
    return "the requested task"


def _latest_observation(messages: list[Message]) -> str:
    """The most recent page rendering in the conversation, if any."""
    for message in reversed(messages):
        if message.role == "user" and "Page content" in message.content:
            return message.content
    return ""


def _find_search_box(observation: str) -> int | None:
    """First text input that looks like a search field."""
    for match in _ELEMENT_RE.finditer(observation):
        index, tag, attrs, label = match.groups()
        if tag not in {"input", "textarea"}:
            continue
        haystack = f"{attrs} {label}".lower()
        if any(hint in haystack for hint in _SEARCH_HINTS) or 'type="text"' in haystack:
            return int(index)
    return None


def _summarise(observation: str, goal: str) -> str:
    """A short, honest summary built from whatever text we actually saw."""
    lines = [
        line.strip()
        for line in observation.splitlines()
        # Skip the numbered element lines: they are UI chrome, not content.
        if line.strip() and not line.lstrip().startswith("[")
    ]
    body = " ".join(lines[3:20])[:600] or "no readable page content"
    return (
        f"[mock provider] Searched for {goal!r} and read the results page. "
        f"What the page showed: {body}"
    )


def _fake_plan(goal: str) -> str:
    """A plausible three-step plan, in the JSON shape the planner expects."""
    return json.dumps(
        {
            "understanding": f"The user wants: {goal}",
            "steps": [
                "Search the web for the requested information",
                "Open the most relevant result",
                "Read the page and report the answer",
            ],
        }
    )


def _hash_embedding(text: str) -> list[float]:
    """Deterministic unit vector derived from a SHA-256 stream of `text`."""
    values: list[float] = []
    counter = 0
    while len(values) < _EMBEDDING_DIMS:
        digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
        # 32 bytes -> 8 floats in [-1, 1).
        for (raw,) in struct.iter_unpack(">I", digest):
            values.append(raw / 2**31 - 1.0)
        counter += 1
    values = values[:_EMBEDDING_DIMS]

    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]
