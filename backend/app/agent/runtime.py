"""Runtime services shared by every graph node.

LangGraph nodes are plain functions of state, which leaves no obvious place to
put the things a node needs but the state must not carry: the LLM client, the
tool registry, the browser registry, the database. Threading them through a
`config` dictionary works but loses all type information.

Instead, `build_graph` closes over one `AgentRuntime`. Nodes are built by
factory functions that capture it, so each node stays a pure function of state
while still reaching real services. Tests construct a runtime with fakes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.browser.observation import Observation
from app.browser.registry import SessionRegistry
from app.browser.session import BrowserSession
from app.config import Settings
from app.db.repository import TaskRepository
from app.events import AgentEvent, EventBus, EventType
from app.llm.base import LLMProvider
from app.memory.store import MemoryStore
from app.safety.policy import SafetyPolicy
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: In `auto` vision mode, attach a screenshot when the text rendering has fewer
#: than this many interactive elements -- a strong hint that the page is
#: canvas-drawn, still loading, or otherwise opaque to DOM inspection.
_SPARSE_ELEMENT_THRESHOLD = 4

#: ...or when the rendering is shorter than this many characters.
_SPARSE_CONTENT_THRESHOLD = 300


@dataclass(slots=True)
class AgentRuntime:
    """Everything the graph needs that is not part of graph state."""

    settings: Settings
    llm: LLMProvider
    tools: ToolRegistry
    sessions: SessionRegistry
    safety: SafetyPolicy
    events: EventBus
    repository: TaskRepository
    memory: MemoryStore | None = None

    # ------------------------------------------------------------- events --

    def emit(
        self,
        task_id: str,
        event_type: EventType,
        message: str = "",
        *,
        step: int | None = None,
        **data: Any,
    ) -> AgentEvent:
        """Publish an event to the live stream and persist it.

        Publishing is synchronous and non-blocking; persistence is fire-and-
        forget so that a slow database cannot slow the agent down.
        """
        event = AgentEvent(
            task_id=task_id, type=event_type, message=message, step=step, data=data
        )
        self.events.publish(event)
        self._persist(event)
        return event

    def _persist(self, event: AgentEvent) -> None:
        try:
            task = asyncio.create_task(self.repository.record_event(event))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        except RuntimeError:
            # No running loop (a synchronous test); persistence is optional.
            pass

    #: Strong references to fire-and-forget persistence tasks.
    _background: set[Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._background = set()

    # ------------------------------------------------------------ browser --

    async def browser_for(self, task_id: str, profile: str | None = None) -> BrowserSession:
        """The browser session for a task, launched on first use."""
        return await self.sessions.acquire(task_id, self.settings, profile)

    def tool_context(self, task_id: str, goal: str, profile: str) -> ToolContext:
        """Build the context handed to every tool call in this run."""

        async def get_browser() -> BrowserSession:
            return await self.browser_for(task_id, profile)

        return ToolContext(
            task_id=task_id,
            settings=self.settings,
            get_browser=get_browser,
            llm=self.llm,
            memory=self.memory,
            emit=self.events.publish,
            goal=goal,
        )

    # ------------------------------------------------------------- vision --

    def should_attach_screenshot(
        self, observation: Observation, *, consecutive_failures: int
    ) -> bool:
        """Decide whether this step needs the vision channel.

        Sending an image every step is the most expensive thing this agent can
        do -- on a free Gemini tier it is also the fastest way to hit a rate
        limit -- so `auto` spends it only where text is demonstrably failing:

        * the page yielded almost no interactive elements or almost no text,
          which is what a canvas app, a captcha, or a still-loading page looks
          like from the DOM's point of view; or
        * the previous action failed, so the model's text-based model of the
          page has already been shown to be wrong.
        """
        mode = self.settings.vision_mode
        if mode == "never":
            return False
        if mode == "always":
            return True

        if consecutive_failures >= 1:
            return True
        if len(observation.elements) < _SPARSE_ELEMENT_THRESHOLD:
            return True
        return len(observation.content) < _SPARSE_CONTENT_THRESHOLD

    # -------------------------------------------------------- screenshots --

    def save_screenshot(self, task_id: str, step: int, image: bytes) -> str:
        """Write a screenshot to disk and return the URL path that serves it."""
        directory = self.settings.screenshot_dir / task_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{step:03d}.jpg"
        path.write_bytes(image)
        return f"/screenshots/{task_id}/{path.name}"
