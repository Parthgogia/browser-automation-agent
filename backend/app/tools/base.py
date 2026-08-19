"""Tool plumbing: what a tool is, and what it is given when it runs.

A "tool" here is the unit of everything the agent can *do*. The model never
touches Playwright, the database or the network directly -- it emits a function
call naming a tool, and this layer executes it. That indirection is what makes
the agent auditable: every effect the agent has on the world passes through one
dispatch point, where it can be logged, rate-limited and gated on approval.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.config import Settings

if TYPE_CHECKING:  # imports only needed for type checking, avoiding cycles
    from app.browser.session import BrowserSession
    from app.events import AgentEvent
    from app.llm.base import LLMProvider
    from app.memory.store import MemoryStore


@dataclass(slots=True)
class ToolResult:
    """What a tool reports back.

    `message` is written for the model to read: it is appended to the
    conversation verbatim, so it should describe the *outcome*, and on failure
    it should say enough for the model to choose a different approach.
    """

    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    #: True for tools that end the run (`finish`, `ask_user`).
    terminal: bool = False

    @classmethod
    def success(cls, message: str, **data: Any) -> ToolResult:
        return cls(ok=True, message=message, data=data)

    @classmethod
    def failure(cls, message: str, **data: Any) -> ToolResult:
        return cls(ok=False, message=message, data=data)

    def render(self) -> str:
        """The exact text the model sees as this tool's result."""
        prefix = "" if self.ok else "FAILED: "
        return f"{prefix}{self.message}"


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to reach.

    Passing this explicitly rather than importing singletons inside tools keeps
    them testable: a test constructs a context with fakes and calls the tool
    directly.
    """

    task_id: str
    settings: Settings
    #: Lazily created -- a task that only searches never launches a browser.
    get_browser: Callable[[], Awaitable[BrowserSession]]
    llm: LLMProvider
    memory: MemoryStore | None = None
    #: Publishes an event to the task's stream. Never blocks.
    emit: Callable[[AgentEvent], None] = lambda event: None
    #: The original user request, for tools that need the wider goal.
    goal: str = ""


#: Tool handlers take a context plus their declared arguments.
ToolHandler = Callable[..., Awaitable[ToolResult]]


@dataclass(slots=True)
class Tool:
    """One callable capability, as advertised to the model."""

    name: str
    description: str
    #: JSON Schema object describing the arguments. Kept to the subset every
    #: provider accepts -- see `app.llm.base.ToolSpec`.
    parameters: dict[str, Any]
    handler: ToolHandler
    #: Whether a successful call can change what the page looks like. Tools
    #: that cannot (a memory write, a search) skip the re-observation step,
    #: which saves a screenshot and a DOM walk per call.
    mutates_page: bool = True
    #: Terminal tools end the agent loop rather than feeding back into it.
    terminal: bool = False

    async def __call__(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        """Invoke the handler, filtering arguments down to what it accepts.

        Models hallucinate extra arguments with some regularity. Silently
        dropping unknown keys is much better behaviour than a `TypeError` that
        the model cannot understand or recover from.
        """
        signature = inspect.signature(self.handler)
        accepted = {
            name
            for name, parameter in signature.parameters.items()
            if name != "context"
            and parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }
        filtered = {key: value for key, value in arguments.items() if key in accepted}

        missing = [
            name
            for name, parameter in signature.parameters.items()
            if name != "context"
            and parameter.default is parameter.empty
            and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
            and name not in filtered
        ]
        if missing:
            return ToolResult.failure(
                f"Missing required argument(s) for {self.name}: {', '.join(missing)}."
            )

        return await self.handler(context, **filtered)


# ------------------------------------------------------- schema shorthands --
#
# Small helpers so tool definitions read as declarations rather than as nested
# dictionary literals.


def schema(**properties: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON Schema object from `name=field(...)` pairs.

    A property is required unless its spec carries ``"_optional": True``.
    """
    required = [
        name for name, spec in properties.items() if not spec.pop("_optional", False)
    ]
    return {"type": "object", "properties": properties, "required": required}


def string(description: str, *, optional: bool = False, enum: list[str] | None = None) -> dict:
    spec: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        spec["enum"] = enum
    if optional:
        spec["_optional"] = True
    return spec


def integer(description: str, *, optional: bool = False) -> dict:
    spec: dict[str, Any] = {"type": "integer", "description": description}
    if optional:
        spec["_optional"] = True
    return spec


def number(description: str, *, optional: bool = False) -> dict:
    spec: dict[str, Any] = {"type": "number", "description": description}
    if optional:
        spec["_optional"] = True
    return spec


def boolean(description: str, *, optional: bool = False) -> dict:
    spec: dict[str, Any] = {"type": "boolean", "description": description}
    if optional:
        spec["_optional"] = True
    return spec
