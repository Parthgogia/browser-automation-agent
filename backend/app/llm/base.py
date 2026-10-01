"""Provider-agnostic LLM interface.

The agent talks to this interface and never to a vendor SDK. That buys three
things worth the small amount of indirection:

* **Swappable models.** Gemini today, a local model or Claude tomorrow, with
  one config line and no changes to the graph.
* **A keyless test path.** `MockProvider` implements the same interface, so the
  entire system -- graph, tools, browser, UI -- can be exercised end to end with
  no API key and no network.
* **One place for tool-schema translation.** Every provider wants function
  declarations in a slightly different shape; the conversion lives beside the
  provider rather than leaking into the tool definitions.

Deliberately *not* used here: LangChain's chat-model wrappers. LangGraph does
not require them, and going direct to the provider SDK keeps multimodal parts
and function-call plumbing explicit rather than hidden behind two abstraction
layers.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from app.llm.capabilities import CapabilityMatch, ModelCapabilities, RequestRequirements

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """A model's request to invoke one tool."""

    name: str
    arguments: dict[str, Any]
    #: Correlates the call with its result. Some providers supply one; for
    #: those that do not we generate it.
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(slots=True)
class Message:
    """One turn of the conversation with the model."""

    role: Role
    content: str = ""
    #: Set on assistant turns where the model asked for tools.
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Set on `tool` turns: which call this is the result of.
    tool_call_id: str | None = None
    #: Set on `tool` turns: the tool's name (Gemini keys results by name).
    name: str | None = None
    #: JPEG bytes attached to a user turn. This is the vision channel.
    images: list[bytes] = field(default_factory=list, repr=False)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, images: list[bytes] | None = None) -> Message:
        return cls(role="user", content=content, images=images or [])

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls(role="tool", content=content, tool_call_id=call.id, name=call.name)


@dataclass(slots=True)
class ToolSpec:
    """A tool as advertised to the model.

    `parameters` is a JSON Schema object. Keep it to the subset every provider
    understands -- ``type``, ``properties``, ``description``, ``enum``,
    ``required``, ``items`` -- since Gemini accepts OpenAPI 3.0 schemas rather
    than full JSON Schema, and rejects constructs such as ``anyOf`` or ``$ref``.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class LLMResponse:
    """What the model said, and what it wants to do next."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Token counts when the provider reports them; useful for cost telemetry.
    usage: dict[str, int] = field(default_factory=dict)
    #: Provider-reported stop reason, for diagnosing truncated responses.
    finish_reason: str = ""


class LLMError(RuntimeError):
    """Raised when a provider fails in a way the agent cannot work around."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        fallback: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.fallback = fallback


class LLMProvider(ABC):
    """The contract every model backend implements."""

    #: Human-readable name, used in logs and events.
    name: str = "llm"

    async def resolve_capabilities(
        self, requirements: RequestRequirements, *, fast: bool = False
    ) -> CapabilityMatch:
        """Return known capabilities for the model selected for this request."""
        return CapabilityMatch(capabilities=ModelCapabilities())

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        force_tool_use: bool = False,
    ) -> LLMResponse:
        """Generate the next assistant turn.

        Args:
            messages: Conversation so far. At most one `system` message, first.
            tools: Tools the model may call this turn.
            temperature: Overrides the configured default.
            model: Overrides the configured default model.
            force_tool_use: Require the model to answer with a tool call rather
                than prose. Used by the act loop, where free text is never the
                right answer.
        """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` for semantic memory. Returns one vector per input."""

    @property
    @abstractmethod
    def embedding_dimensions(self) -> int:
        """Vector width produced by :meth:`embed`.

        The memory table's column type is fixed at creation time, so this has
        to be known before any embedding is generated.
        """
