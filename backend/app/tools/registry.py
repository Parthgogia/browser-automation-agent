"""The tool registry: single source of truth for what the agent can do."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.llm.base import ToolSpec
from app.tools.base import Tool, ToolContext, ToolHandler, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Holds :class:`Tool` definitions and dispatches calls to them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ---------------------------------------------------------- definition --

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        mutates_page: bool = True,
        terminal: bool = False,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator form of :meth:`register`.

        The decorated function is returned unchanged so it stays directly
        callable from tests.
        """

        def decorator(handler: ToolHandler) -> ToolHandler:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=handler,
                    mutates_page=mutates_page,
                    terminal=terminal,
                )
            )
            return handler

        return decorator

    # ------------------------------------------------------------- lookup --

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, *, include: set[str] | None = None) -> list[ToolSpec]:
        """Tool declarations to advertise to the model.

        `include` narrows the set, which is how the graph offers a different
        vocabulary at different phases (for example, hiding `finish` until at
        least one action has been taken).
        """
        return [
            ToolSpec(
                name=tool.name, description=tool.description, parameters=tool.parameters
            )
            for name, tool in sorted(self._tools.items())
            if include is None or name in include
        ]

    # ----------------------------------------------------------- dispatch --

    async def execute(
        self, name: str, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Run the named tool, converting any escaped exception into a result.

        Nothing a tool does should be able to kill the run. An unexpected
        exception becomes a failed `ToolResult`, which lands in the transcript
        and gives the model a chance to try something else.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.failure(
                f"There is no tool called {name!r}. Available tools: {', '.join(self.names)}."
            )
        try:
            result = await tool(context, arguments)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            logger.exception("Tool %s raised", name)
            return ToolResult.failure(f"{name} raised an unexpected error: {exc}")

        result.terminal = result.terminal or tool.terminal
        return result


def build_registry() -> ToolRegistry:
    """Construct a registry with every built-in tool registered.

    The imports happen inside the function because each tool module registers
    onto the registry it is handed; keeping them local avoids import-order
    surprises and makes it obvious where the tool set comes from.
    """
    from app.tools import browser_tools, control_tools, memory_tools, search_tools

    registry = ToolRegistry()
    for module in (browser_tools, search_tools, memory_tools, control_tools):
        module.register_tools(registry)
    return registry
