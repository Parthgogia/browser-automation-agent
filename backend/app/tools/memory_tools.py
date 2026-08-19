"""Long-term memory tools.

Memory is what turns a stateless task runner into something that gets better at
serving one particular person. "I prefer Dell laptops", "my usual delivery
address is X", "last time the cheapest option was on Croma" -- facts like these
are worth carrying between runs, and the agent decides for itself what is worth
keeping.

Recall is *not* automatic on every step, because injecting memories the task
does not need is a reliable way to derail it. Instead the planner is given
relevant memories once, at the start, and the model can pull more with `recall`
when it realises it needs them.
"""

from __future__ import annotations

import logging

from app.tools.base import ToolContext, ToolResult, integer, schema, string
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_tools(registry: ToolRegistry) -> None:
    """Add the memory tools to `registry`."""

    @registry.tool(
        "remember",
        "Save a durable fact about the user or a site for future tasks -- a "
        "stated preference, a delivery address, which site turned out cheapest. "
        "Only save things that will still be true next week. Never save "
        "passwords, card numbers or anything else secret.",
        schema(
            fact=string("The fact to remember, written as a complete sentence"),
            category=string(
                "What kind of fact this is",
                enum=["preference", "profile", "site_knowledge", "task_outcome"],
                optional=True,
            ),
        ),
        mutates_page=False,
    )
    async def remember(
        context: ToolContext, fact: str, category: str = "preference"
    ) -> ToolResult:
        if context.memory is None:
            return ToolResult.failure(
                "Long-term memory is unavailable in this run, so nothing was saved."
            )
        try:
            await context.memory.add(
                content=fact, category=category, task_id=context.task_id
            )
        except Exception as exc:  # noqa: BLE001 - memory must never break a run
            logger.warning("Failed to store memory: %s", exc)
            return ToolResult.failure(f"Could not save that: {exc}")
        return ToolResult.success(f"Remembered: {fact}")

    @registry.tool(
        "recall",
        "Search everything remembered about this user from previous tasks. Use "
        "it when the task depends on a preference or detail you were not told, "
        "such as a shipping address or a brand the user favours.",
        schema(
            query=string("What you are trying to remember about"),
            limit=integer("How many memories to return (default 5)", optional=True),
        ),
        mutates_page=False,
    )
    async def recall(context: ToolContext, query: str, limit: int = 5) -> ToolResult:
        if context.memory is None:
            return ToolResult.failure("Long-term memory is unavailable in this run.")
        try:
            hits = await context.memory.search(query, limit=max(1, min(int(limit), 10)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory search failed: %s", exc)
            return ToolResult.failure(f"Could not search memory: {exc}")

        if not hits:
            return ToolResult.success(f"Nothing remembered about {query!r}.")

        lines = [f"Remembered about {query!r}:"]
        lines += [f"- ({hit.category}) {hit.content}" for hit in hits]
        return ToolResult.success("\n".join(lines), memories=[hit.content for hit in hits])
