"""Control-flow tools: the two ways a run can end.

Both are `terminal`, meaning the agent loop stops as soon as one is called.
Giving the model an explicit way to stop is important -- without it, a model
that has finished the task keeps acting, and a model that is stuck keeps
flailing. `finish` and `ask_user` are how it says "done" and "I need you".
"""

from __future__ import annotations

from app.tools.base import ToolContext, ToolResult, boolean, schema, string
from app.tools.registry import ToolRegistry


def register_tools(registry: ToolRegistry) -> None:
    """Add the control tools to `registry`."""

    @registry.tool(
        "finish",
        "End the task and report back to the user. Call this as soon as the "
        "goal is met, and also when you are certain it cannot be met -- in that "
        "case set success=false and explain what blocked you. The summary is "
        "the only thing the user reads, so answer their actual question in it, "
        "with the concrete details (prices, names, links) you found.",
        schema(
            summary=string(
                "The answer for the user. Include concrete findings, not a "
                "description of what you did."
            ),
            success=boolean("True if the goal was achieved", optional=True),
        ),
        mutates_page=False,
        terminal=True,
    )
    async def finish(
        context: ToolContext, summary: str, success: bool = True
    ) -> ToolResult:
        return ToolResult(ok=success, message=summary, data={"summary": summary}, terminal=True)

    @registry.tool(
        "ask_user",
        "Stop and ask the user a question. Use this when the task is genuinely "
        "ambiguous, when a choice is the user's to make (which of three flights, "
        "which size), or when you hit a login, CAPTCHA or payment step that only "
        "they can complete. Do not use it for things you could find out by "
        "looking at the page.",
        schema(
            question=string("The question to put to the user, in plain language"),
            context_note=string(
                "One line on what you have done so far, so the question makes sense",
                optional=True,
            ),
        ),
        mutates_page=False,
        terminal=True,
    )
    async def ask_user(
        context: ToolContext, question: str, context_note: str = ""
    ) -> ToolResult:
        return ToolResult(
            ok=True,
            message=question,
            data={"question": question, "context": context_note},
            terminal=True,
        )
