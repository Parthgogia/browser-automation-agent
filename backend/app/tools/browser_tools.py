"""Browser tools: the agent's hands.

Every tool here is a thin adapter between the model-facing schema and a verb in
`app.browser.actions`. The descriptions matter as much as the code -- they are
the only documentation the model ever reads, so they state not just what a tool
does but *when to reach for it* and what its arguments mean in terms of the
numbered page rendering the model was just shown.
"""

from __future__ import annotations

from app.browser import actions
from app.tools.base import (
    ToolContext,
    ToolResult,
    boolean,
    integer,
    number,
    schema,
    string,
)
from app.tools.registry import ToolRegistry


def _from_action(result: actions.ActionResult) -> ToolResult:
    """Lift a browser `ActionResult` into a `ToolResult`."""
    return ToolResult(ok=result.ok, message=result.message, data=result.data)


def register_tools(registry: ToolRegistry) -> None:
    """Add every browser tool to `registry`."""

    # ------------------------------------------------------------ navigate --

    @registry.tool(
        "navigate",
        "Load a web page in the current tab. Use a full URL when you know it, "
        "otherwise search first. Navigating discards the current page, so read "
        "anything you still need before you call this.",
        schema(url=string("Absolute URL to open, e.g. https://www.amazon.in")),
    )
    async def navigate(context: ToolContext, url: str) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.navigate(session, url))

    @registry.tool(
        "go_back",
        "Return to the previous page in this tab's history, exactly like the "
        "browser back button.",
        schema(),
    )
    async def go_back(context: ToolContext) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.go_back(session))

    @registry.tool(
        "reload",
        "Reload the current page. Useful when content failed to load or a "
        "transient error is showing.",
        schema(),
    )
    async def reload(context: ToolContext) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.reload(session))

    # --------------------------------------------------------- interaction --

    @registry.tool(
        "click",
        "Click an interactive element. Pass the number shown in square "
        "brackets in the page rendering, e.g. click(index=12) for "
        "[12]<button>Add to cart</button>. Indices change every time the page "
        "changes, so always use numbers from the most recent page rendering.",
        schema(index=integer("The [n] number of the element to click")),
    )
    async def click(context: ToolContext, index: int) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.click(session, index))

    @registry.tool(
        "type_text",
        "Type into a text field, replacing whatever it currently contains. Set "
        "press_enter=true to submit a search box in the same step. Never use "
        "this for passwords, card numbers or one-time codes -- ask the user to "
        "enter those.",
        schema(
            index=integer("The [n] number of the input or textarea"),
            text=string("The text to type"),
            press_enter=boolean(
                "Press Enter after typing, to submit a search or form",
                optional=True,
            ),
        ),
    )
    async def type_text(
        context: ToolContext, index: int, text: str, press_enter: bool = False
    ) -> ToolResult:
        session = await context.get_browser()
        return _from_action(
            await actions.type_text(session, index, text, press_enter=press_enter)
        )

    @registry.tool(
        "press_key",
        "Press a single key, sent to whatever currently has keyboard focus. Use "
        "Escape to dismiss a modal, Enter to submit, Tab to move between fields.",
        schema(
            key=string(
                "Key name",
                enum=sorted(actions.ALLOWED_KEYS),
            )
        ),
    )
    async def press_key(context: ToolContext, key: str) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.press_key(session, key))

    @registry.tool(
        "hover",
        "Hover the mouse over an element without clicking. Use this to open "
        "menus that expand on hover.",
        schema(index=integer("The [n] number of the element to hover")),
    )
    async def hover(context: ToolContext, index: int) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.hover(session, index))

    @registry.tool(
        "select_option",
        "Choose a value in a dropdown (<select>). Pass the visible option text.",
        schema(
            index=integer("The [n] number of the select element"),
            value=string("The option's visible text, e.g. 'Large' or 'India'"),
        ),
    )
    async def select_option(context: ToolContext, index: int, value: str) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.select_option(session, index, value))

    @registry.tool(
        "upload_file",
        "Attach a file from the local machine to a file-upload field.",
        schema(
            index=integer("The [n] number of the file input"),
            path=string("Absolute path of the file to upload"),
        ),
    )
    async def upload_file(context: ToolContext, index: int, path: str) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.upload_file(session, index, path))

    # -------------------------------------------------------------- scroll --

    @registry.tool(
        "scroll",
        "Scroll the page. The page rendering tells you how many pixels remain "
        "below; if it says 0 there is nothing further down and scrolling again "
        "will not help.",
        schema(
            direction=string(
                "Which way to scroll",
                enum=["down", "up", "top", "bottom"],
                optional=True,
            ),
            amount=integer("Pixels to scroll; defaults to about one screen", optional=True),
        ),
    )
    async def scroll(
        context: ToolContext, direction: str = "down", amount: int | None = None
    ) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.scroll(session, direction, amount))  # type: ignore[arg-type]

    @registry.tool(
        "scroll_to_text",
        "Jump straight to a piece of text on the page. Much faster than "
        "scrolling repeatedly down a long page looking for something.",
        schema(text=string("Text to scroll to; a distinctive fragment is enough")),
    )
    async def scroll_to_text(context: ToolContext, text: str) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.scroll_to_text(session, text))

    # ---------------------------------------------------------------- tabs --

    @registry.tool(
        "new_tab",
        "Open a new browser tab and switch to it. Use this to compare two sites "
        "without losing your place on the first.",
        schema(url=string("URL to open in the new tab", optional=True)),
    )
    async def new_tab(context: ToolContext, url: str | None = None) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.new_tab(session, url))

    @registry.tool(
        "switch_tab",
        "Switch to another open tab by its number, as listed in 'Open tabs'.",
        schema(index=integer("Zero-based tab number")),
    )
    async def switch_tab(context: ToolContext, index: int) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.switch_tab(session, index))

    @registry.tool(
        "close_tab",
        "Close a tab you no longer need.",
        schema(index=integer("Zero-based tab number")),
    )
    async def close_tab(context: ToolContext, index: int) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.close_tab(session, index))

    # ------------------------------------------------------------- reading --

    @registry.tool(
        "extract_text",
        "Read the full visible text of the current page. The page rendering you "
        "are shown each step is optimised for clicking and drops long prose, so "
        "use this whenever the task is to actually read something: an article, "
        "a specification table, a set of reviews, a price breakdown.",
        schema(),
        mutates_page=False,
    )
    async def extract_text(context: ToolContext) -> ToolResult:
        session = await context.get_browser()
        result = await actions.extract_text(session)
        if not result.ok:
            return _from_action(result)
        # The text itself is the point, so it goes in the message rather than
        # only in `data` -- `data` is not shown to the model.
        return ToolResult.success(
            f"{result.message}\n\n{result.data.get('text', '')}",
            **result.data,
        )

    @registry.tool(
        "wait",
        "Pause before looking at the page again. Use this when a page is still "
        "loading or an animation is in progress. Do not use it to 'try again' "
        "after a failure -- change your approach instead.",
        schema(seconds=number("How long to wait, up to 15 seconds")),
        mutates_page=False,
    )
    async def wait(context: ToolContext, seconds: float) -> ToolResult:
        session = await context.get_browser()
        return _from_action(await actions.wait(session, seconds))
