"""The verbs the agent can perform in the browser.

Each function here is a thin, *forgiving* wrapper over Playwright. Two design
rules apply throughout:

1. **Never raise for an ordinary failure.** A click that misses, an element
   that vanished, a page that timed out -- these are all normal events in the
   life of a browser agent, and the model is the thing best placed to recover
   from them. So they come back as an :class:`ActionResult` with ``ok=False``
   and a message written *for the model to read*.

2. **Say what happened, not what was attempted.** "Clicked [12] Add to cart;
   page navigated to /cart" lets the model verify its own progress. "OK" does
   not.

Actual exceptions are reserved for programmer error and for a browser that has
genuinely died.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from playwright.async_api import Error as PlaywrightError

from app.browser.session import BrowserSession

logger = logging.getLogger(__name__)

#: Keys the model is allowed to press. Restricting the set keeps a confused
#: model from inventing key names that Playwright will reject.
ALLOWED_KEYS = {
    "Enter", "Tab", "Escape", "Backspace", "Delete", "ArrowUp", "ArrowDown",
    "ArrowLeft", "ArrowRight", "PageUp", "PageDown", "Home", "End", "Space",
    "Control+a", "Control+c", "Control+v",
}


@dataclass(slots=True)
class ActionResult:
    """Outcome of one browser action, as reported back to the model."""

    ok: bool
    #: Model-facing description of what actually happened.
    message: str
    #: Structured payload for actions that return data (extracted text, etc.).
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, message: str, **data: Any) -> ActionResult:
        return cls(ok=True, message=message, data=data)

    @classmethod
    def failure(cls, message: str, **data: Any) -> ActionResult:
        return cls(ok=False, message=message, data=data)


def _describe(label: str, index: int) -> str:
    """Short human/model-readable name for an element."""
    trimmed = (label or "").strip()
    return f"[{index}] {trimmed[:60]}" if trimmed else f"[{index}]"


# --------------------------------------------------------------- navigation --


async def navigate(session: BrowserSession, url: str) -> ActionResult:
    """Load `url` in the active tab."""
    if not url.startswith(("http://", "https://", "file://", "about:")):
        url = f"https://{url}"
    try:
        response = await session.page.goto(url, wait_until="domcontentloaded")
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not load {url}: {_clean(exc)}")

    status = response.status if response else None
    if status and status >= 400:
        return ActionResult.failure(
            f"Loaded {session.page.url} but the server returned HTTP {status}.",
            status=status,
        )
    return ActionResult.success(f"Navigated to {session.page.url}", status=status)


async def go_back(session: BrowserSession) -> ActionResult:
    try:
        await session.page.go_back(wait_until="domcontentloaded")
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not go back: {_clean(exc)}")
    return ActionResult.success(f"Went back to {session.page.url}")


async def go_forward(session: BrowserSession) -> ActionResult:
    try:
        await session.page.go_forward(wait_until="domcontentloaded")
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not go forward: {_clean(exc)}")
    return ActionResult.success(f"Went forward to {session.page.url}")


async def reload(session: BrowserSession) -> ActionResult:
    try:
        await session.page.reload(wait_until="domcontentloaded")
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not reload: {_clean(exc)}")
    return ActionResult.success(f"Reloaded {session.page.url}")


# ------------------------------------------------------------- interaction --


async def click(session: BrowserSession, index: int) -> ActionResult:
    """Click the element at `index`.

    Reports navigation and new tabs explicitly, because those are the two
    outcomes most likely to invalidate the model's mental model of the page.
    """
    try:
        handle, element = await session.resolve(index)
    except LookupError as exc:
        return ActionResult.failure(str(exc))

    url_before = session.page.url
    tabs_before = len(session.pages)

    try:
        await handle.scroll_into_view_if_needed(timeout=3_000)
    except PlaywrightError:
        # Not fatal: the element may be in a fixed-position overlay that is
        # already on screen, in which case scrolling is a no-op anyway.
        pass

    try:
        await handle.click(timeout=session.settings.browser_action_timeout_ms)
    except PlaywrightError as exc:
        # A cookie banner or sticky header intercepting the pointer is the most
        # common cause. A synthetic DOM click bypasses hit-testing entirely, so
        # try that once before giving up.
        if "intercepts pointer events" in str(exc) or "not visible" in str(exc):
            try:
                await handle.evaluate("el => el.click()")
            except PlaywrightError as inner:
                return ActionResult.failure(
                    f"Could not click {_describe(element.label, index)}: {_clean(inner)}"
                )
        else:
            return ActionResult.failure(
                f"Could not click {_describe(element.label, index)}: {_clean(exc)}"
            )

    await session.wait_until_settled()

    notes = []
    if len(session.pages) > tabs_before:
        session.switch_to_page(len(session.pages) - 1)
        notes.append(f"opened a new tab, now on {session.page.url}")
    elif session.page.url != url_before:
        notes.append(f"page navigated to {session.page.url}")

    suffix = f"; {', '.join(notes)}" if notes else ""
    return ActionResult.success(f"Clicked {_describe(element.label, index)}{suffix}")


async def type_text(
    session: BrowserSession,
    index: int,
    text: str,
    *,
    clear: bool = True,
    press_enter: bool = False,
) -> ActionResult:
    """Type `text` into the input at `index`."""
    try:
        handle, element = await session.resolve(index)
    except LookupError as exc:
        return ActionResult.failure(str(exc))

    try:
        await handle.scroll_into_view_if_needed(timeout=3_000)
    except PlaywrightError:
        pass

    try:
        if clear:
            # fill("") is the only reliable way to clear React-controlled
            # inputs; select-all + Backspace often leaves stale internal state.
            await handle.fill("")
        # type() emits real key events, which sites with autocomplete and
        # keystroke validation depend on. fill() would skip them.
        await handle.type(text, delay=25)
    except PlaywrightError as exc:
        return ActionResult.failure(
            f"Could not type into {_describe(element.label, index)}: {_clean(exc)}"
        )

    url_before = session.page.url
    if press_enter:
        try:
            await handle.press("Enter")
        except PlaywrightError as exc:
            return ActionResult.failure(f"Typed the text but Enter failed: {_clean(exc)}")
        await session.wait_until_settled()

    # Never echo secrets back into the transcript or the event stream.
    shown = "***" if _looks_secret(element.attrs, element.tag) else text
    suffix = " and pressed Enter" if press_enter else ""
    if press_enter and session.page.url != url_before:
        suffix += f"; page navigated to {session.page.url}"
    return ActionResult.success(
        f"Typed {shown!r} into {_describe(element.label, index)}{suffix}"
    )


async def press_key(session: BrowserSession, key: str) -> ActionResult:
    """Send a keyboard key to whatever currently has focus."""
    if key not in ALLOWED_KEYS:
        return ActionResult.failure(
            f"Key {key!r} is not permitted. Allowed keys: {', '.join(sorted(ALLOWED_KEYS))}"
        )
    try:
        await session.page.keyboard.press(key)
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not press {key}: {_clean(exc)}")
    await session.wait_until_settled()
    return ActionResult.success(f"Pressed {key}")


async def hover(session: BrowserSession, index: int) -> ActionResult:
    """Hover the element at `index`, e.g. to open a dropdown menu."""
    try:
        handle, element = await session.resolve(index)
    except LookupError as exc:
        return ActionResult.failure(str(exc))
    try:
        await handle.hover(timeout=session.settings.browser_action_timeout_ms)
    except PlaywrightError as exc:
        return ActionResult.failure(
            f"Could not hover {_describe(element.label, index)}: {_clean(exc)}"
        )
    # Menus animate in; a short settle beats an immediate, empty observation.
    await asyncio.sleep(0.4)
    return ActionResult.success(f"Hovered {_describe(element.label, index)}")


async def select_option(session: BrowserSession, index: int, value: str) -> ActionResult:
    """Choose `value` in the <select> at `index`.

    Matches by visible label first, then by value attribute, because the model
    sees labels in the page rendering and rarely knows the underlying values.
    """
    try:
        handle, element = await session.resolve(index)
    except LookupError as exc:
        return ActionResult.failure(str(exc))

    for strategy in ({"label": value}, {"value": value}):
        try:
            await handle.select_option(**strategy)
        except PlaywrightError:
            continue
        await session.wait_until_settled()
        return ActionResult.success(f"Selected {value!r} in {_describe(element.label, index)}")

    # Neither strategy matched, so tell the model what it could have picked.
    try:
        options = await handle.eval_on_selector_all(
            "option", "els => els.map(e => e.label || e.textContent).slice(0, 30)"
        )
    except PlaywrightError:
        options = []

    message = f"No option matching {value!r} in {_describe(element.label, index)}."
    if options:
        message += f" Available options: {options}"
    return ActionResult.failure(message)


async def upload_file(session: BrowserSession, index: int, path: str) -> ActionResult:
    """Attach a local file to the file input at `index`."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        return ActionResult.failure(f"No such file: {file_path}")
    try:
        handle, element = await session.resolve(index)
        await handle.set_input_files(str(file_path))
    except LookupError as exc:
        return ActionResult.failure(str(exc))
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not upload to [{index}]: {_clean(exc)}")
    return ActionResult.success(f"Uploaded {file_path.name} to {_describe(element.label, index)}")


# ----------------------------------------------------------------- scroll --


async def scroll(
    session: BrowserSession,
    direction: Literal["down", "up", "top", "bottom"] = "down",
    amount: int | None = None,
) -> ActionResult:
    """Scroll the page. `amount` is in pixels; it defaults to one viewport."""
    page = session.page
    viewport = session.settings.browser_viewport_height
    delta = amount if amount is not None else int(viewport * 0.8)

    try:
        if direction == "top":
            await page.evaluate("() => window.scrollTo({top: 0})")
        elif direction == "bottom":
            await page.evaluate("() => window.scrollTo({top: document.body.scrollHeight})")
        else:
            signed = delta if direction == "down" else -delta
            await page.evaluate("d => window.scrollBy(0, d)", signed)
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not scroll: {_clean(exc)}")

    # Lazy-loaded content needs a beat to render before the next observation.
    await asyncio.sleep(0.4)
    return ActionResult.success(f"Scrolled {direction}")


async def scroll_to_text(session: BrowserSession, text: str) -> ActionResult:
    """Scroll until `text` is visible. Useful on very long pages."""
    try:
        locator = session.page.get_by_text(text, exact=False).first
        await locator.scroll_into_view_if_needed(timeout=5_000)
    except PlaywrightError:
        return ActionResult.failure(
            f"Could not find the text {text!r} anywhere on this page."
        )
    await asyncio.sleep(0.3)
    return ActionResult.success(f"Scrolled to text {text!r}")


# ------------------------------------------------------------------- tabs --


async def new_tab(session: BrowserSession, url: str | None = None) -> ActionResult:
    try:
        page = await session.context.new_page()
        session.switch_to_page(len(session.pages) - 1)
        if url:
            return await navigate(session, url)
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not open a tab: {_clean(exc)}")
    return ActionResult.success(f"Opened a new tab ({page.url})")


async def switch_tab(session: BrowserSession, index: int) -> ActionResult:
    try:
        page = session.switch_to_page(index)
    except IndexError:
        return ActionResult.failure(
            f"No tab {index}; there are {len(session.pages)} open tabs (0-indexed)."
        )
    return ActionResult.success(f"Switched to tab {index} ({page.url})")


async def close_tab(session: BrowserSession, index: int) -> ActionResult:
    pages = session.pages
    if index >= len(pages):
        return ActionResult.failure(f"No tab {index}; there are {len(pages)} open tabs.")
    if len(pages) == 1:
        return ActionResult.failure("Refusing to close the only open tab.")
    try:
        await pages[index].close()
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not close tab {index}: {_clean(exc)}")
    session.switch_to_page(len(session.pages) - 1)
    return ActionResult.success(f"Closed tab {index}; now on {session.page.url}")


# ---------------------------------------------------------------- reading --


async def extract_text(session: BrowserSession, max_chars: int = 12_000) -> ActionResult:
    """Return the page's visible text.

    The numbered rendering in an observation is optimised for *acting*; it
    drops long prose to stay small. When the task is to read something -- an
    article, a spec table, a set of reviews -- this returns the text instead.
    """
    try:
        text = await session.page.evaluate(
            "() => (document.body && document.body.innerText) || ''"
        )
    except PlaywrightError as exc:
        return ActionResult.failure(f"Could not read the page: {_clean(exc)}")

    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return ActionResult.success(
        f"Extracted {len(text)} characters from {session.page.url}"
        + (" (truncated)" if truncated else ""),
        text=text,
        truncated=truncated,
    )


async def wait(session: BrowserSession, seconds: float) -> ActionResult:
    """Pause. Capped so a confused model cannot stall the run indefinitely."""
    seconds = max(0.0, min(float(seconds), 15.0))
    await asyncio.sleep(seconds)
    return ActionResult.success(f"Waited {seconds:g}s")


# ---------------------------------------------------------------- helpers --


def _clean(exc: BaseException) -> str:
    """Trim Playwright's multi-line error dumps down to the useful first line.

    Raw Playwright errors include a full call log that can run to dozens of
    lines. Feeding that to the model wastes context and buries the cause.
    """
    first = str(exc).strip().splitlines()
    return first[0][:300] if first else exc.__class__.__name__


def _looks_secret(attrs: str, tag: str) -> bool:
    """Heuristic: would echoing this field's value leak a credential?"""
    haystack = f"{attrs} {tag}".lower()
    return any(
        marker in haystack
        for marker in ('type="password"', "password", "otp", "cvv", "secret", "pin")
    )
