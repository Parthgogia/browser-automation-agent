"""Browser session: one persistent Chromium profile driven by the agent.

A `BrowserSession` owns a Playwright *persistent context*, which is what makes
"log into GitHub and create a repository" work across runs -- cookies,
localStorage and saved logins live in a real profile directory on disk rather
than evaporating with the process.

The session is deliberately not part of the LangGraph state. Graph state has to
be serialisable for checkpointing, and a live browser is the opposite of that.
Instead the graph carries a task id and looks the session up in the registry;
see `docs/ARCHITECTURE.md` for why that split matters.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from playwright.async_api import (
    BrowserContext,
    ElementHandle,
    Frame,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError

from app.browser.observation import IndexedElement, Observation, ScrollInfo
from app.config import Settings

logger = logging.getLogger(__name__)

#: The injected indexer, read once at import time.
_INDEX_SCRIPT = (Path(__file__).parent / "js" / "build_dom_index.js").read_text(encoding="utf-8")

#: Screenshots are JPEG rather than PNG: roughly 5x smaller for the same
#: legibility, which matters when they are base64-encoded into a model prompt.
_SCREENSHOT_QUALITY = 72


class BrowserSession:
    """Owns a browser context and exposes observation + element resolution.

    Construct via :meth:`start`; always release with :meth:`close`.
    """

    def __init__(self, settings: Settings, profile: str) -> None:
        self.settings = settings
        self.profile = profile
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._last_observation: Observation | None = None
        #: Files the browser downloaded during this session.
        self.downloads: list[str] = []
        #: Strong references to in-flight download tasks; the event loop
        #: only holds weak references, so bare tasks can be collected.
        self._pending_downloads: set[asyncio.Task[None]] = set()
        #: When false, tabs opened programmatically do not steal focus.
        self._auto_focus_new_pages = True
        self._closed = False

    # ------------------------------------------------------------ lifecycle --

    @classmethod
    async def start(cls, settings: Settings, profile: str | None = None) -> BrowserSession:
        """Launch Chromium against a persistent profile directory."""
        session = cls(settings, profile or settings.browser_default_profile)
        await session._launch()
        return session

    async def _launch(self) -> None:
        profile_dir = self.settings.browser_profile_dir / self.profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = await async_playwright().start()

        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=self.settings.browser_headless,
            viewport={
                "width": self.settings.browser_viewport_width,
                "height": self.settings.browser_viewport_height,
            },
            accept_downloads=True,
            downloads_path=str(self.settings.download_dir),
            args=[
                # Cosmetic: suppresses the automation infobar, which otherwise
                # steals vertical space from every screenshot.
                "--disable-blink-features=AutomationControlled",
                "--no-default-browser-check",
                "--no-first-run",
            ],
        )
        self._context.set_default_timeout(self.settings.browser_action_timeout_ms)

        # Injected before any page script, on every navigation and every frame,
        # so the indexer is always available without a per-call round trip.
        await self._context.add_init_script(_INDEX_SCRIPT)

        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()
        self._context.on("page", self._on_new_page)
        self._page.on("download", self._on_download)

        await self._open_start_page()

        logger.info(
            "Browser session started (profile=%s, headless=%s)",
            self.profile,
            self.settings.browser_headless,
        )

    async def _open_start_page(self) -> None:
        """Move off ``about:blank`` before anyone -- model or human -- looks.

        A blank tab is the worst possible first observation. It has no
        elements, so the DOM index comes back empty; an empty index is exactly
        what the vision fallback reads as "this page is opaque", so the first
        and most expensive decision of every run gets a screenshot of nothing
        attached to it. It is also what the human watches in the preview pane
        for as long as that decision takes. One navigation removes both.

        Best effort throughout: a machine that is offline, or behind a proxy
        that blocks the start page, must still end up with a working browser.
        """
        url = self.settings.browser_start_url.strip()
        page = self._page
        if not url or page is None or not page.url.startswith("about:"):
            return
        try:
            await page.goto(
                url,
                wait_until="domcontentloaded",
                # Deliberately tighter than the action timeout: this is pure
                # launch latency the user is waiting through, and giving up
                # early costs nothing but a blank first page.
                timeout=min(self.settings.browser_action_timeout_ms, 10_000),
            )
        except PlaywrightError as exc:
            logger.info(
                "Start page %s did not load (%s); beginning on about:blank",
                url,
                exc,
            )

    def _on_new_page(self, page: Page) -> None:
        """Follow ``target=_blank`` navigations: the newest tab becomes active.

        This mirrors what a human sees -- clicking a link that opens a tab puts
        that tab in front -- and saves the model from having to notice the new
        tab and switch to it explicitly.
        """
        page.on("download", self._on_download)
        if self._auto_focus_new_pages:
            self._page = page

    @asynccontextmanager
    async def background_page(self) -> AsyncIterator[Page]:
        """A scratch tab that never becomes the active one.

        Used by tools that need the browser for something other than the task
        at hand -- running a web search, for instance -- without moving the
        agent off the page it is working on. The tab is always closed again.
        """
        self._auto_focus_new_pages = False
        try:
            page = await self.context.new_page()
        finally:
            self._auto_focus_new_pages = True
        try:
            yield page
        finally:
            try:
                await page.close()
            except PlaywrightError:  # pragma: no cover
                pass

    def _on_download(self, download: Any) -> None:
        """Persist a download to the configured directory.

        Playwright fires this synchronously but saving is async, so the work
        is handed to a task that we keep a strong reference to.
        """

        async def save() -> None:
            try:
                target = self.settings.download_dir / download.suggested_filename
                await download.save_as(str(target))
                self.downloads.append(str(target))
                logger.info("Downloaded %s", target)
            except PlaywrightError as exc:  # pragma: no cover - network dependent
                logger.warning("Download failed: %s", exc)

        task = asyncio.create_task(save())
        self._pending_downloads.add(task)
        task.add_done_callback(self._pending_downloads.discard)

    async def close(self) -> None:
        """Tear the browser down. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._context is not None:
            try:
                await self._context.close()
            except PlaywrightError:  # pragma: no cover
                pass
        if self._playwright is not None:
            await self._playwright.stop()
        logger.info("Browser session closed (profile=%s)", self.profile)

    # --------------------------------------------------------------- access --

    @property
    def page(self) -> Page:
        """The currently active tab, recovering if it was closed underneath us."""
        if self._page is None or self._page.is_closed():
            live = self.pages
            if not live:
                raise RuntimeError("Browser has no open pages")
            self._page = live[-1]
        return self._page

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("Browser session is not started")
        return self._context

    @property
    def pages(self) -> list[Page]:
        """Every open tab, closed ones filtered out."""
        return [p for p in self.context.pages if not p.is_closed()]

    def switch_to_page(self, index: int) -> Page:
        """Make tab `index` active. Raises `IndexError` if it does not exist."""
        self._page = self.pages[index]
        return self._page

    @property
    def last_observation(self) -> Observation | None:
        return self._last_observation

    # ---------------------------------------------------------- page waits --

    async def wait_until_settled(self, timeout_ms: int | None = None) -> None:
        """Best-effort wait for the page to stop changing.

        Deliberately forgiving: many real sites hold long-polling connections
        open forever, so waiting for `networkidle` would time out on them every
        single time. We wait for DOM readiness, give the network a short chance
        to go quiet, and proceed regardless.
        """
        timeout = timeout_ms or self.settings.browser_action_timeout_ms
        page = self.page
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout)
        except PlaywrightError:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=min(2500, timeout))
        except PlaywrightError:
            # Expected on sites with persistent sockets; not an error.
            pass

    # --------------------------------------------------------- observation --

    async def observe(self, *, screenshot: bool = True) -> Observation:
        """Build a fresh :class:`Observation` of the active tab.

        Runs the injected indexer once per frame and merges the results into a
        single, globally-numbered element list.
        """
        await self.wait_until_settled()
        page = self.page

        elements: list[IndexedElement] = []
        content_blocks: list[str] = []
        scroll = ScrollInfo()
        truncated = False
        url, title = page.url, ""

        for frame_index, frame in enumerate(page.frames):
            result = await self._index_frame(frame, start_index=len(elements))
            if result is None:
                continue

            for raw in result.get("elements", []):
                box = raw.get("box", {})
                elements.append(
                    IndexedElement(
                        index=raw["index"],
                        tag=raw.get("tag", ""),
                        label=raw.get("label", ""),
                        attrs=raw.get("attrs", ""),
                        in_viewport=bool(raw.get("inViewport")),
                        box=(
                            box.get("x", 0),
                            box.get("y", 0),
                            box.get("width", 0),
                            box.get("height", 0),
                        ),
                        frame_index=frame_index,
                        local_index=raw["localIndex"],
                    )
                )

            block = result.get("content", "")
            if block:
                # Label subframe content so the model understands why the page
                # text suddenly changes subject.
                if frame_index > 0:
                    block = f"--- iframe: {frame.url[:100]} ---\n{block}"
                content_blocks.append(block)

            truncated = truncated or bool(result.get("truncated"))

            if frame_index == 0:
                raw_scroll = result.get("scroll", {})
                scroll = ScrollInfo(
                    x=raw_scroll.get("x", 0),
                    y=raw_scroll.get("y", 0),
                    pixels_above=raw_scroll.get("pixelsAbove", 0),
                    pixels_below=raw_scroll.get("pixelsBelow", 0),
                    viewport_height=raw_scroll.get("viewportHeight", 0),
                    document_height=raw_scroll.get("documentHeight", 0),
                )
                meta = result.get("meta", {})
                url = meta.get("url", url)
                title = meta.get("title", "")

        observation = Observation(
            url=url,
            title=title,
            content="\n".join(content_blocks),
            elements=elements,
            scroll=scroll,
            tabs=[{"url": p.url, "title": ""} for p in self.pages],
            truncated=truncated,
        )

        if screenshot:
            observation.screenshot_bytes = await self.capture_screenshot(
                annotate_elements=elements
            )

        self._last_observation = observation
        return observation

    async def _index_frame(self, frame: Frame, *, start_index: int) -> dict[str, Any] | None:
        """Run the indexer inside one frame, tolerating frames we cannot reach.

        Frames detach mid-navigation constantly on real sites, and some (a
        cross-origin ad slot, a sandboxed payment iframe) refuse execution
        outright. Any such frame is skipped rather than failing the whole
        observation.
        """
        try:
            # The init script normally has this defined already; the explicit
            # evaluate covers frames created before the script was registered.
            has_indexer = await frame.evaluate(
                "() => typeof window.__agentBuildIndex === 'function'"
            )
            if not has_indexer:
                await frame.evaluate(_INDEX_SCRIPT)
            return await frame.evaluate(
                "opts => window.__agentBuildIndex(opts)", {"startIndex": start_index}
            )
        except PlaywrightError as exc:
            logger.debug("Skipping frame %s: %s", frame.url[:80], exc)
            return None

    # --------------------------------------------------------- screenshots --

    async def capture_screenshot(
        self, *, annotate_elements: list[IndexedElement] | None = None
    ) -> bytes | None:
        """Screenshot the viewport, optionally with numbered element overlays.

        The overlay (a "set of marks") is what makes the image *actionable*:
        the model can read a number off the picture and pass it straight to a
        tool, instead of trying to describe what it wants to click.
        """
        page = self.page
        annotated_frames: list[Frame] = []
        try:
            if annotate_elements:
                by_frame: dict[int, list[dict[str, int]]] = {}
                for element in annotate_elements:
                    if element.in_viewport:
                        by_frame.setdefault(element.frame_index, []).append(
                            {"localIndex": element.local_index, "index": element.index}
                        )
                for frame_index, items in by_frame.items():
                    try:
                        frame = page.frames[frame_index]
                        await frame.evaluate("items => window.__agentHighlight(items)", items)
                        annotated_frames.append(frame)
                    except (PlaywrightError, IndexError):
                        continue

            return await page.screenshot(
                type="jpeg", quality=_SCREENSHOT_QUALITY, timeout=10_000
            )
        except PlaywrightError as exc:
            logger.debug("Screenshot failed: %s", exc)
            return None
        finally:
            for frame in annotated_frames:
                try:
                    await frame.evaluate("() => window.__agentClearHighlights()")
                except PlaywrightError:
                    pass

    # -------------------------------------------------- element resolution --

    async def resolve(self, index: int) -> tuple[ElementHandle, IndexedElement]:
        """Turn a model-supplied index into a live Playwright element handle.

        Raises `LookupError` with an actionable message when the index is not
        in the most recent observation. That message goes straight back to the
        model as a tool error, so it has to explain how to recover.
        """
        observation = self._last_observation
        if observation is None:
            raise LookupError("No observation has been taken yet; the page must be read first.")

        element = observation.element_by_index(index)
        if element is None:
            valid = [el.index for el in observation.elements]
            hint = f"valid indices are {valid[0]}-{valid[-1]}" if valid else "the page has none"
            raise LookupError(
                f"Element index {index} does not exist on the current page ({hint}). "
                "The page may have changed; read it again before acting."
            )

        try:
            frame = self.page.frames[element.frame_index]
            handle = await frame.evaluate_handle(
                "i => window.__agentElements[i]", element.local_index
            )
        except (PlaywrightError, IndexError) as exc:
            raise LookupError(
                f"Element {index} could not be resolved because the page changed ({exc}). "
                "Read the page again to get fresh indices."
            ) from exc

        node = handle.as_element()
        if node is None:
            raise LookupError(
                f"Element {index} is no longer attached to the page. "
                "Read the page again to get fresh indices."
            )
        return node, element
