"""Structured representation of what the agent currently sees.

An :class:`Observation` is the sole input the reasoning model gets about the
page. It is produced after every action and is deliberately *derived*, never
raw: raw HTML is far too large and far too noisy to reason over, and raw
screenshots alone give the model no way to name what it wants to click.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class IndexedElement:
    """One interactive element the agent may act on, addressed by `index`.

    `index` is unique across the whole page (all frames merged). `frame_index`
    and `local_index` are the coordinates needed to resolve the element back to
    a live Playwright handle: the element lives in `page.frames[frame_index]`
    at position `local_index` of that frame's ``window.__agentElements``.
    """

    index: int
    tag: str
    label: str
    attrs: str
    in_viewport: bool
    #: Frame-local viewport coordinates: x, y, width, height.
    box: tuple[int, int, int, int]
    frame_index: int
    local_index: int

    def render(self) -> str:
        """The single line shown to the model for this element."""
        attrs = f" {self.attrs}" if self.attrs else ""
        return f"[{self.index}]<{self.tag}{attrs}>{self.label}</{self.tag}>"


@dataclass(slots=True)
class ScrollInfo:
    """Where we are in the document, and how much is left to see."""

    x: int = 0
    y: int = 0
    pixels_above: int = 0
    pixels_below: int = 0
    viewport_height: int = 0
    document_height: int = 0

    def render(self) -> str:
        above = f"{self.pixels_above}px above" if self.pixels_above else "top of page"
        below = f"{self.pixels_below}px below" if self.pixels_below else "end of page"
        return f"Scroll position: {above}, {below}."


@dataclass(slots=True)
class Observation:
    """A complete snapshot of the browser at one instant."""

    url: str
    title: str
    #: Text rendering of the page in document order, interactive elements
    #: inlined as `[n]<tag ...>label</tag>`.
    content: str
    elements: list[IndexedElement] = field(default_factory=list)
    scroll: ScrollInfo = field(default_factory=ScrollInfo)
    #: Titles/urls of every open tab, so the model can switch between them.
    tabs: list[dict[str, str]] = field(default_factory=list)
    #: Relative path of the annotated screenshot, when one was captured.
    screenshot_path: str | None = None
    #: Raw JPEG bytes of that screenshot, present only when the vision fallback
    #: decided the model needs to actually look at the page.
    screenshot_bytes: bytes | None = field(default=None, repr=False)
    #: True when the page rendering hit the size cap and was cut short.
    truncated: bool = False
    #: Populated when observation itself failed (navigation in flight, crash).
    error: str | None = None

    def element_by_index(self, index: int) -> IndexedElement | None:
        """Look up an element by the number the model used."""
        # Linear scan: element counts are in the hundreds at most, and keeping
        # a list preserves document order for rendering.
        return next((el for el in self.elements if el.index == index), None)

    def render(self, *, max_chars: int = 20_000) -> str:
        """Format this observation as the text block handed to the model."""
        parts = [
            f"URL: {self.url}",
            f"Title: {self.title}",
        ]
        if len(self.tabs) > 1:
            tabs = "; ".join(
                f"[{i}] {t.get('title') or t.get('url', '')}" for i, t in enumerate(self.tabs)
            )
            parts.append(f"Open tabs: {tabs}")
        parts.append(self.scroll.render())

        if self.error:
            parts.append(f"Observation error: {self.error}")

        content = self.content
        if len(content) > max_chars:
            # Keep the head: document order means the top of the page is
            # usually where navigation and search controls live.
            content = content[:max_chars] + "\n... [page rendering truncated]"
        elif self.truncated:
            content += "\n... [page rendering truncated]"

        parts.append("")
        parts.append("Page content (interactive elements are numbered):")
        parts.append(content or "[page appears empty]")
        return "\n".join(parts)
