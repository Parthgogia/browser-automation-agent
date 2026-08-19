"""Browser control: perception (DOM indexing) and action (Playwright verbs)."""

from app.browser.actions import ActionResult
from app.browser.observation import IndexedElement, Observation, ScrollInfo
from app.browser.registry import SessionRegistry, get_session_registry
from app.browser.session import BrowserSession

__all__ = [
    "ActionResult",
    "BrowserSession",
    "IndexedElement",
    "Observation",
    "ScrollInfo",
    "SessionRegistry",
    "get_session_registry",
]
