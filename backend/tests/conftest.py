"""Shared fixtures.

The important one is `runtime`: a fully wired `AgentRuntime` with the mock LLM,
no database, and a *fake browser* pre-registered. That combination lets the
whole graph be exercised -- planning, deciding, safety, approval, execution,
reflection, completion -- in milliseconds and with no network, no Postgres and
no Chromium. Anything that genuinely needs a real browser is marked
`integration` and skipped by default.
"""

from __future__ import annotations

import pytest

from app.agent.runtime import AgentRuntime
from app.browser.observation import IndexedElement, Observation, ScrollInfo
from app.browser.registry import SessionRegistry
from app.config import Settings
from app.db.repository import TaskRepository
from app.db.session import Database
from app.events import EventBus
from app.llm.mock import MockProvider
from app.safety.policy import SafetyPolicy
from app.tools.registry import build_registry


class FakeBrowserSession:
    """A browser that never launches, for testing everything around it.

    Implements just enough of `BrowserSession` for the graph and tools: an
    observation to reason over, a URL that changes on navigation, and a record
    of what was asked of it.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.profile = "test"
        self.url = "about:blank"
        self.calls: list[tuple[str, tuple, dict]] = []
        self._observation = _make_observation(self.url)

    # -- the surface the agent actually uses --------------------------------

    async def observe(self, *, screenshot: bool = True) -> Observation:
        # Only rebuild when the page actually changed, so that a test which
        # customises the current observation (relabelling a field as a
        # password, say) does not have it silently reset on the next step.
        if self._observation.url != self.url:
            self._observation = _make_observation(self.url)
        return self._observation

    @property
    def last_observation(self) -> Observation:
        return self._observation

    async def resolve(self, index: int):
        element = self._observation.element_by_index(index)
        if element is None:
            raise LookupError(f"Element index {index} does not exist on the current page.")
        return _FakeHandle(self), element

    async def wait_until_settled(self, timeout_ms: int | None = None) -> None:
        return None

    async def capture_screenshot(self, **_: object) -> bytes:
        return b"\xff\xd8\xff"  # JPEG magic bytes; enough to be non-empty

    async def close(self) -> None:
        return None

    @property
    def pages(self) -> list[object]:
        return [self]

    @property
    def page(self) -> _FakePage:
        return _FakePage(self)

    def switch_to_page(self, index: int) -> _FakePage:
        if index != 0:
            raise IndexError(index)
        return self.page


class _FakePage:
    """Minimal stand-in for a Playwright `Page`."""

    def __init__(self, session: FakeBrowserSession) -> None:
        self._session = session

    @property
    def url(self) -> str:
        return self._session.url

    async def goto(self, url: str, **_: object):
        self._session.calls.append(("goto", (url,), {}))
        self._session.url = url
        return _FakeResponse(200)

    async def evaluate(self, _script: str, *args: object):
        return "" if not args else None

    async def wait_for_load_state(self, *_: object, **__: object) -> None:
        return None


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status


class _FakeHandle:
    """Element handle that records interactions instead of performing them."""

    def __init__(self, session: FakeBrowserSession) -> None:
        self._session = session

    async def scroll_into_view_if_needed(self, **_: object) -> None:
        return None

    async def click(self, **_: object) -> None:
        self._session.calls.append(("click", (), {}))

    async def fill(self, value: str) -> None:
        self._session.calls.append(("fill", (value,), {}))

    async def type(self, text: str, **_: object) -> None:
        self._session.calls.append(("type", (text,), {}))

    async def press(self, key: str) -> None:
        self._session.calls.append(("press", (key,), {}))


def _make_observation(url: str) -> Observation:
    """A small but realistic page: a search box and two buttons."""
    elements = [
        IndexedElement(
            index=0, tag="input", label="Search", attrs='placeholder="Search" type="text"',
            in_viewport=True, box=(10, 10, 300, 30), frame_index=0, local_index=0,
        ),
        IndexedElement(
            index=1, tag="button", label="Add to cart", attrs="",
            in_viewport=True, box=(10, 60, 120, 30), frame_index=0, local_index=1,
        ),
        IndexedElement(
            index=2, tag="button", label="Place your order", attrs="",
            in_viewport=True, box=(10, 100, 140, 30), frame_index=0, local_index=2,
        ),
    ]
    return Observation(
        url=url,
        title="Test page",
        content="\n".join(element.render() for element in elements)
        + "\nSome ordinary page text about a laptop costing 119990.",
        elements=elements,
        scroll=ScrollInfo(pixels_below=0, viewport_height=900, document_height=900),
        tabs=[{"url": url, "title": "Test page"}],
        screenshot_bytes=b"\xff\xd8\xff",
    )


# ------------------------------------------------------------------ fixtures --


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at a temporary directory, with tight budgets."""
    return Settings(
        llm_provider="mock",
        database_url="postgresql+asyncpg://unused",
        data_dir=tmp_path,
        browser_profile_dir=tmp_path / "profiles",
        agent_max_steps=6,
        agent_max_failures=4,
        agent_max_reflections=2,
        require_approval=True,
        vision_mode="never",
    )


@pytest.fixture
def fake_session(settings: Settings) -> FakeBrowserSession:
    return FakeBrowserSession(settings)


@pytest.fixture
def runtime(settings: Settings, fake_session: FakeBrowserSession) -> AgentRuntime:
    """A runtime whose browser is the fake, and whose database is absent."""
    settings.ensure_directories()

    sessions = SessionRegistry()
    database = Database(settings)  # never connected, so `available` stays False

    built = AgentRuntime(
        settings=settings,
        llm=MockProvider(settings),
        tools=build_registry(),
        sessions=sessions,
        safety=SafetyPolicy(settings),
        events=EventBus(),
        repository=TaskRepository(database),
        memory=None,
    )

    # Pre-register the fake so `acquire` never launches Chromium.
    sessions._sessions["task-1"] = fake_session  # type: ignore[assignment]
    return built
