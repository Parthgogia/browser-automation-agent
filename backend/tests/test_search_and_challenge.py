"""Search backends and anti-bot detection.

The bug these guard against is a quiet one: a search that is being *blocked*
reports itself as a search that found *nothing*, the model dutifully rephrases
the query, and the run grinds through its whole step budget against a wall it
was never told about. So the two outcomes are asserted separately and by
message, not just by `ok`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

import app.tools.search_tools as search_tools
from app.browser.challenge import challenge_in_url, detect_challenge
from app.tools.base import ToolContext
from app.tools.registry import build_registry

# --------------------------------------------------------------- unwrapping --


@pytest.mark.parametrize(
    ("wrapped", "expected"),
    [
        (
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync%2Dio%2F&rut=abc",
            "https://realpython.com/async-io/",
        ),
        (
            # Bing base64url-encodes the destination behind an `a1` prefix.
            "https://www.bing.com/ck/a?!&&p=123&u=a1aHR0cHM6Ly9kb2NzLnB5dGhvbi5vcmcvMy8",
            "https://docs.python.org/3/",
        ),
        # Anything that is not a wrapper must survive untouched.
        ("https://www.pcmag.com/picks/best-laptops", "https://www.pcmag.com/picks/best-laptops"),
        ("https://duckduckgo.com/?q=laptops", "https://duckduckgo.com/?q=laptops"),
    ],
)
def test_redirect_wrappers_are_unwrapped(wrapped: str, expected: str) -> None:
    assert search_tools._unwrap(wrapped) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://duckduckgo.com/l/?uddg=",  # empty target
        "https://www.bing.com/ck/a?u=a1!!!not-base64!!!",
        "https://www.bing.com/ck/a?u=nothing",
        "",
    ],
)
def test_a_malformed_wrapper_is_returned_as_is(url: str) -> None:
    """Never raise on a link the engine formatted unexpectedly."""
    assert search_tools._unwrap(url) == url


# ---------------------------------------------------------------- challenges --


def test_a_block_page_is_recognised_from_its_url() -> None:
    assert challenge_in_url("https://www.google.com/sorry/index?continue=x") is not None
    assert challenge_in_url("https://www.amazon.in/dp/B0ABC") is None


class _FakePage:
    """Enough of a Playwright page for the challenge check and the extractor."""

    def __init__(self, url: str, text: str = "", results: list | None = None) -> None:
        self.url = url
        self._text = text
        self._results = results or []
        self.ready = bool(results)

    async def goto(self, url: str, **_: object) -> None:
        self.url = url

    async def evaluate(self, script: str, *args: object):
        # The challenge probe is the only script that reads innerText.
        if "innerText" in script:
            return self._text[: args[0] if args else None]
        return self._results

    async def wait_for_selector(self, _selector: str, **_kwargs: object) -> None:
        if not self.ready:
            raise TimeoutError("no results rendered")

    async def wait_for_timeout(self, _ms: int) -> None:
        return None


async def test_a_short_challenge_page_is_detected() -> None:
    page = _FakePage(
        "https://html.duckduckgo.com/html/?q=laptops",
        "DuckDuckGo\nUnfortunately, bots use DuckDuckGo too.\nSelect all squares "
        "containing a duck.\nSubmit",
    )
    assert await detect_challenge(page) == "a DuckDuckGo bot challenge"  # type: ignore[arg-type]


async def test_a_long_article_mentioning_captchas_is_not_a_challenge() -> None:
    """The brevity guard: a page *about* bot walls is not a bot wall."""
    article = "A history of the captcha. " * 200 + "\nunusual traffic\n"
    page = _FakePage("https://example.com/blog/captchas", article)

    assert len(article) > 2000
    assert await detect_challenge(page) is None  # type: ignore[arg-type]


# ------------------------------------------------------- the search backend --


class _FakeSession:
    """A browser whose background tab always shows `page`."""

    def __init__(self, page: _FakePage) -> None:
        self._page = page

    @asynccontextmanager
    async def background_page(self):
        yield self._page


def _context(settings, page: _FakePage) -> ToolContext:
    session = _FakeSession(page)

    async def get_browser():
        return session

    return ToolContext(
        task_id="task-1",
        settings=settings,
        get_browser=get_browser,
        llm=None,
        memory=None,
        emit=lambda _event: None,
        goal="find a laptop",
    )


async def test_being_blocked_is_reported_as_being_blocked(settings, monkeypatch) -> None:
    """The message must steer the model away from rephrasing."""
    page = _FakePage(
        "https://html.duckduckgo.com/html/?q=laptops",
        "Unfortunately, bots use DuckDuckGo too. Please complete the challenge.",
    )
    monkeypatch.setattr(search_tools, "_ENGINES", search_tools._ENGINES[:1])

    result = await build_registry().execute(
        "web_search", {"query": "cheapest laptop"}, _context(settings, page)
    )

    assert not result.ok
    assert result.data.get("blocked") is True
    assert "Rephrasing the query will not help" in result.message
    assert "returned nothing" not in result.message


async def test_an_empty_result_set_still_suggests_rewording(settings, monkeypatch) -> None:
    """The opposite case must keep its original, opposite advice."""
    page = _FakePage("https://html.duckduckgo.com/html/?q=zzqq", "Ordinary results page")
    monkeypatch.setattr(search_tools, "_ENGINES", search_tools._ENGINES[:1])

    result = await build_registry().execute(
        "web_search", {"query": "zzqq"}, _context(settings, page)
    )

    assert not result.ok
    assert result.data.get("blocked") is None
    assert "Try different wording" in result.message


async def test_a_walled_engine_falls_through_to_the_next_one(settings) -> None:
    """One blocked engine must not cost the search."""
    walled = _FakePage("https://html.duckduckgo.com/html/?q=laptops", "unusual traffic")
    working = _FakePage(
        "https://www.bing.com/search?q=laptops",
        "Ordinary results page",
        results=[
            {
                "title": "Best budget laptops",
                "url": "https://www.bing.com/ck/a?u=a1aHR0cHM6Ly9wY21hZy5jb20v",
                "snippet": "Our picks.",
            }
        ],
    )

    pages = iter([walled, working])

    class _TwoTabSession:
        @asynccontextmanager
        async def background_page(self):
            yield next(pages)

    async def get_browser():
        return _TwoTabSession()

    context = ToolContext(
        task_id="task-1",
        settings=settings,
        get_browser=get_browser,
        llm=None,
        memory=None,
        emit=lambda _event: None,
        goal="find a laptop",
    )

    result = await build_registry().execute(
        "web_search", {"query": "cheapest laptop"}, context
    )

    assert result.ok
    # And the surviving engine's wrapper was unwrapped on the way out.
    assert "https://pcmag.com/" in result.message
