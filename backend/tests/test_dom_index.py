"""The DOM indexer, tested against a real Chromium.

The indexer is the agent's eyes, and almost every hard bug in a browser agent
is a perception bug: an element that was listed but is not clickable, one that
exists but was never listed, an index that points at the wrong thing. Those are
not reproducible against a mock DOM, so these tests drive real Chromium over
`page.set_content` fixtures -- no network, no live site, but a real layout
engine doing real hit-testing.

Skipped automatically when no browser binary is installed.
"""

from __future__ import annotations

import re

import pytest

from app.browser.session import _INDEX_SCRIPT

playwright_api = pytest.importorskip("playwright.async_api")


@pytest.fixture
async def page():
    """A headless page with the indexer injected on every navigation.

    Function-scoped deliberately. A module-scoped async fixture would be bound
    to a different event loop than the tests under pytest-asyncio's auto mode,
    which deadlocks rather than failing; a fresh browser per test costs about a
    second and keeps the tests independent.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Chromium is not installed: {exc}")

        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        await context.add_init_script(_INDEX_SCRIPT)
        target = await context.new_page()
        yield target
        await browser.close()


async def build(page, html: str) -> dict:
    """Render `html` and return the index it produces."""
    await page.set_content(html)
    return await page.evaluate("() => window.__agentBuildIndex({startIndex: 0})")


def indices_for(result: dict, tag: str) -> list[int]:
    return [el["index"] for el in result["elements"] if el["tag"] == tag]


def labels(result: dict) -> list[str]:
    return [el["label"] for el in result["elements"]]


# ---------------------------------------------------------------- the basics --


async def test_indexes_ordinary_controls(page) -> None:
    result = await build(
        page,
        """
        <button>Add to cart</button>
        <a href="/cart">View cart</a>
        <input type="text" placeholder="Search">
        <select><option>Small</option><option>Large</option></select>
        <textarea></textarea>
        """,
    )
    tags = {el["tag"] for el in result["elements"]}
    assert {"button", "a", "input", "select", "textarea"} <= tags
    assert "Add to cart" in labels(result)


async def test_indices_are_contiguous_and_start_where_told(page) -> None:
    """Numbering must be dense and offsettable, so frames can be merged."""
    await page.set_content("<button>A</button><button>B</button><button>C</button>")
    result = await page.evaluate("() => window.__agentBuildIndex({startIndex: 10})")
    assert [el["index"] for el in result["elements"]] == [10, 11, 12]


async def test_rendering_inlines_indices_with_page_text(page) -> None:
    """The text view is what the model reads; it must interleave both."""
    result = await build(
        page,
        "<h1>Gaming laptops</h1><p>From 119990</p><button>Buy now</button>",
    )
    content = result["content"]
    assert "Gaming laptops" in content
    assert "119990" in content
    assert re.search(r"\[\d+\]<button[^>]*>Buy now</button>", content)


# ----------------------------------------------------------------- exclusion --


async def test_invisible_elements_are_not_indexed(page) -> None:
    """Anything a human cannot see must not be offered to the model."""
    result = await build(
        page,
        """
        <button style="display:none">Hidden by display</button>
        <button style="visibility:hidden">Hidden by visibility</button>
        <button style="opacity:0">Transparent</button>
        <button hidden>Hidden attribute</button>
        <button aria-hidden="true">Aria hidden</button>
        <input type="hidden" value="csrf">
        <button>Real button</button>
        """,
    )
    assert labels(result) == ["Real button"]


async def test_disabled_controls_are_not_indexed(page) -> None:
    result = await build(
        page,
        """
        <button disabled>Out of stock</button>
        <button aria-disabled="true">Unavailable</button>
        <button>In stock</button>
        """,
    )
    assert labels(result) == ["In stock"]


async def test_covered_elements_are_not_indexed(page) -> None:
    """The hit-test is what stops the agent clicking through a modal.

    A button underneath a full-screen overlay looks perfectly clickable in the
    DOM. Clicking it is what produces the classic "element intercepts pointer
    events" failure, so it must never be listed in the first place.
    """
    result = await build(
        page,
        """
        <button style="position:absolute;top:100px;left:100px">Underneath</button>
        <div style="position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:99">
          <button>Accept cookies</button>
        </div>
        """,
    )
    assert "Accept cookies" in labels(result)
    assert "Underneath" not in labels(result)


# ------------------------------------------------------------ harder markup --


async def test_descends_into_shadow_dom(page) -> None:
    """Web components are invisible to querySelectorAll but real to users."""
    await page.set_content("<div id='host'></div>")
    await page.evaluate(
        """
        () => {
          const root = document.getElementById('host').attachShadow({mode: 'open'});
          root.innerHTML = '<button>Inside shadow</button>';
        }
        """
    )
    result = await page.evaluate("() => window.__agentBuildIndex({startIndex: 0})")
    assert "Inside shadow" in labels(result)


async def test_recognises_aria_and_div_buttons(page) -> None:
    """Real sites build controls out of divs; the agent must still see them."""
    result = await build(
        page,
        """
        <div role="button" tabindex="0">Role button</div>
        <div style="cursor:pointer" onclick="void 0">Cursor button</div>
        <div>Just text</div>
        """,
    )
    found = labels(result)
    assert "Role button" in found
    assert "Cursor button" in found
    assert "Just text" not in found


async def test_prefers_aria_label_over_inner_text(page) -> None:
    result = await build(page, '<button aria-label="Close dialog">&times;</button>')
    assert labels(result) == ["Close dialog"]


@pytest.mark.parametrize(
    "markup",
    [
        '<input type="password" value="hunter2">',
        '<input name="cvv" value="hunter2">',
        '<input id="otp-code" value="hunter2">',
        '<input autocomplete="current-password" value="hunter2">',
    ],
)
async def test_credential_values_are_never_exposed(page, markup: str) -> None:
    """No path may echo a secret field's contents into the model's context.

    There are three of them -- the attribute dump, the `current=` hint, and the
    label fallback chain -- and it only takes one to leak a password.
    """
    result = await build(page, markup)
    assert "hunter2" not in result["content"]
    assert all("hunter2" not in str(el) for el in result["elements"])


# ---------------------------------------------------------------- scrolling --


async def test_scroll_metrics_describe_what_is_left(page) -> None:
    """`pixelsBelow` is how the model knows scrolling would achieve anything."""
    await page.set_content("<div style='height:5000px'>tall</div><button>End</button>")
    top = await page.evaluate("() => window.__agentBuildIndex({startIndex: 0})")
    assert top["scroll"]["pixelsAbove"] == 0
    assert top["scroll"]["pixelsBelow"] > 1000

    await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    bottom = await page.evaluate("() => window.__agentBuildIndex({startIndex: 0})")
    assert bottom["scroll"]["pixelsBelow"] == 0
    assert bottom["scroll"]["pixelsAbove"] > 1000


# -------------------------------------------------------------- annotations --


async def test_highlight_overlay_is_added_and_removed_cleanly(page) -> None:
    """Set-of-marks boxes must not leak into the next observation."""
    await build(page, "<button>One</button><button>Two</button>")

    await page.evaluate(
        "() => window.__agentHighlight([{localIndex: 0, index: 0},"
        " {localIndex: 1, index: 1}])"
    )
    assert await page.evaluate(
        "() => !!document.getElementById('__agent_highlight_layer__')"
    )

    await page.evaluate("() => window.__agentClearHighlights()")
    assert not await page.evaluate(
        "() => !!document.getElementById('__agent_highlight_layer__')"
    )

    # And the overlay must never appear as an indexed element itself.
    result = await page.evaluate("() => window.__agentBuildIndex({startIndex: 0})")
    assert labels(result) == ["One", "Two"]
