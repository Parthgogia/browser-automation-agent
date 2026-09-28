"""Recognising an anti-bot wall.

A site that has decided we look like a robot almost never says so with an HTTP
error. It answers ``200 OK`` with a perfectly well-formed page that happens to
contain a puzzle instead of content, or it redirects to a "sorry" URL that also
answers ``200``. Both of those look like success to :meth:`Page.goto`, and both
look like *"the page was empty"* to a scraper.

That second reading is the dangerous one. Telling the model a search "returned
nothing" sends it off rephrasing a query that was never the problem, and every
rephrasing hits the same wall -- a loop that burns the step budget without ever
touching the real obstacle. Naming the wall costs one `evaluate` call and turns
an unbounded loop into a decision the model can actually make.

The check lives here, in one place, because both `navigate` and `web_search`
need exactly the same judgement.
"""

from __future__ import annotations

import logging

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page

logger = logging.getLogger(__name__)

#: Path fragments that only ever appear on an interstitial. Matched against the
#: *final* URL, so a redirect to one of these is caught even though the
#: navigation itself succeeded.
_URL_MARKERS: tuple[tuple[str, str], ...] = (
    ("/sorry/index", "a Google 'unusual traffic' block page"),
    ("/recaptcha/", "a reCAPTCHA challenge"),
    ("cdn-cgi/challenge", "a Cloudflare challenge"),
    ("error-pages/blocked", "a block page"),
)

#: Phrases that identify a challenge from the rendered text. Lower-cased before
#: matching. Each maps to the description handed to the model.
_TEXT_MARKERS: tuple[tuple[str, str], ...] = (
    ("unfortunately, bots use", "a DuckDuckGo bot challenge"),
    ("select all squares", "an image-selection captcha"),
    ("i'm not a robot", "a reCAPTCHA challenge"),
    ("complete the challenge", "a bot challenge"),
    ("verification required", "a verification challenge"),
    ("unusual traffic", "an 'unusual traffic' block page"),
    ("automated queries", "an automated-traffic block page"),
    ("access temporarily suspended", "a temporary block page"),
    ("checking your browser before", "a browser-integrity check"),
    ("enable javascript and cookies to continue", "a browser-integrity check"),
)

#: An interstitial is a *short* page by nature: a sentence of explanation and a
#: puzzle. Requiring brevity is what stops an article that merely discusses
#: captchas, or a forum thread about being rate-limited, from being mistaken
#: for the real thing. URL markers are exempt -- those are unambiguous.
_MAX_CHALLENGE_TEXT = 2000

#: Only the opening of the page is scanned; a challenge always leads with its
#: explanation, and this bounds what we pull across the wire.
_TEXT_SAMPLE = 4000


def challenge_in_url(url: str) -> str | None:
    """Describe the wall named by `url`, or None if it looks like a real page."""
    lowered = (url or "").lower()
    for marker, description in _URL_MARKERS:
        if marker in lowered:
            return description
    return None


async def detect_challenge(page: Page) -> str | None:
    """Describe the wall `page` is showing, or None if it looks like content.

    Never raises: a page that cannot be inspected is reported as *not* a
    challenge, because a false "you are blocked" would stop a run that was
    working. Missing a wall only costs the step it would have saved.
    """
    from_url = challenge_in_url(page.url)
    if from_url is not None:
        return from_url

    try:
        text = await page.evaluate(
            "(limit) => (document.body ? document.body.innerText : '').slice(0, limit)",
            _TEXT_SAMPLE,
        )
    except PlaywrightError as exc:
        logger.debug("Could not read page text for a challenge check: %s", exc)
        return None

    if not isinstance(text, str) or len(text) > _MAX_CHALLENGE_TEXT:
        return None

    lowered = text.lower()
    for marker, description in _TEXT_MARKERS:
        if marker in lowered:
            return description
    return None
