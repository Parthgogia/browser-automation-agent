"""Web search.

Search is how a task that starts with "find the cheapest..." gets from a
sentence to a set of URLs worth visiting. Two backends, chosen automatically:

* **Tavily**, when `TAVILY_API_KEY` is set. Purpose-built for agents: returns
  clean titles, URLs and extracted snippets in one call.
* **The browser itself**, otherwise. We drive a throwaway background tab to a
  search engine and scrape the results. This needs no key and no extra
  dependency, and because it is a real browser with a real profile it is far
  less likely to be blocked than a bare HTTP client would be.

The fallback is genuinely useful, not a stub -- the keyless setup is a
first-class path in this project, not a degraded one. Three things learned the
hard way keep it that way:

* **Wait for the results, do not just wait for the document.** `domcontentloaded`
  fires while the results list is still empty on every engine here, and an
  extractor that runs at that moment reports zero results from a page that was
  about to work perfectly.
* **Try more than one engine.** Any single engine can be having a bad day with
  this IP, this profile or this user agent. A second opinion costs one page
  load and is usually the difference between a task that proceeds and one that
  gives up.
* **Distinguish "no results" from "we were blocked".** They call for opposite
  responses from the model, and conflating them is what produces the loop where
  it rephrases the same query five times into the same wall.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from app.browser.challenge import detect_challenge
from app.tools.base import ToolContext, ToolResult, integer, schema, string
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"

#: How long to wait for an engine's first result to appear in the DOM. Short,
#: because this is on top of the page load and there is another engine to try.
_RESULTS_TIMEOUT_MS = 6000

#: A brief settle after the first result appears, so the rest of the list is
#: there too rather than just the one node we waited on.
_SETTLE_MS = 400

#: Pulls results out of DuckDuckGo's no-JavaScript endpoint. Stable markup, no
#: consent interstitial, server-rendered.
_DDG_EXTRACT_JS = """
(limit) => {
  const out = [];
  for (const node of document.querySelectorAll(".result, .web-result")) {
    const link = node.querySelector("a.result__a");
    if (!link) continue;
    const snippet = node.querySelector(".result__snippet");
    out.push({
      title: (link.textContent || "").trim(),
      url: link.href,
      snippet: (snippet?.textContent || "").trim().slice(0, 300),
    });
    if (out.length >= limit) break;
  }
  return out;
}
"""

#: The same for Bing, whose organic results are `li.b_algo` directly under
#: `#b_results`. The `>` matters: without it the ads and the "people also ask"
#: panels come along too.
_BING_EXTRACT_JS = """
(limit) => {
  const out = [];
  for (const node of document.querySelectorAll("#b_results > li.b_algo")) {
    const link = node.querySelector("h2 a[href]");
    if (!link) continue;
    const snippet = node.querySelector(".b_caption p, p.b_lineclamp2, p");
    out.push({
      title: (link.textContent || "").trim(),
      url: link.href,
      snippet: (snippet?.textContent || "").trim().slice(0, 300),
    });
    if (out.length >= limit) break;
  }
  return out;
}
"""


@dataclass(frozen=True, slots=True)
class _Engine:
    """One scrapable search engine."""

    name: str
    #: Query string is appended, URL-encoded.
    endpoint: str
    #: Selector whose appearance means "the results are rendered".
    ready: str
    #: Returns `[{title, url, snippet}]` when evaluated with a result limit.
    extract: str


#: Tried in order until one produces results. DuckDuckGo first: it is
#: server-rendered, so it answers in a fraction of the time Bing takes.
_ENGINES: tuple[_Engine, ...] = (
    _Engine(
        name="DuckDuckGo",
        endpoint="https://html.duckduckgo.com/html/?q=",
        ready=".result__a",
        extract=_DDG_EXTRACT_JS,
    ),
    _Engine(
        name="Bing",
        endpoint="https://www.bing.com/search?q=",
        ready="#b_results > li.b_algo h2 a",
        extract=_BING_EXTRACT_JS,
    ),
)


def register_tools(registry: ToolRegistry) -> None:
    """Add the search tool to `registry`."""

    @registry.tool(
        "web_search",
        "Search the web and get back a list of titles, URLs and snippets. Use "
        "this first whenever you do not already know which site to visit. It "
        "does not change the page you are currently on.",
        schema(
            query=string("What to search for, phrased as you would type it"),
            max_results=integer("How many results to return (default 6)", optional=True),
        ),
        mutates_page=False,
    )
    async def web_search(
        context: ToolContext, query: str, max_results: int = 6
    ) -> ToolResult:
        limit = max(1, min(int(max_results or 6), 10))
        blocked_by: list[str] = []

        if context.settings.tavily_api_key:
            try:
                results = await _search_tavily(context, query, limit)
            except Exception as exc:  # noqa: BLE001 - fall back rather than fail
                logger.warning("Tavily search failed (%s); falling back to the browser", exc)
                results = await _search_browser(context, query, limit, blocked_by)
        else:
            results = await _search_browser(context, query, limit, blocked_by)

        if results:
            return ToolResult.success(_render(query, results), results=results)

        # Being walled and finding nothing look identical from here, but they
        # call for opposite next moves, so never report one as the other.
        if blocked_by:
            return ToolResult.failure(
                f"Search is unavailable: {_join(blocked_by)}. Rephrasing the query "
                "will not help -- the search engines are refusing this browser, not "
                "this wording. Navigate directly to a site that is likely to have "
                "the answer instead, or ask the user which site to use.",
                blocked=True,
            )
        return ToolResult.failure(
            f"The search for {query!r} returned nothing. Try different wording, "
            "or navigate to a site you know directly."
        )


# ------------------------------------------------------------- backends ----


async def _search_tavily(
    context: ToolContext, query: str, limit: int
) -> list[dict[str, str]]:
    """Query the Tavily search API."""
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            _TAVILY_ENDPOINT,
            json={
                "api_key": context.settings.tavily_api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
            },
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": (item.get("content", "") or "")[:300],
        }
        for item in payload.get("results", [])[:limit]
    ]


async def _search_browser(
    context: ToolContext, query: str, limit: int, blocked_by: list[str]
) -> list[dict[str, str]]:
    """Scrape each engine in turn, leaving the agent's own tab alone.

    Appends a description of any bot wall encountered to `blocked_by`, so the
    caller can tell the model *why* it has no results.
    """
    session = await context.get_browser()
    encoded = quote_plus(query)

    for engine in _ENGINES:
        try:
            async with session.background_page() as page:
                await page.goto(
                    f"{engine.endpoint}{encoded}", wait_until="domcontentloaded"
                )

                wall = await detect_challenge(page)
                if wall is not None:
                    logger.info("%s presented %s", engine.name, wall)
                    blocked_by.append(f"{engine.name} presented {wall}")
                    continue

                # The results are what we came for, not the document; an
                # extractor that runs before they render reports nothing.
                try:
                    await page.wait_for_selector(
                        engine.ready, timeout=_RESULTS_TIMEOUT_MS, state="attached"
                    )
                    await page.wait_for_timeout(_SETTLE_MS)
                except Exception as exc:  # noqa: BLE001 - try the next engine
                    logger.info("%s rendered no results (%s)", engine.name, _brief(exc))
                    continue

                results = list(await page.evaluate(engine.extract, limit) or [])
        except Exception as exc:  # noqa: BLE001 - one bad engine is not fatal
            logger.warning("%s search failed: %s", engine.name, _brief(exc))
            continue

        if results:
            for item in results:
                item["url"] = _unwrap(item.get("url", ""))
            logger.info("%s returned %d results", engine.name, len(results))
            return results

    return []


# -------------------------------------------------------------- rendering --


def _unwrap(url: str) -> str:
    """Recover the destination behind an engine's click-tracking redirect.

    Both engines hand back a link to themselves rather than to the site, which
    the model then has to follow blind: it cannot tell from
    ``duckduckgo.com/l/?uddg=...`` whether it is about to open the retailer it
    wanted or an unrelated blog. Unwrapping restores the one piece of
    information the result list is for.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return url

    host, path = parsed.netloc.lower(), parsed.path
    query = parse_qs(parsed.query)

    if host.endswith("duckduckgo.com") and path == "/l/":
        # `parse_qs` has already percent-decoded the target.
        return query.get("uddg", [""])[0] or url

    if host.endswith("bing.com") and path.startswith("/ck/"):
        raw = query.get("u", [""])[0]
        if raw.startswith("a1"):
            body = raw[2:]
            try:
                return base64.urlsafe_b64decode(
                    body + "=" * (-len(body) % 4)
                ).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return url

    return url


def _render(query: str, results: list[dict[str, str]]) -> str:
    """Format results as the compact text block the model reads."""
    lines = [f"Search results for {query!r}:"]
    for position, item in enumerate(results, start=1):
        lines.append(f"{position}. {item.get('title', '(untitled)')}")
        lines.append(f"   {item.get('url', '')}")
        if item.get("snippet"):
            lines.append(f"   {item['snippet']}")
    return "\n".join(lines)


def _join(reasons: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` -- read by the model, so keep it plain."""
    if len(reasons) == 1:
        return reasons[0]
    return f"{', '.join(reasons[:-1])} and {reasons[-1]}"


def _brief(exc: Exception) -> str:
    """First line of an exception; Playwright's are many lines of advice."""
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
