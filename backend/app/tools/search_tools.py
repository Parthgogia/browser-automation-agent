"""Web search.

Search is how a task that starts with "find the cheapest..." gets from a
sentence to a set of URLs worth visiting. Two backends, chosen automatically:

* **Tavily**, when `TAVILY_API_KEY` is set. Purpose-built for agents: returns
  clean titles, URLs and extracted snippets in one call.
* **The browser itself**, otherwise. We drive a throwaway background tab to
  DuckDuckGo and scrape the results. This needs no key and no extra
  dependency, and because it is a real browser with a real profile it is far
  less likely to be blocked than a bare HTTP client would be.

The fallback is genuinely useful, not a stub -- the keyless setup is a
first-class path in this project, not a degraded one.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.tools.base import ToolContext, ToolResult, integer, schema, string
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"

#: DuckDuckGo's no-JavaScript endpoint. Stable markup, no consent interstitial.
_DDG_ENDPOINT = "https://html.duckduckgo.com/html/?q="

#: Runs inside the background tab to pull results out of DuckDuckGo's HTML.
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

        if context.settings.tavily_api_key:
            try:
                results = await _search_tavily(context, query, limit)
            except Exception as exc:  # noqa: BLE001 - fall back rather than fail
                logger.warning("Tavily search failed (%s); falling back to the browser", exc)
                results = await _search_browser(context, query, limit)
        else:
            results = await _search_browser(context, query, limit)

        if not results:
            return ToolResult.failure(
                f"The search for {query!r} returned nothing. Try different wording, "
                "or navigate to a site you know directly."
            )
        return ToolResult.success(_render(query, results), results=results)


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
    context: ToolContext, query: str, limit: int
) -> list[dict[str, str]]:
    """Scrape DuckDuckGo in a background tab, leaving the active tab alone."""
    from urllib.parse import quote_plus

    session = await context.get_browser()
    async with session.background_page() as page:
        await page.goto(f"{_DDG_ENDPOINT}{quote_plus(query)}", wait_until="domcontentloaded")
        results = await page.evaluate(_DDG_EXTRACT_JS, limit)
    return list(results or [])


def _render(query: str, results: list[dict[str, str]]) -> str:
    """Format results as the compact text block the model reads."""
    lines = [f"Search results for {query!r}:"]
    for position, item in enumerate(results, start=1):
        lines.append(f"{position}. {item.get('title', '(untitled)')}")
        lines.append(f"   {item.get('url', '')}")
        if item.get("snippet"):
            lines.append(f"   {item['snippet']}")
    return "\n".join(lines)
