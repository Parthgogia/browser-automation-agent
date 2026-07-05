import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        context = await browser.new_context(
            viewport={"width": 1400, "height": 900}
        )

        page = await context.new_page()

        # open google
        await page.goto("https://www.google.com")

        #accept cookies if prompt appears
        try:
            await page.get_by_role("button", name="Accept all").click(timeout=3000)
        except:
            pass

        #find search bar and type query
        search_box = page.get_by_role("combobox")
        await search_box.press_sequentially("Playwright Python") #simulate typing the query
        await search_box.press("Enter")

        await page.wait_for_load_state("networkidle")

        titles = page.locator("h3") #locator() doesn't immediately return elements. It creates a live locator that Playwright can query whenever needed.
        count = await titles.count()

        for i in range(min(5, count)):
            title = await titles.nth(i).inner_text()
            print(f"{i + 1}. {title}")

        # titles.nth(i) selects the ith matching <h3>.
        # .inner_text() retrieves the visible text that a user sees.

        await page.screenshot(
            path="google_results.png",
            full_page=True
        )

        await browser.close()

asyncio.run(main())


