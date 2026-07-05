import asyncio

from backend.app.browser.browser_manager import BrowserManager


async def main():
    async with BrowserManager(headless=False) as browser:

        page = await browser.new_page()

        await page.goto("https://github.com")

        print(await browser.title(page))
        print(await browser.current_url(page))
        await browser.screenshot(page)


asyncio.run(main())