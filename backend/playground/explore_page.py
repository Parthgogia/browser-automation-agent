import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        
        browser = await p.chromium.launch(
            headless=False
        )

        context = await browser.new_context()

        page = await context.new_page()

        await page.goto("https://google.com")

        title = await page.title()
        url = page.url
        html = await page.content()
        await page.screenshot(path="page.png")

        await page.wait_for_timeout(5000)

        await browser.close()

        print(f"Title: {title}")
        print(f"URL: {url}")
        print(f"HTML: {html[:200]}")
        print(f"Screenshot saved as page.png")

asyncio.run(main())