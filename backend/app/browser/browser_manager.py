"""   BrowserManager Responsibilities

Launch Browser
↓
Create Context
↓
Create Pages
↓
Save Sessions
↓
Load Sessions
↓
Take Screenshots
↓
Close Browser
↓
Cleanup

Everything browser-related lives here.
"""

from playwright.async_api import async_playwright

class BrowserManager:

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None


    async def start(self): #Start Playwright and launch the browser.

        self.playwright = await async_playwright().start()
        
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless
        )

        self.context = await self.browser.new_context()
        

    async def stop(self): #Close browser and stop Playwright

        if self.context:
            await self.context.close()

        if self.browser:
            await self.browser.close()

        if self.playwright:
            await self.playwright.stop()

    async def new_page(self): #Create a new page in the current browser context and return it.
        return await self.context.new_page()

    async def get_pages(self): #Return a list of all open pages in the current browser context.
        return self.context.pages

    async def close_page(self, page): #Close the specified page.
        await page.close()

    async def clear_cookies(self): #Clear all cookies in the current browser context.
        await self.context.clear_cookies()

    async def current_url(self, page): #Return the current URL of the page.
        return page.url

    async def title(self, page): #Return current page title.
        return await page.title()

    async def screenshot(self, page, path="screenshot.png"): #Take a screenshot of the current page and save it to the specified path.
        await page.screenshot(path=path, full_page=True)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):  #exc_type → ValueError, exc → ValueError("Invalid") , tb → the traceback object
        if exc:
            print(f"An error occurred: {exc}")
        await self.stop()

"""
async with BrowserManager() as browser:
│
├── __aenter__()
│      │
│      ├── start Playwright
│      ├── launch browser
│      └── return BrowserManager
│
├── Code runs
│      │
│      ├── new_page()
│      ├── goto()
│      ├── click()
│      └── ...
│
└── __aexit__()
       │
       ├── close context
       ├── close browser
       └── stop Playwright
"""