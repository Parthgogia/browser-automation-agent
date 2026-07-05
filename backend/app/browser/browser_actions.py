"""
This class will contain every action the agent can perform.
Think of it as the robot's hands.

BrowserManager
      │
      ▼
gets a Page
      │
      ▼
BrowserActions
      │
      ├── goto()
      ├── click()
      ├── type()
      ├── press()
      ├── wait()
      ├── get_inputs()
      └── ...
"""


from playwright.async_api import TimeoutError


class BrowserActions:
    DEFAULT_TIMEOUT = 10000  # milliseconds

    @staticmethod #avoids unnecessary object creation.
    async def goto(page, url: str): #Navigate to a URL.
        await page.goto(url)

    @staticmethod
    async def back(page): #Go back in browser history.
        await page.go_back()

    @staticmethod
    async def forward(page): #Go forward in browser history.
        await page.go_forward()

    @staticmethod
    async def reload(page):
        await page.reload()

    @staticmethod
    async def click(page, selector: str): #Click an element.
        await page.click(selector, timeout=BrowserActions.DEFAULT_TIMEOUT)

    @staticmethod
    async def double_click(page, selector: str): #Double-click an element.
        await page.dblclick(selector, timeout=BrowserActions.DEFAULT_TIMEOUT)

    @staticmethod
    async def right_click(page, selector: str): #Right-click an element.
        await page.click(
            selector,
            button="right",
            timeout=BrowserActions.DEFAULT_TIMEOUT
        )

    @staticmethod
    async def hover(page, selector: str):
        await page.hover(selector)

    @staticmethod
    async def fill(page, selector: str, text: str): #Clear existing text and type new text.
        await page.fill(selector, text)

    @staticmethod
    async def type(page, selector: str, text: str): #Type into an element.
        await page.locator(selector).type(text)

    @staticmethod
    async def press(page, selector: str, key: str):
        await page.press(selector, key)

    @staticmethod
    async def select_option(page, selector: str, value: str):
        await page.select_option(selector, value)

    @staticmethod
    async def check(page, selector: str):
        await page.check(selector)

    @staticmethod
    async def uncheck(page, selector: str):
        await page.uncheck(selector)

    @staticmethod
    async def wait_for_selector(page, selector: str): #Wait until an element appears.
        await page.wait_for_selector(
            selector,
            timeout=BrowserActions.DEFAULT_TIMEOUT
        )

    @staticmethod
    async def wait(seconds: float): #Sleep for a number of seconds.
        import asyncio
        await asyncio.sleep(seconds)

    @staticmethod
    async def wait_for_load(page): #Wait until network activity settles.
        await page.wait_for_load_state("networkidle")

    @staticmethod
    async def text(page, selector: str): #Get an element's text.
        return await page.locator(selector).inner_text()

    @staticmethod
    async def html(page, selector: str): #Get an element's HTML.
        return await page.locator(selector).inner_html()

    @staticmethod
    async def attribute(page, selector: str, name: str): #Get an element attribute.
        return await page.locator(selector).get_attribute(name)

    @staticmethod
    async def title(page):
        return await page.title()

    @staticmethod
    async def url(page):
        return page.url

    @staticmethod
    async def scroll_down(page, pixels=1000): #Scroll down
        await page.evaluate(f"window.scrollBy(0, {pixels})")

    @staticmethod
    async def scroll_up(page, pixels=1000): #Scroll up
        await page.evaluate(f"window.scrollBy(0, {-pixels})")

    @staticmethod
    async def scroll_to_element(page, selector: str): #Scroll an element into view.
        await page.locator(selector).scroll_into_view_if_needed()

    @staticmethod
    async def screenshot(page, path="screenshot.png"): #Take a screenshot.
        await page.screenshot(path=path, full_page=True)

    @staticmethod
    async def evaluate(page, script: str): #Execute JavaScript on the page.
        return await page.evaluate(script)
    

"""
Why are all methods @staticmethod?

Notice we don't store any state in BrowserActions.

It simply receives a page and performs an action.

We use :
    await BrowserActions.click(page, "#login")
Instead of:
    actions = BrowserActions()
    await actions.click(page, "#login")

Since there's no instance data (self) involved, @staticmethod is appropriate and avoids unnecessary object creation.
"""