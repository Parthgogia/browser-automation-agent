"""
Responsibilities:
PageObserver should do three things:

Scan the page.
Build an internal mapping id -> Locator.
Return JSON describing the interactive elements.

"""

from playwright.async_api import Page


class PageObserver:
    def __init__(self):
        self.element_map = {}

    async def extract_interactive_elements(self, page: Page):
        """
        Extract interactive elements from the current page.

        Returns:
            List[dict]
        """

        self.element_map.clear()

        selectors = [
            "button",
            "a[href]",
            "input",
            "textarea",
            "select",
            "[role='button']",
            "[role='link']",
            "[contenteditable='true']"
        ]

        elements = []
        idx = 0

        for selector in selectors:
            locators = page.locator(selector)
            count = await locators.count()

            for i in range(count):
                locator = locators.nth(i)
                if not await locator.is_visible():
                    continue

                tag = await locator.evaluate(
                    "(el) => el.tagName.toLowerCase()"
                )

                text = (
                    await locator.inner_text()
                ).strip()

                placeholder = await locator.get_attribute("placeholder")
                href = await locator.get_attribute("href")
                input_type = await locator.get_attribute("type")
                enabled = await locator.is_enabled()
                
                self.element_map[idx] = locator

                elements.append({
                    "id": idx,
                    "tag": tag,
                    "text": text,
                    "placeholder": placeholder,
                    "href": href,
                    "type": input_type,
                    "enabled": enabled
                })

                idx += 1

        return elements

    async def get_visible_text(self, page: Page): #Return all visible page text.
        return await page.locator("body").inner_text()

    def get_locator(self, element_id: int): #Retrieve locator by ID.
        return self.element_map.get(element_id)

    def clear(self): #Clear cached element mapping.
        self.element_map.clear()



"""
A more robust approach is to inject a small JavaScript snippet into the page that assigns each interactive element a stable 
attribute such as:

<button data-agent-id="17">

Then the observer can rediscover elements by data-agent-id, making IDs stable across rescans and allowing the agent to reason 
about the page more reliably.
"""