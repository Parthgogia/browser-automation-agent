import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        context = await browser.new_context()

        page = await context.new_page()

        await page.goto("https://the-internet.herokuapp.com/")

        checkbox = page.get_by_role(
            "link",
            name = "Form Authentication"
        )

        await checkbox.click()

        username_input = page.get_by_role(
            "textbox",
            name = "username"
        )

        password_input = page.get_by_role(
            "textbox",
            name = "password"
        )

        await username_input.press_sequentially("tomsmith")
        await password_input.press_sequentially("SuperSecretPassword!")

        submit_button = page.get_by_role(
            "button",
            name = "Login"
        )
        await submit_button.click()

        try:
            success = page.get_by_text("You logged into a secure area!")
            await success.wait_for(timeout=5000)  # Wait for the success message to appear
            message = await success.inner_text()
            print(message)

            await page.screenshot(
                path="login_results.png",
                full_page=True
            )

            logout_button = page.get_by_role(
                "link",
                name = "Logout"
            )
            await logout_button.click()

            try:
                success = page.get_by_text("You logged out of the secure area!")
                await success.wait_for(timeout=5000)
                message = await success.inner_text()
                print(message)
            except:
                print("Failed to find success message after logout.")

        except:
            print("Login failed.")

        await browser.close()

asyncio.run(main())