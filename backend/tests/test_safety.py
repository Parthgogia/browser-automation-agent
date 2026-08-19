"""The safety policy is the one component where a wrong answer costs money."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.safety.policy import RiskLevel, SafetyPolicy


@pytest.fixture
def policy() -> SafetyPolicy:
    return SafetyPolicy(Settings(require_approval=True))


# ------------------------------------------------------------ risky labels --


@pytest.mark.parametrize(
    "label",
    [
        "Place your order",
        "Buy now",
        "Complete purchase",
        "Pay 1,29,900",
        "Delete repository",
        "Deactivate account",
        "Send message",
        "Subscribe",
        "Confirm booking",
    ],
)
def test_irreversible_clicks_need_approval(policy: SafetyPolicy, label: str) -> None:
    decision = policy.evaluate("click", {"index": 3}, element_label=label)
    assert decision.requires_approval, f"{label!r} should have required approval"


@pytest.mark.parametrize(
    "label",
    ["Next page", "Show more results", "Sort by price", "Filter", "Home", "Reviews"],
)
def test_ordinary_clicks_run_unattended(policy: SafetyPolicy, label: str) -> None:
    decision = policy.evaluate("click", {"index": 3}, element_label=label)
    assert not decision.requires_approval
    assert not decision.blocked


def test_checkout_pages_escalate_even_bland_clicks(policy: SafetyPolicy) -> None:
    decision = policy.evaluate(
        "click",
        {"index": 3},
        element_label="Continue",
        current_url="https://shop.example.com/checkout/payment",
    )
    assert decision.requires_approval


# ------------------------------------------------------------- credentials --


@pytest.mark.parametrize(
    "label",
    ['type="password"', "Password", "CVV", "Card number", "One-time code", "Enter PIN"],
)
def test_credential_fields_are_blocked_outright(policy: SafetyPolicy, label: str) -> None:
    decision = policy.evaluate("type_text", {"index": 1, "text": "hunter2"},
                               element_label=label)
    assert decision.blocked, f"{label!r} should never be typed into"


def test_ordinary_typing_is_allowed(policy: SafetyPolicy) -> None:
    decision = policy.evaluate(
        "type_text", {"index": 1, "text": "rtx 5070 laptop"}, element_label="Search"
    )
    assert decision.risk is RiskLevel.LOW


# ----------------------------------------------------------------- domains --


def test_blocklist_matches_subdomains_but_not_lookalikes() -> None:
    policy = SafetyPolicy(Settings(blocked_domains=["example.com"]))
    assert policy.check_url("https://example.com/a").blocked
    assert policy.check_url("https://shop.example.com/a").blocked
    # The classic bypass: a domain that merely *ends with* the blocked string.
    assert not policy.check_url("https://notexample.com/a").blocked


def test_allowlist_excludes_everything_else() -> None:
    policy = SafetyPolicy(Settings(allowed_domains=["amazon.in"]))
    assert not policy.check_url("https://www.amazon.in/dp/x").blocked
    assert policy.check_url("https://flipkart.com").blocked


def test_non_web_schemes_are_blocked() -> None:
    policy = SafetyPolicy(Settings())
    assert policy.check_url("file:///C:/Users/me/.ssh/id_rsa").blocked
    assert policy.check_url("chrome://settings").blocked


def test_navigation_itself_is_never_risky() -> None:
    policy = SafetyPolicy(Settings())
    decision = policy.evaluate("navigate", {"url": "https://www.amazon.in"})
    assert decision.risk is RiskLevel.SAFE


def test_disabling_approval_downgrades_risky_clicks() -> None:
    policy = SafetyPolicy(Settings(require_approval=False))
    decision = policy.evaluate("click", {"index": 1}, element_label="Place your order")
    assert not decision.requires_approval
