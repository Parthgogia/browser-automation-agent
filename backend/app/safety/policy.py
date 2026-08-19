"""Safety policy: the gate every tool call passes through before execution.

The agent is autonomous, which means the interesting question is not "can it
click that" but "*should* it click that without asking". This module answers
that question, and it does so deterministically -- in Python, not by asking the
model to police itself. A model that has been talked into buying something is
also a model that can be talked into believing the purchase is safe.

Three checks run, in order:

1. **Domain policy.** An optional allowlist and a blocklist, evaluated on the
   registrable domain so that `evil.com.attacker.net` cannot pass as `evil.com`.
2. **Hard blocks.** Actions that are never permitted regardless of approval.
3. **Risk classification.** Everything else is rated. Anything at or above
   `CONFIRM` pauses the run and asks the human.

Classification deliberately errs toward asking. A spurious approval prompt
costs the user two seconds; a spurious purchase costs them money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from app.config import Settings


class RiskLevel(IntEnum):
    """How much damage could this action do if the model has misunderstood?

    Ordered, so policy can be expressed as a threshold comparison.
    """

    #: Reading, scrolling, navigating. Freely reversible.
    SAFE = 0
    #: Changes page state but not the world: filling a field, opening a tab.
    LOW = 1
    #: Consequences outside the browser that a human should sign off on:
    #: purchases, submissions, messages, deletions.
    CONFIRM = 2
    #: Never permitted by this agent, approval or not.
    BLOCKED = 3


@dataclass(slots=True)
class SafetyDecision:
    """Verdict for one proposed tool call."""

    risk: RiskLevel
    #: Shown to the human in the approval prompt, or logged on a block.
    reason: str

    @property
    def requires_approval(self) -> bool:
        return self.risk is RiskLevel.CONFIRM

    @property
    def blocked(self) -> bool:
        return self.risk is RiskLevel.BLOCKED


#: Tools whose effects never leave the browser. Everything not listed here is
#: examined more closely.
_READ_ONLY_TOOLS = frozenset(
    {
        "read_page", "extract_text", "scroll", "scroll_to_text", "hover",
        "go_back", "go_forward", "reload", "switch_tab", "wait",
        "web_search", "remember", "recall", "screenshot",
    }
)

#: Words that, in a button or link label, mean money or irreversibility.
#: Matched case-insensitively against the element label the model is targeting.
_SENSITIVE_LABEL_PATTERNS = (
    # Money
    # `\w+\s+` slots absorb the filler real buttons use: "Place YOUR order",
    # "Complete THIS purchase". Matching only the bare phrasing misses most of
    # the buttons that actually spend money.
    r"\b(buy|purchase|pay|payment|checkout)\b",
    r"\b(place|submit|confirm)\s+(\w+\s+)?(order|payment|purchase|booking|bid)\b",
    r"\border\s+now\b",
    r"\b(subscribe|upgrade|book\s+now)\b",
    r"\bcomplete\s+(\w+\s+)?(purchase|order|checkout)\b",
    r"\b(transfer|send\s+money|donate|bid\b)",
    # Destruction
    r"\b(delete|deactivate|erase|wipe)\b",
    r"\b(remove|close)\s+(\w+\s+)?account\b",
    r"\b(cancel\s+subscription|unsubscribe\s+all)\b",
    # Broadcasting
    r"\b(send|post|publish|submit|tweet|reply\s+all)\b",
    # Identity
    r"\b(sign\s+up|create\s+account|accept\s+terms|agree\s+and)\b",
)

_SENSITIVE_LABEL_RE = re.compile("|".join(_SENSITIVE_LABEL_PATTERNS), re.IGNORECASE)

#: URLs that indicate a checkout or payment flow is in progress. Any click on
#: such a page gets extra scrutiny even if the label looks innocuous.
_SENSITIVE_URL_RE = re.compile(
    r"/(checkout|payment|billing|purchase|order/confirm|transfer)\b", re.IGNORECASE
)

#: Field names whose contents must never be typed by the agent on its own.
#: Credentials are the user's to enter, via the browser profile or by hand.
_CREDENTIAL_FIELD_RE = re.compile(
    r"\b(password|passwd|cvv|cvc|card\s*number|cardnum|otp|one[-\s]?time|"
    r"security\s*code|pin\b|ssn|social\s*security)\b",
    re.IGNORECASE,
)

#: Schemes the agent must never navigate to.
_BLOCKED_SCHEMES = frozenset({"file", "chrome", "chrome-extension", "devtools", "view-source"})


class SafetyPolicy:
    """Evaluates proposed tool calls against configured guardrails."""

    def __init__(self, settings: Settings) -> None:
        self._allowed = {d.lower().lstrip(".") for d in settings.allowed_domains}
        self._blocked = {d.lower().lstrip(".") for d in settings.blocked_domains}
        self._require_approval = settings.require_approval

    # ------------------------------------------------------------- domains --

    def check_url(self, url: str) -> SafetyDecision:
        """Is the agent allowed to visit `url`?"""
        parsed = urlparse(url if "://" in url else f"https://{url}")
        scheme = (parsed.scheme or "https").lower()

        if scheme in _BLOCKED_SCHEMES:
            return SafetyDecision(
                RiskLevel.BLOCKED,
                f"Navigating to a {scheme}: URL is not permitted; it would give the "
                "agent access outside the web.",
            )

        host = (parsed.hostname or "").lower()
        if not host:
            return SafetyDecision(RiskLevel.BLOCKED, f"{url!r} is not a valid web address.")

        if self._matches(host, self._blocked):
            return SafetyDecision(RiskLevel.BLOCKED, f"{host} is on the blocked-domain list.")

        if self._allowed and not self._matches(host, self._allowed):
            return SafetyDecision(
                RiskLevel.BLOCKED,
                f"{host} is not on the allowed-domain list "
                f"({', '.join(sorted(self._allowed))}).",
            )
        return SafetyDecision(RiskLevel.SAFE, "Domain permitted.")

    @staticmethod
    def _matches(host: str, domains: set[str]) -> bool:
        """True if `host` is one of `domains` or a subdomain of one.

        Compared label-wise rather than by substring so that
        `notexample.com` does not match a rule for `example.com`.
        """
        return any(host == d or host.endswith(f".{d}") for d in domains)

    # ---------------------------------------------------------- tool calls --

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        element_label: str = "",
        current_url: str = "",
    ) -> SafetyDecision:
        """Classify one proposed tool call.

        `element_label` is the label of the element the call targets, resolved
        from the current observation by the caller. It is the single most
        informative signal available -- "Place your order" tells us far more
        than the fact that a click is about to happen.
        """
        if tool_name == "navigate":
            url_decision = self.check_url(str(arguments.get("url", "")))
            if url_decision.blocked:
                return url_decision
            return SafetyDecision(RiskLevel.SAFE, "Navigation is reversible.")

        if tool_name in _READ_ONLY_TOOLS:
            return SafetyDecision(RiskLevel.SAFE, "Read-only action.")

        if tool_name == "type_text":
            return self._evaluate_typing(arguments, element_label)

        if tool_name in {"click", "select_option", "press_key", "upload_file"}:
            return self._evaluate_interaction(tool_name, arguments, element_label, current_url)

        # An unrecognised tool is not automatically safe.
        return SafetyDecision(RiskLevel.LOW, f"Action {tool_name!r} has no specific policy.")

    def _evaluate_typing(self, arguments: dict[str, Any], element_label: str) -> SafetyDecision:
        """Typing is low risk unless the target field holds a credential."""
        haystack = f"{element_label} {arguments.get('field_description', '')}"
        if _CREDENTIAL_FIELD_RE.search(haystack):
            return SafetyDecision(
                RiskLevel.BLOCKED,
                "This looks like a password, card or one-time-code field. The agent "
                "never types credentials; sign in yourself in the browser window, or "
                "use a saved profile.",
            )
        if arguments.get("press_enter") and _SENSITIVE_LABEL_RE.search(element_label):
            return SafetyDecision(
                RiskLevel.CONFIRM,
                f"Pressing Enter in {element_label!r} may submit a sensitive form.",
            )
        return SafetyDecision(RiskLevel.LOW, "Filling a form field.")

    def _evaluate_interaction(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        element_label: str,
        current_url: str,
    ) -> SafetyDecision:
        """Clicks and their relatives: judged by what the target says it does."""
        if not self._require_approval:
            return SafetyDecision(RiskLevel.LOW, "Approval is disabled by configuration.")

        match = _SENSITIVE_LABEL_RE.search(element_label or "")
        if match:
            return SafetyDecision(
                RiskLevel.CONFIRM,
                f"About to {tool_name} {element_label!r}, which looks irreversible "
                f"(matched {match.group(0)!r}).",
            )

        if tool_name == "click" and _SENSITIVE_URL_RE.search(current_url or ""):
            return SafetyDecision(
                RiskLevel.CONFIRM,
                f"Clicking {element_label or 'an element'} during a checkout or payment "
                f"flow ({current_url}).",
            )

        if tool_name == "upload_file":
            return SafetyDecision(
                RiskLevel.CONFIRM,
                f"About to upload {arguments.get('path', 'a file')} to {current_url}.",
            )

        return SafetyDecision(RiskLevel.LOW, "Ordinary page interaction.")


@lru_cache
def get_safety_policy() -> SafetyPolicy:
    """Process-wide policy built from the current settings."""
    from app.config import get_settings

    return SafetyPolicy(get_settings())
