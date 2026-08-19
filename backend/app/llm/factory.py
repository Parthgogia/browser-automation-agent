"""Construction of the configured LLM provider."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider named by `settings.llm_provider`."""
    if settings.llm_provider == "gemini":
        from app.llm.gemini import GeminiProvider

        return GeminiProvider(settings)

    from app.llm.mock import MockProvider

    return MockProvider(settings)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Process-wide provider singleton.

    Cached because provider construction opens an HTTP client; there is no
    reason to build one per request.
    """
    settings = get_settings()
    provider = build_provider(settings)
    logger.info("LLM provider: %s (model=%s)", provider.name, settings.llm_model)
    return provider
