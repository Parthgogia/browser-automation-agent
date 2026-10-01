"""Construction of the configured LLM provider."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider named by `settings.llm_provider`."""
    if settings.llm_provider == "gemini":
        from app.llm.gemini import GeminiProvider

        return GeminiProvider(settings)

    if settings.llm_provider in {"adaptive", "free"}:
        from app.llm.openai_compatible import FallbackProvider, OpenAICompatibleProvider

        providers: list[LLMProvider] = []
        # Keep Ollama in the chain even without a hosted key: it is the default
        # for routine work and the local fallback for every task tier.
        providers.append(_ollama_provider(settings, OpenAICompatibleProvider))
        if settings.groq_api_key:
            providers.append(
                OpenAICompatibleProvider(
                    name="groq",
                    base_url=settings.groq_base_url,
                    api_key=settings.groq_api_key,
                    model=settings.groq_model,
                    fast_model=settings.groq_fast_model,
                    temperature=settings.llm_temperature,
                )
            )
        if settings.openrouter_api_key:
            providers.append(
                OpenAICompatibleProvider(
                    name="openrouter",
                    base_url=settings.openrouter_base_url,
                    api_key=settings.openrouter_api_key,
                    model=settings.openrouter_model,
                    fast_model=settings.openrouter_fast_model,
                    temperature=settings.llm_temperature,
                    supports_vision=True,
                )
            )
        if settings.nvidia_api_key:
            nvidia_models = (
                (
                    "nvidia_muse",
                    settings.nvidia_muse_model,
                    True,
                    {"top_p": 0.95},
                ),
                (
                    "nvidia_glm",
                    settings.nvidia_glm_model,
                    False,
                    {
                        "reasoning_effort": "low",
                        "chat_template_kwargs": {"clear_thinking": True},
                    },
                ),
                (
                    "nvidia_nemotron",
                    settings.nvidia_nemotron_model,
                    False,
                    {
                        "max_tokens": 1024,
                        "chat_template_kwargs": {"enable_thinking": True},
                        "reasoning_budget": 512,
                    },
                ),
            )
            for name, model_id, supports_vision, extra_payload in nvidia_models:
                providers.append(
                    OpenAICompatibleProvider(
                        name=name,
                        base_url=settings.nvidia_base_url,
                        api_key=settings.nvidia_api_key,
                        model=model_id,
                        fast_model=model_id,
                        temperature=settings.llm_temperature,
                        supports_vision=supports_vision,
                        extra_payload=extra_payload,
                    )
                )
        gemini_keys = (
            settings.gemini_api_key,
            settings.gemini_api_key_2,
            settings.gemini_api_key_3,
        )
        if any(key.strip() for key in gemini_keys):
            from app.llm.gemini import GeminiProvider

            providers.append(GeminiProvider(settings))

        return FallbackProvider(
            providers,
            settings.llm_fallback_cooldown_s,
            provider_order={
                "routine": [
                    "ollama",
                    "groq",
                    "nvidia_nemotron",
                    "nvidia_glm",
                    "nvidia_muse",
                    "openrouter",
                    "gemini",
                ],
                "complex": [
                    "gemini",
                    "nvidia_glm",
                    "nvidia_muse",
                    "nvidia_nemotron",
                    "groq",
                    "openrouter",
                    "ollama",
                ],
                "recovery": [
                    "nvidia_glm",
                    "nvidia_nemotron",
                    "nvidia_muse",
                    "groq",
                    "openrouter",
                    "gemini",
                    "ollama",
                ],
            },
        )

    if settings.llm_provider in {"groq", "openrouter", "ollama"}:
        from app.llm.openai_compatible import OpenAICompatibleProvider

        if settings.llm_provider == "ollama":
            return _ollama_provider(settings, OpenAICompatibleProvider)
        if settings.llm_provider == "groq":
            api_key = settings.groq_api_key
            base_url = settings.groq_base_url
            model = settings.groq_model
            fast_model = settings.groq_fast_model
            supports_vision = False
        else:
            api_key = settings.openrouter_api_key
            base_url = settings.openrouter_base_url
            model = settings.openrouter_model
            fast_model = settings.openrouter_fast_model
            supports_vision = True
        if not api_key:
            raise LLMError(f"{settings.llm_provider.upper()}_API_KEY is required.")
        return OpenAICompatibleProvider(
            name=settings.llm_provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            fast_model=fast_model,
            temperature=settings.llm_temperature,
            supports_vision=supports_vision,
        )

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
    logger.info("LLM provider: %s", provider.name)
    return provider


def _ollama_provider(settings: Settings, provider_type):
    """Build the local chat and embedding endpoint once per provider graph."""
    return provider_type(
        name="ollama",
        base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
        api_key="ollama",
        model=settings.ollama_model,
        fast_model=settings.ollama_fast_model,
        temperature=settings.llm_temperature,
        embedding_model=settings.ollama_embedding_model,
        supports_vision=True,
        timeout_s=180,
    )
