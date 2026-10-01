"""Capability requirements and maintained profiles for adaptive model routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.base import Message, ToolSpec


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Features a configured model is known to support; None means unknown."""

    tools: bool | None = None
    forced_tool_use: bool | None = None
    vision: bool | None = None
    context_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class RequestRequirements:
    """Minimum model features needed to safely serve one completion."""

    tools: bool
    forced_tool_use: bool
    vision: bool
    context_tokens: int

    def accepts(self, capabilities: ModelCapabilities) -> bool:
        if self.tools and capabilities.tools is not True:
            return False
        if self.forced_tool_use and capabilities.forced_tool_use is not True:
            return False
        if self.vision and capabilities.vision is not True:
            return False
        return (
            capabilities.context_tokens is not None
            and capabilities.context_tokens >= self.context_tokens
        )


@dataclass(frozen=True, slots=True)
class CapabilityMatch:
    capabilities: ModelCapabilities
    #: OpenRouter free routing resolves to a concrete free model per request.
    model_id: str | None = None


def request_requirements(
    messages: list[Message], tools: list[ToolSpec] | None, force_tool_use: bool
) -> RequestRequirements:
    # A conservative chars/4 estimate avoids adding a tokenizer dependency.
    # Include the tool schemas and reserve 1K tokens per image for vision input.
    text = sum(len(message.content) for message in messages)
    tool_text = len(
        json.dumps(
            [
                {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
                for tool in tools or []
            ]
        )
    )
    image_count = sum(len(message.images) for message in messages)
    token_estimate = 2048 + (text + tool_text + 3) // 4 + image_count * 1024
    return RequestRequirements(
        tools=bool(tools),
        forced_tool_use=force_tool_use and bool(tools),
        vision=image_count > 0,
        context_tokens=max(1, token_estimate),
    )


def maintained_profile(provider: str, model_id: str) -> ModelCapabilities:
    """Known provider/model profiles used when live metadata is absent."""
    normalized = model_id.lower()
    if provider == "gemini" and normalized.startswith("gemini-"):
        return ModelCapabilities(True, True, True, 1_000_000)
    if provider == "groq" and normalized in {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }:
        return ModelCapabilities(True, True, False, 131_072)
    if provider == "ollama" and normalized in {"qwen3.5:9b", "qwen3.5:4b"}:
        return ModelCapabilities(True, True, True, 32_768)
    if provider == "nvidia_muse" and normalized == "meta/muse-glimmer-30b":
        return ModelCapabilities(True, True, True, 131_072)
    if provider == "nvidia_glm" and normalized == "z-ai/glm-5.3-flash":
        # Text and forced tool calls passed, but the user's endpoint image test
        # failed, so keep this model out of screenshot-based decisions.
        return ModelCapabilities(True, True, False, 1_048_576)
    if provider == "nvidia_nemotron" and normalized == "nvidia/nemotron-3.5-lightning-30b-a3b":
        return ModelCapabilities(True, True, False, 1_048_576)
    return ModelCapabilities()
