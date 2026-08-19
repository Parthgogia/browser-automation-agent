"""Google Gemini provider, via the `google-genai` SDK.

Two things about Gemini shape this file:

* Gemini has no `tool` role. A tool result is a *user* turn containing a
  `functionResponse` part, keyed by tool **name** rather than by call id. The
  translation happens in :func:`_to_contents`.
* Gemini validates function schemas against OpenAPI 3.0, not full JSON Schema.
  `_sanitise_schema` strips the constructs it rejects rather than letting the
  request fail at the API boundary with an opaque 400.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import Settings
from app.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
)

logger = logging.getLogger(__name__)

#: Schema keywords Gemini's function-calling parser rejects outright.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"$schema", "$ref", "$defs", "definitions", "anyOf", "oneOf", "allOf",
     "additionalProperties", "patternProperties", "not", "const", "default"}
)

#: Transient API failures worth retrying. Everything else fails fast.
_RETRYABLE_MARKERS = ("429", "500", "502", "503", "504", "deadline", "unavailable")

_MAX_ATTEMPTS = 4


def _sanitise_schema(schema: Any) -> Any:
    """Recursively drop JSON Schema keywords Gemini cannot parse."""
    if isinstance(schema, dict):
        return {
            key: _sanitise_schema(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_sanitise_schema(item) for item in schema]
    return schema


class GeminiProvider(LLMProvider):
    """LLM backend backed by the Gemini API."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Either add a key to .env, or set "
                "LLM_PROVIDER=mock to run without one."
            )
        # Imported lazily so that the mock provider works in an environment
        # where google-genai is not installed at all.
        from google import genai

        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._default_model = settings.llm_model

    # ------------------------------------------------------------ requests --

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        force_tool_use: bool = False,
    ) -> LLMResponse:
        from google.genai import types

        system_instruction, contents = _to_contents(messages, types)

        config_kwargs: dict[str, Any] = {
            "temperature": (
                temperature if temperature is not None else self._settings.llm_temperature
            ),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            config_kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=_sanitise_schema(tool.parameters),
                        )
                        for tool in tools
                    ]
                )
            ]
            if force_tool_use:
                # "ANY" makes a function call the only legal output, which is
                # exactly what the act loop wants: prose there is a dead end.
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )

        response = await self._request(
            model=model or self._default_model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return _to_llm_response(response)

    async def _request(self, **kwargs: Any) -> Any:
        """Call the API with bounded exponential backoff.

        Free-tier Gemini rate-limits aggressively, and a browser agent issues a
        request per step, so retrying 429s is the difference between a task
        that completes and one that dies halfway.
        """
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return await self._client.aio.models.generate_content(**kwargs)
            except Exception as exc:  # SDK raises a family of error types
                last_error = exc
                message = str(exc).lower()
                if not any(marker in message for marker in _RETRYABLE_MARKERS):
                    raise LLMError(f"Gemini request failed: {exc}") from exc
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                delay = 2**attempt
                logger.warning("Gemini request failed (%s); retrying in %ss", exc, delay)
                await asyncio.sleep(delay)
        raise LLMError(f"Gemini request failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    # ---------------------------------------------------------- embeddings --

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.aio.models.embed_content(
                model=self._settings.embedding_model,
                contents=texts,
            )
        except Exception as exc:
            raise LLMError(f"Gemini embedding failed: {exc}") from exc
        return [list(embedding.values or []) for embedding in (response.embeddings or [])]

    @property
    def embedding_dimensions(self) -> int:
        # gemini-embedding-001 defaults to 3072; the older text-embedding-004
        # is 768. Keyed by model so the memory table is created correctly.
        return 768 if "text-embedding-004" in self._settings.embedding_model else 3072


# ------------------------------------------------------------ translation --


def _to_contents(messages: list[Message], types: Any) -> tuple[str, list[Any]]:
    """Convert our messages into Gemini `Content` objects.

    Returns the system instruction separately, because Gemini takes it as
    request configuration rather than as a conversation turn.
    """
    system_parts: list[str] = []
    contents: list[Any] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue

        if message.role == "tool":
            # Gemini models a tool result as a user turn holding a
            # functionResponse part, matched to the call by name.
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=message.name or "tool",
                            response={"result": message.content},
                        )
                    ],
                )
            )
            continue

        parts: list[Any] = []
        if message.content:
            parts.append(types.Part.from_text(text=message.content))
        for image in message.images:
            parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
        for call in message.tool_calls:
            parts.append(
                types.Part.from_function_call(name=call.name, args=call.arguments)
            )
        if not parts:
            # Gemini rejects empty turns; a placeholder keeps the alternation
            # of roles intact.
            parts.append(types.Part.from_text(text="(no content)"))

        contents.append(
            types.Content(role="model" if message.role == "assistant" else "user", parts=parts)
        )

    return "\n\n".join(system_parts), contents


def _to_llm_response(response: Any) -> LLMResponse:
    """Flatten a Gemini response into our provider-neutral shape."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    finish_reason = ""

    for candidate in response.candidates or []:
        finish_reason = str(getattr(candidate, "finish_reason", "") or "")
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                tool_calls.append(ToolCall(name=call.name, arguments=dict(call.args or {})))
        # One candidate is all we ever request.
        break

    usage: dict[str, int] = {}
    metadata = getattr(response, "usage_metadata", None)
    if metadata is not None:
        usage = {
            "prompt_tokens": getattr(metadata, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(metadata, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(metadata, "total_token_count", 0) or 0,
        }

    return LLMResponse(
        text="\n".join(text_parts).strip(),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
    )
