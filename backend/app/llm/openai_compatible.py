"""OpenAI-compatible chat providers and adaptive provider routing.

Groq, OpenRouter and Ollama expose compatible chat-completion APIs. The router
uses local Ollama for routine tasks, hosted models for complex work, and falls
through on quota, connectivity or server errors.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall, ToolSpec
from app.llm.capabilities import (
    CapabilityMatch,
    ModelCapabilities,
    RequestRequirements,
    maintained_profile,
    request_requirements,
)
from app.llm.telemetry import record_provider_request

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_DIMENSIONS = 768
_COMPLEX_TASK_MARKERS = (
    "compare", "comparison", "versus", " vs ", "across", "research", "evaluate",
    "analyze", "analyse", "recommend", "choose between", "pros and cons", "itinerary",
    "book a", "plan a", "multiple sites", "several sites",
)
_CONSTRAINT_MARKERS = (
    "under ", "below ", "less than ", "above ", "over ", "more than ", "between ",
    "at least ", "no more than ", "before ", "after ", "within ",
)


class OpenAICompatibleProvider(LLMProvider):
    """Provider using `/chat/completions`, optionally with Ollama embeddings."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        fast_model: str,
        temperature: float,
        embedding_model: str | None = None,
        supports_vision: bool = False,
        extra_payload: dict[str, Any] | None = None,
        timeout_s: float = 90,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._fast_model = fast_model
        self._temperature = temperature
        self._embedding_model = embedding_model
        self.supports_vision = supports_vision
        self._extra_payload = dict(extra_payload or {})
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._models_cache: list[dict[str, Any]] | None = None
        self._capabilities_cache: dict[str, ModelCapabilities] = {}

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        force_tool_use: bool = False,
        selected_model: str | None = None,
    ) -> LLMResponse:
        if any(message.images for message in messages) and not self.supports_vision:
            raise LLMError(f"{self.name} does not support image inputs.")

        payload: dict[str, Any] = {
            "model": selected_model or (self._fast_model if model is not None else self._model),
            "messages": [_to_openai_message(message) for message in messages],
            "temperature": temperature if temperature is not None else self._temperature,
        }
        payload.update(self._extra_payload)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            # Ollama accepts tool schemas but does not consistently enforce
            # `tool_choice` across versions of its OpenAI-compatible endpoint.
            if self.name != "ollama":
                payload["tool_choice"] = "required" if force_tool_use else "auto"

        response = await self._post("/chat/completions", payload)
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} returned an invalid chat completion.") from exc

        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise LLMError(
                    f"{self.name} returned malformed arguments for tool {name!r}.",
                    fallback=True,
                ) from exc
            if not isinstance(arguments, dict):
                raise LLMError(
                    f"{self.name} returned non-object arguments for tool {name!r}.",
                    fallback=True,
                )
            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    **({"id": item["id"]} if item.get("id") else {}),
                )
            )

        if force_tool_use and tools and not calls:
            raise LLMError(
                f"{self.name} did not produce a required tool call.", fallback=True
            )

        usage = response.get("usage") or {}
        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=calls,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "output_tokens": usage.get("completion_tokens", 0) or 0,
                "total_tokens": usage.get("total_tokens", 0) or 0,
            },
            finish_reason=choice.get("finish_reason") or "",
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._embedding_model:
            raise LLMError(f"{self.name} does not provide embeddings.")

        # Ollama's native embedding endpoint is available alongside its
        # OpenAI-compatible chat endpoint. It keeps memory embeddings local.
        endpoint = f"{_origin(self._base_url)}/api/embed"
        try:
            record_provider_request(self.name)
            response = await self._client.post(
                endpoint,
                json={"model": self._embedding_model, "input": texts},
            )
            response.raise_for_status()
            payload = response.json()
            vectors = payload.get("embeddings") or []
            if len(vectors) != len(texts):
                raise LLMError("Ollama returned an unexpected number of embeddings.")
            return [_normalise(vector) for vector in vectors]
        except httpx.HTTPStatusError as exc:
            raise _http_error(self.name, exc.response) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Could not reach {self.name} embeddings: {exc}", retryable=True
            ) from exc

    @property
    def embedding_dimensions(self) -> int:
        return _DEFAULT_EMBEDDING_DIMENSIONS

    @property
    def supports_embeddings(self) -> bool:
        return bool(self._embedding_model)

    async def resolve_capabilities(
        self, requirements: RequestRequirements, *, fast: bool = False
    ) -> CapabilityMatch:
        model_id = self._fast_model if fast else self._model
        if model_id in self._capabilities_cache:
            cached = self._capabilities_cache[model_id]
            return CapabilityMatch(cached, model_id)
        profile = maintained_profile(self.name, model_id)
        if self.name == "openrouter":
            try:
                models = await self._openrouter_models()
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                self._capabilities_cache[model_id] = profile
                return CapabilityMatch(profile)
            if model_id == "openrouter/free":
                eligible = [
                    (item, _openrouter_capabilities(item))
                    for item in models
                    if _is_free_model(item, require_image=requirements.vision)
                    and isinstance(item.get("id"), str)
                    and requirements.accepts(_openrouter_capabilities(item))
                ]
                if not eligible:
                    return CapabilityMatch(ModelCapabilities())
                item, capabilities = max(
                    eligible, key=lambda pair: pair[1].context_tokens or 0
                )
                selected = str(item.get("id"))
                return CapabilityMatch(capabilities, selected)
            item = next((entry for entry in models if entry.get("id") == model_id), None)
            if item is None:
                return CapabilityMatch(profile)
            capabilities = _openrouter_capabilities(item)
            self._capabilities_cache[model_id] = capabilities
            return CapabilityMatch(capabilities, model_id)

        if self.name == "ollama":
            live = await self._ollama_capabilities(model_id)
            capabilities = _merge_capabilities(live, profile)
            self._capabilities_cache[model_id] = capabilities
            return CapabilityMatch(capabilities, model_id)

        if self.name == "groq":
            live = await self._groq_capabilities(model_id)
            capabilities = _merge_capabilities(live, profile)
            self._capabilities_cache[model_id] = capabilities
            return CapabilityMatch(capabilities, model_id)

        self._capabilities_cache[model_id] = profile
        return CapabilityMatch(profile, model_id)

    async def _openrouter_models(self) -> list[dict[str, Any]]:
        if self._models_cache is not None:
            return self._models_cache
        response = await self._client.get(
            f"{self._base_url}/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        models = response.json().get("data")
        if not isinstance(models, list):
            raise ValueError("OpenRouter returned an invalid model catalog.")
        self._models_cache = [item for item in models if isinstance(item, dict)]
        return self._models_cache

    async def _ollama_capabilities(self, model_id: str) -> ModelCapabilities:
        try:
            response = await self._client.post(
                f"{_origin(self._base_url)}/api/show", json={"model": model_id}
            )
            response.raise_for_status()
            payload = response.json()
            if "capabilities" not in payload:
                return ModelCapabilities()
            raw_caps = payload.get("capabilities")
            model_info = payload.get("model_info") or {}
            context = next(
                (
                    value
                    for key, value in model_info.items()
                    if key.endswith(".context_length") and isinstance(value, int)
                ),
                None,
            )
            if not isinstance(raw_caps, list):
                return ModelCapabilities()
            return ModelCapabilities(
                tools="tools" in raw_caps,
                forced_tool_use=None,
                vision="vision" in raw_caps,
                context_tokens=context,
            )
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return ModelCapabilities()

    async def _groq_capabilities(self, model_id: str) -> ModelCapabilities:
        try:
            if self._models_cache is None:
                response = await self._client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
                models = response.json().get("data")
                if not isinstance(models, list):
                    return ModelCapabilities()
                self._models_cache = [item for item in models if isinstance(item, dict)]
            item = next(
                (entry for entry in self._models_cache if entry.get("id") == model_id),
                None,
            )
            if item is None:
                return ModelCapabilities()
            context = item.get("context_window")
            return ModelCapabilities(context_tokens=context if isinstance(context, int) else None)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return ModelCapabilities()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            record_provider_request(self.name)
            response = await self._client.post(
                f"{self._base_url}{path}", json=payload, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise _http_error(
                self.name, exc.response, tool_call_context=bool(payload.get("tools"))
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Could not reach {self.name}: {exc}", retryable=True) from exc
        except ValueError as exc:
            raise LLMError(
                f"{self.name} returned invalid JSON.", fallback=bool(payload.get("tools"))
            ) from exc


@dataclass(slots=True)
class _ProviderHealth:
    provider: LLMProvider
    disabled_until: float = 0


class FallbackProvider(LLMProvider):
    """Route tasks to local/hosted models and fail over across providers."""

    name = "adaptive-router"

    def __init__(
        self,
        providers: list[LLMProvider],
        cooldown_s: int = 60,
        provider_order: dict[str, list[str]] | None = None,
    ) -> None:
        if not providers:
            raise LLMError(
                "No LLM providers are configured. Add a Gemini, Groq, or "
                "OpenRouter API key, or start Ollama locally."
            )
        self._providers = [_ProviderHealth(provider) for provider in providers]
        self._provider_order = provider_order or {}
        self._cooldown_s = max(1, cooldown_s)
        self._embedding_provider = next(
            (
                entry.provider
                for entry in reversed(self._providers)
                if _supports_embeddings(entry.provider)
            ),
            None,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        force_tool_use: bool = False,
    ) -> LLMResponse:
        now = time.monotonic()
        requirements = request_requirements(messages, tools, force_tool_use)
        resolved: list[tuple[_ProviderHealth, CapabilityMatch]] = []
        for entry in self._providers:
            match = await entry.provider.resolve_capabilities(
                requirements, fast=model is not None
            )
            if requirements.accepts(match.capabilities):
                resolved.append((entry, match))
        eligible = [entry for entry, _ in resolved]
        if not eligible:
            raise LLMError(
                "No configured model is known to support this request's tool, image, "
                "and context requirements. Check configured model capabilities."
            )
        tier = _task_tier(messages)
        configured_order = self._provider_order.get(tier, [])
        order = {name: index for index, name in enumerate(configured_order)}
        eligible.sort(key=lambda entry: order.get(entry.provider.name, len(order)))
        candidates = [entry for entry in eligible if entry.disabled_until <= now]
        if not candidates:
            candidates = [min(eligible, key=lambda entry: entry.disabled_until)]

        failures: list[str] = []
        matches = {id(entry.provider): match for entry, match in resolved}
        for entry in candidates:
            match = matches[id(entry.provider)]
            try:
                response = await entry.provider.complete(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    model=model,
                    force_tool_use=force_tool_use,
                    **(
                        {"selected_model": match.model_id}
                        if isinstance(entry.provider, OpenAICompatibleProvider)
                        else {}
                    ),
                )
                _validate_tool_response(response, tools, force_tool_use)
                return response
            except LLMError as exc:
                if not exc.retryable and not exc.fallback:
                    raise
                if exc.retryable:
                    cooldown = exc.retry_after or self._cooldown_s
                    entry.disabled_until = time.monotonic() + cooldown
                failures.append(f"{entry.provider.name}: {exc}")
                logger.warning(
                    "Provider %s unavailable; trying next provider (%s)",
                    entry.provider.name,
                    exc,
                )

        summary = "; ".join(failures) or "All configured providers are cooling down."
        cooled_down = [entry.disabled_until for entry in eligible if entry.disabled_until > 0]
        retry_after = max(1, int(min(cooled_down) - time.monotonic())) if cooled_down else None
        raise LLMError(
            f"All configured LLM providers are unavailable: {summary}",
            retryable=bool(cooled_down),
            retry_after=retry_after,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_provider is None:
            raise LLMError("No configured provider supports embeddings.")
        return await self._embedding_provider.embed(texts)

    @property
    def embedding_dimensions(self) -> int:
        return self._embedding_provider.embedding_dimensions if self._embedding_provider else 0


def _supports_embeddings(provider: LLMProvider) -> bool:
    return isinstance(provider, OpenAICompatibleProvider) and provider.supports_embeddings


def _validate_tool_response(
    response: LLMResponse, tools: list[ToolSpec] | None, force_tool_use: bool
) -> None:
    if not tools:
        return
    specifications = {tool.name: tool for tool in tools}
    if force_tool_use and not response.tool_calls:
        raise LLMError("Model did not return a required tool call.", fallback=True)
    for call in response.tool_calls:
        tool = specifications.get(call.name)
        if tool is None:
            raise LLMError(
                f"Model returned an unadvertised tool name: {call.name!r}.",
                fallback=True,
            )
        error = _argument_schema_error(call.arguments, tool.parameters)
        if error:
            raise LLMError(
                f"Model returned invalid arguments for tool {call.name!r}: {error}",
                fallback=True,
            )


def _argument_schema_error(
    value: Any, schema: dict[str, Any], path: str = "arguments"
) -> str | None:
    """Validate the JSON Schema subset used by the built-in tool definitions."""
    expected = schema.get("type")
    type_matches = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    predicate = type_matches.get(expected)
    if predicate is None:
        return f"tool schema has unsupported type {expected!r} at {path}"
    if not predicate(value):
        return f"{path} must be {expected}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']!r}"
    if expected == "object":
        properties = schema.get("properties") or {}
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            return f"{path} is missing required field(s): {', '.join(missing)}"
        for name, item in value.items():
            if name in properties:
                error = _argument_schema_error(item, properties[name], f"{path}.{name}")
                if error:
                    return error
            elif schema.get("additionalProperties") is False:
                return f"{path} has an unrecognized field {name!r}"
    elif expected == "array" and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            error = _argument_schema_error(item, schema["items"], f"{path}[{index}]")
            if error:
                return error
    return None


def _openrouter_capabilities(item: dict[str, Any]) -> ModelCapabilities:
    parameters = set(item.get("supported_parameters") or [])
    architecture = item.get("architecture") or {}
    modalities = set(architecture.get("input_modalities") or [])
    context = item.get("context_length")
    return ModelCapabilities(
        tools="tools" in parameters,
        forced_tool_use="tool_choice" in parameters,
        vision="image" in modalities,
        context_tokens=context if isinstance(context, int) else None,
    )


def _is_free_model(item: dict[str, Any], *, require_image: bool = False) -> bool:
    pricing = item.get("pricing") or {}
    required_prices = ("prompt", "completion")
    try:
        if not all(float(pricing[key]) == 0 for key in required_prices):
            return False
        if require_image and float(pricing["image"]) != 0:
            return False
        return all(
            float(pricing[key]) == 0
            for key in ("image", "request")
            if key in pricing and pricing[key] not in (None, "")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _merge_capabilities(
    live: ModelCapabilities, fallback: ModelCapabilities
) -> ModelCapabilities:
    return ModelCapabilities(
        tools=live.tools if live.tools is not None else fallback.tools,
        forced_tool_use=(
            live.forced_tool_use
            if live.forced_tool_use is not None
            else fallback.forced_tool_use
        ),
        vision=live.vision if live.vision is not None else fallback.vision,
        context_tokens=(
            live.context_tokens
            if live.context_tokens is not None
            else fallback.context_tokens
        ),
    )


def _task_tier(messages: list[Message]) -> str:
    """Use local for ordinary browsing; reserve Gemini priority for hard tasks.

    Two or more independent constraints, comparative/research language, or a
    long compound request route to the complex tier. A failed last action gets
    stronger hosted recovery on the next decision.
    """
    last_result = next((m.content for m in reversed(messages) if m.role == "tool"), "")
    if last_result.startswith("FAILED:"):
        return "recovery"

    user_messages = [message.content for message in messages if message.role == "user"]
    # Agent decisions carry a compact reconstruction of the task context. Only
    # its explicit Goal section should drive task complexity; page contents,
    # action history, and the plan routinely exceed the old word-count limit.
    goal_section = next(
        (content for content in user_messages if content.startswith("Goal:")), None
    )
    if goal_section is not None:
        goal = goal_section.partition("\n\n")[0][len("Goal:") :].strip().lower()
    else:
        # Keep direct provider callers compatible with plain user prompts.
        goal = "\n".join(user_messages).lower()
    if any(marker in goal for marker in _COMPLEX_TASK_MARKERS):
        return "complex"
    constraints = sum(goal.count(marker) for marker in _CONSTRAINT_MARKERS)
    words = len(goal.split())
    if constraints >= 2 or words > 35:
        return "complex"
    return "routine"


def _to_openai_message(message: Message) -> dict[str, Any]:
    content: Any = message.content
    if message.images:
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"type": "text", "text": message.content})
        for image in message.images:
            encoded = base64.b64encode(image).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        content = parts

    rendered: dict[str, Any] = {"role": message.role, "content": content}
    if message.name:
        rendered["name"] = message.name
    if message.tool_call_id:
        rendered["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        rendered["tool_calls"] = [
            {
                "id": call.id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for index, call in enumerate(message.tool_calls)
        ]
    return rendered


def _http_error(
    provider: str, response: httpx.Response, *, tool_call_context: bool = False
) -> LLMError:
    status = response.status_code
    retryable = status in {408, 409, 425, 429} or status >= 500
    retry_after = _retry_after(response) if retryable else None
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or payload.get("message")
    except (ValueError, AttributeError):
        detail = None
    detail = str(detail or response.text or "request failed")[:400]
    return LLMError(
        f"{provider} returned HTTP {status}: {detail}",
        retryable=retryable,
        retry_after=retry_after,
        fallback=status == 400 and tool_call_context,
    )


def _retry_after(response: httpx.Response) -> float | None:
    for header in (
        "retry-after",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-requests-day",
    ):
        value = response.headers.get(header)
        if value:
            seconds = _duration_seconds(value)
            if seconds is not None:
                return max(1.0, min(seconds, 86_400.0))
    value = response.headers.get("retry-after")
    if value:
        try:
            reset_at = parsedate_to_datetime(value).timestamp()
            return max(1.0, min(reset_at - time.time(), 86_400.0))
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def _duration_seconds(value: str) -> float | None:
    """Parse numeric seconds or provider values such as `2m59.5s`."""
    try:
        return float(value)
    except ValueError:
        pass

    match = re.fullmatch(
        r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
        value.strip().lower(),
    )
    if not match or not any(match.groupdict().values()):
        return None
    return (
        float(match.group("hours") or 0) * 3600
        + float(match.group("minutes") or 0) * 60
        + float(match.group("seconds") or 0)
    )


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _normalise(vector: list[float]) -> list[float]:
    magnitude = sum(float(value) * float(value) for value in vector) ** 0.5
    if not magnitude:
        return [float(value) for value in vector]
    return [float(value) / magnitude for value in vector]
