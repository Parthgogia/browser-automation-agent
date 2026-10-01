from __future__ import annotations

import httpx
import pytest

from app.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall, ToolSpec
from app.llm.capabilities import CapabilityMatch, ModelCapabilities, RequestRequirements
from app.llm.openai_compatible import (
    FallbackProvider,
    OpenAICompatibleProvider,
    _http_error,
    _task_tier,
)


class StubProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        response: LLMResponse | Exception,
        capabilities: ModelCapabilities,
    ) -> None:
        self.name = name
        self.response = response
        self.capabilities = capabilities
        self.calls = 0

    async def resolve_capabilities(
        self, requirements: RequestRequirements, *, fast: bool = False
    ) -> CapabilityMatch:
        return CapabilityMatch(self.capabilities, self.name)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        model: str | None = None,
        force_tool_use: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]

    @property
    def embedding_dimensions(self) -> int:
        return 3


@pytest.fixture
def navigate_tool() -> ToolSpec:
    return ToolSpec(
        name="navigate",
        description="Open a URL",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )


def _caps(
    *,
    tools: bool = True,
    forced: bool = True,
    vision: bool = False,
    context: int = 128_000,
) -> ModelCapabilities:
    return ModelCapabilities(tools, forced, vision, context)


def test_task_tier_uses_original_goal_not_reconstructed_page_context() -> None:
    context = "\n\n".join(
        [
            "Goal: Open the homepage",
            "Task interpretation: Navigate to the requested destination.",
            "Plan:\n1. Open the homepage",
            "Current page:\n" + ("Rendered page text " * 100),
        ]
    )

    assert _task_tier([Message.user(context)]) == "routine"


def test_task_tier_uses_complexity_signals_from_original_goal() -> None:
    messages = [
        Message.user(
            "Goal: Compare prices across three stores and find the cheapest option.\n\n"
            + ("Page context " * 100)
        )
    ]

    assert _task_tier(messages) == "complex"


@pytest.mark.asyncio
async def test_router_skips_models_missing_requirements_and_keeps_configured_order(
    navigate_tool: ToolSpec,
) -> None:
    no_vision = StubProvider("text-only", LLMResponse(), _caps(vision=False))
    no_forced_tools = StubProvider(
        "no-forced-tools", LLMResponse(), _caps(forced=False, vision=True)
    )
    too_small = StubProvider(
        "too-small", LLMResponse(), _caps(vision=True, context=1024)
    )
    unknown = StubProvider("unknown", LLMResponse(), ModelCapabilities())
    compatible = StubProvider(
        "compatible", LLMResponse(tool_calls=[ToolCall("navigate", {"url": "https://x.test"})]),
        _caps(vision=True),
    )
    router = FallbackProvider(
        [no_vision, no_forced_tools, too_small, unknown, compatible],
        provider_order={
            "routine": ["text-only", "no-forced-tools", "too-small", "unknown", "compatible"]
        },
    )

    result = await router.complete(
        [Message.user("Read this screen", images=[b"image"])],
        tools=[navigate_tool],
        force_tool_use=True,
    )

    assert result.tool_calls[0].name == "navigate"
    assert no_vision.calls == 0
    assert no_forced_tools.calls == 0
    assert too_small.calls == 0
    assert unknown.calls == 0
    assert compatible.calls == 1


@pytest.mark.asyncio
async def test_invalid_tool_output_falls_back_without_cooling_down_provider(
    navigate_tool: ToolSpec,
) -> None:
    invalid = StubProvider(
        "first",
        LLMResponse(tool_calls=[ToolCall("navigate", {"url": 42})]),
        _caps(),
    )
    valid = StubProvider(
        "second",
        LLMResponse(tool_calls=[ToolCall("navigate", {"url": "https://x.test"})]),
        _caps(),
    )
    router = FallbackProvider([invalid, valid], provider_order={"routine": ["first", "second"]})

    response = await router.complete(
        [Message.user("Open this site")], tools=[navigate_tool], force_tool_use=True
    )

    assert response.tool_calls[0].arguments["url"] == "https://x.test"
    assert invalid.calls == 1
    assert router._providers[0].disabled_until == 0
    assert valid.calls == 1


@pytest.mark.asyncio
async def test_router_falls_back_for_missing_required_call_and_unknown_tool(
    navigate_tool: ToolSpec,
) -> None:
    first = StubProvider("first", LLMResponse(text="I'll do it"), _caps())
    second = StubProvider(
        "second",
        LLMResponse(tool_calls=[ToolCall("navigate", {"url": "https://x.test"})]),
        _caps(),
    )
    router = FallbackProvider([first, second])

    result = await router.complete(
        [Message.user("Open a site")], tools=[navigate_tool], force_tool_use=True
    )

    assert result.tool_calls[0].name == "navigate"
    assert first.calls == second.calls == 1


@pytest.mark.asyncio
async def test_router_falls_back_for_unadvertised_tool_name(
    navigate_tool: ToolSpec,
) -> None:
    first = StubProvider(
        "first", LLMResponse(tool_calls=[ToolCall("delete_everything", {})]), _caps()
    )
    second = StubProvider(
        "second",
        LLMResponse(tool_calls=[ToolCall("navigate", {"url": "https://x.test"})]),
        _caps(),
    )
    router = FallbackProvider([first, second])

    response = await router.complete(
        [Message.user("Open a site")], tools=[navigate_tool], force_tool_use=True
    )

    assert response.tool_calls[0].name == "navigate"
    assert router._providers[0].disabled_until == 0


@pytest.mark.asyncio
async def test_openrouter_free_selects_a_free_capable_model_and_caches_catalog() -> None:
    requests: list[httpx.Request] = []
    models = [
        {
            "id": "example/no-tools-free",
            "supported_parameters": ["temperature"],
            "architecture": {"input_modalities": ["text"]},
            "context_length": 128_000,
            "pricing": {"prompt": "0", "completion": "0", "image": "0"},
        },
        {
            "id": "example/tool-vision-free",
            "supported_parameters": ["tools", "tool_choice"],
            "architecture": {"input_modalities": ["text", "image"]},
            "context_length": 64_000,
            "pricing": {"prompt": "0", "completion": "0", "image": "0"},
        },
        {
            "id": "example/tool-paid",
            "supported_parameters": ["tools", "tool_choice"],
            "architecture": {"input_modalities": ["text", "image"]},
            "context_length": 256_000,
            "pricing": {"prompt": "0.1", "completion": "0.1"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": models})

    provider = OpenAICompatibleProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        model="openrouter/free",
        fast_model="openrouter/free",
        temperature=0,
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    requirements = RequestRequirements(True, True, True, 10_000)

    first = await provider.resolve_capabilities(requirements)
    second = await provider.resolve_capabilities(requirements)

    assert first.model_id == second.model_id == "example/tool-vision-free"
    assert first.capabilities.vision is True
    assert len(requests) == 1
    await provider._client.aclose()


@pytest.mark.asyncio
async def test_ollama_uses_maintained_profile_when_live_discovery_is_unavailable() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    provider = OpenAICompatibleProvider(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="qwen3.5:9b",
        fast_model="qwen3.5:4b",
        temperature=0,
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))

    match = await provider.resolve_capabilities(RequestRequirements(True, True, True, 5000))

    assert match.capabilities.tools is True
    assert match.capabilities.forced_tool_use is True
    assert match.capabilities.vision is True
    assert match.capabilities.context_tokens == 32_768
    await provider._client.aclose()


@pytest.mark.asyncio
async def test_malformed_json_arguments_are_marked_for_fallback(
    navigate_tool: ToolSpec,
) -> None:
    provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="openai/gpt-oss-120b",
        fast_model="openai/gpt-oss-20b",
        temperature=0,
    )

    async def malformed_response(path: str, payload: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "navigate", "arguments": "{"},
                            }
                        ]
                    }
                }
            ]
        }

    provider._post = malformed_response
    with pytest.raises(LLMError) as raised:
        await provider.complete(
            [Message.user("Open a site")], tools=[navigate_tool], force_tool_use=True
        )

    assert raised.value.fallback is True
    assert raised.value.retryable is False
    await provider._client.aclose()


def test_tool_parse_http_400_is_fallback_without_provider_cooldown() -> None:
    response = httpx.Response(
        400,
        json={"error": {"message": "could not parse tool arguments"}},
        request=httpx.Request("POST", "https://api.example.test/chat/completions"),
    )

    error = _http_error("provider", response, tool_call_context=True)

    assert error.fallback is True
    assert error.retryable is False


def test_non_tool_http_400_remains_non_fallback() -> None:
    response = httpx.Response(
        400,
        json={"error": {"message": "invalid request"}},
        request=httpx.Request("POST", "https://api.example.test/chat/completions"),
    )

    error = _http_error("provider", response)

    assert error.fallback is False
    assert error.retryable is False
