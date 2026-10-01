from __future__ import annotations

import pytest

from app.agent.nodes.finish import make_finish_node
from app.agent.state import initial_state
from app.db.models import TaskStatus
from app.events import EventType
from app.llm.telemetry import record_provider_request


@pytest.mark.asyncio
async def test_provider_requests_are_counted_and_included_in_task_result(runtime) -> None:
    task_id = "llm-usage"
    with runtime.track_llm_requests(task_id):
        record_provider_request("gemini")
        record_provider_request("gemini")
        # Embedding requests use the same instrumentation as completions.
        record_provider_request("ollama")

    state = initial_state(task_id, "test goal", "test")
    state.update(status=TaskStatus.COMPLETED, result="Task done", success=True)
    final = await make_finish_node(runtime)(state)

    assert "LLM API requests: 3" in final["result"]
    assert "- gemini: 2" in final["result"]
    assert "- ollama: 1" in final["result"]
    completed = next(
        event
        for event in runtime.events.history(task_id)
        if event.type is EventType.TASK_COMPLETED
    )
    assert completed.data["llm_api_requests"] == 3
    assert completed.data["llm_providers"] == {"gemini": 2, "ollama": 1}


@pytest.mark.asyncio
async def test_failed_task_result_includes_provider_usage(runtime) -> None:
    task_id = "llm-usage-failed"
    with runtime.track_llm_requests(task_id):
        record_provider_request("groq")

    state = initial_state(task_id, "test goal", "test")
    state.update(status=TaskStatus.FAILED, error="Provider failed", success=False)
    final = await make_finish_node(runtime)(state)

    assert "Provider failed" in final["result"]
    assert "LLM API requests: 1" in final["result"]
    failed = next(
        event
        for event in runtime.events.history(task_id)
        if event.type is EventType.TASK_FAILED
    )
    assert failed.data["llm_providers"] == {"groq": 1}
