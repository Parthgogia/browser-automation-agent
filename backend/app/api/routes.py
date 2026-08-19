"""REST endpoints.

Deliberately thin: create a task, watch it, answer it, cancel it, read history.
Anything that needs to happen *while* a task runs goes over the WebSocket
instead -- see `app.api.ws`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status

from app.agent.runner import AgentRunner, TaskNotRunning
from app.api.schemas import (
    ApprovalRequest,
    CreateTaskRequest,
    EventOut,
    HealthResponse,
    TaskDetail,
    TaskSummary,
    ToolCallOut,
)
from app.db.models import Task, TaskStatus
from app.db.repository import TaskRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tasks"])


def _runner(request: Request) -> AgentRunner:
    return request.app.state.runner


def _repository(request: Request) -> TaskRepository:
    return request.app.state.repository


# --------------------------------------------------------------- lifecycle --


@router.post("/tasks", response_model=TaskDetail, status_code=status.HTTP_201_CREATED)
async def create_task(request: Request, body: CreateTaskRequest) -> TaskDetail:
    """Start a new agent run.

    Returns immediately with the task id; the run proceeds in the background
    and narrates itself over `/ws/tasks/{id}`.
    """
    runner = _runner(request)
    task_id = await runner.start(body.goal.strip(), profile=body.profile)

    task = await _repository(request).get_task(task_id)
    if task is not None:
        return _to_detail(task, runner)

    # The database is unavailable; synthesise a response so the run is still
    # usable from the UI.
    return TaskDetail(
        id=task_id,
        goal=body.goal.strip(),
        status=TaskStatus.RUNNING,
        profile=body.profile or request.app.state.settings.browser_default_profile,
        created_at=datetime.now(UTC),
        running=True,
    )


@router.post("/tasks/{task_id}/approval", response_model=TaskDetail)
async def resolve_approval(
    request: Request, task_id: str, body: ApprovalRequest
) -> TaskDetail:
    """Approve or reject the action a paused task is waiting on."""
    runner = _runner(request)
    try:
        await runner.resume(task_id, body.approved)
    except TaskNotRunning as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    return await _load_detail(request, task_id)


@router.post("/tasks/{task_id}/cancel", response_model=TaskDetail)
async def cancel_task(request: Request, task_id: str) -> TaskDetail:
    """Stop a running task and close its browser."""
    await _runner(request).cancel(task_id)
    return await _load_detail(request, task_id)


# ------------------------------------------------------------------ reads --


@router.get("/tasks", response_model=list[TaskSummary])
async def list_tasks(request: Request, limit: int = 50, offset: int = 0) -> list[TaskSummary]:
    """Task history, newest first."""
    tasks = await _repository(request).list_tasks(limit=min(limit, 200), offset=offset)
    return [TaskSummary.model_validate(task, from_attributes=True) for task in tasks]


@router.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(request: Request, task_id: str) -> TaskDetail:
    """One task, including its plan and result."""
    return await _load_detail(request, task_id)


@router.get("/tasks/{task_id}/events", response_model=list[EventOut])
async def get_events(request: Request, task_id: str) -> list[EventOut]:
    """Replay a task's narration.

    Served from the in-memory bus when the task is recent (which is faster and
    works with no database), falling back to the stored history.
    """
    buffered = request.app.state.events.history(task_id)
    if buffered:
        return [EventOut(**event.model_dump(mode="python")) for event in buffered]

    stored = await _repository(request).list_events(task_id)
    return [
        EventOut(
            id=event.id,
            task_id=event.task_id,
            type=event.type,
            message=event.message,
            data=event.data or {},
            step=event.step,
            created_at=event.created_at,
        )
        for event in stored
    ]


@router.get("/tasks/{task_id}/tool-calls", response_model=list[ToolCallOut])
async def get_tool_calls(request: Request, task_id: str) -> list[ToolCallOut]:
    """The audit trail: every action the agent took, with its risk rating."""
    calls = await _repository(request).list_tool_calls(task_id)
    return [ToolCallOut.model_validate(call, from_attributes=True) for call in calls]


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """What is actually wired up right now.

    The UI shows this so a degraded setup -- mock LLM, no database -- is
    visible rather than mysterious.
    """
    app_state = request.app.state
    settings = app_state.settings
    return HealthResponse(
        status="ok",
        llm_provider=app_state.llm.name,
        llm_model=settings.llm_model,
        database=app_state.database.available,
        memory=app_state.memory is not None and app_state.memory.available,
        browser_headless=settings.browser_headless,
        vision_mode=settings.vision_mode,
        active_sessions=len(app_state.sessions.active_task_ids),
    )


@router.get("/tools", response_model=list[dict])
async def list_tools(request: Request) -> list[dict]:
    """The agent's tool vocabulary, as advertised to the model."""
    return [
        {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
        for spec in request.app.state.tools.specs()
    ]


# ---------------------------------------------------------------- helpers --


async def _load_detail(request: Request, task_id: str) -> TaskDetail:
    task = await _repository(request).get_task(task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No task {task_id}")
    return _to_detail(task, _runner(request))


def _to_detail(task: Task, runner: AgentRunner) -> TaskDetail:
    """Merge stored state with live runner state.

    The database records what the agent last persisted; the runner knows
    whether the coroutine is still alive and whether it is parked on an
    approval. The UI needs both to decide what to render.
    """
    return TaskDetail(
        id=task.id,
        goal=task.goal,
        status=task.status,
        profile=task.profile,
        success=task.success,
        steps_taken=task.steps_taken,
        created_at=task.created_at,
        finished_at=task.finished_at,
        plan=task.plan or [],
        result_summary=task.result_summary,
        error=task.error,
        pending_question=task.pending_question,
        awaiting_approval=runner.is_awaiting_approval(task.id),
        running=runner.is_running(task.id),
    )
