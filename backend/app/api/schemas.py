"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateTaskRequest(BaseModel):
    """Body of `POST /api/tasks`."""

    goal: str = Field(
        min_length=3,
        max_length=2000,
        description="What the agent should do, in plain language.",
    )
    profile: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Persistent browser profile to run in. Profiles keep cookies and "
            "logins between runs; omit to use the configured default."
        ),
    )


class ApprovalRequest(BaseModel):
    """Body of `POST /api/tasks/{id}/approval`."""

    approved: bool
    #: Optional free-text note from the user, recorded in the audit trail.
    note: str = ""


class TaskSummary(BaseModel):
    """A task as it appears in the history list."""

    id: str
    goal: str
    status: str
    profile: str
    success: bool | None = None
    steps_taken: int = 0
    created_at: datetime
    finished_at: datetime | None = None


class TaskDetail(TaskSummary):
    """A task with everything needed to rebuild its page."""

    plan: list[str] = Field(default_factory=list)
    result_summary: str | None = None
    error: str | None = None
    pending_question: str | None = None
    #: True while the run is parked waiting for a human decision.
    awaiting_approval: bool = False
    running: bool = False


class EventOut(BaseModel):
    """One narration event, as sent over REST or WebSocket."""

    id: str
    task_id: str
    type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    step: int | None = None
    created_at: datetime


class ToolCallOut(BaseModel):
    """One row of the audit trail."""

    step: int
    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    result: str
    duration_ms: int
    risk: int
    approved: bool | None
    page_url: str | None
    created_at: datetime


class HealthResponse(BaseModel):
    """Body of `GET /api/health`, used by the UI to explain a degraded setup."""

    status: str
    llm_provider: str
    llm_model: str
    database: bool
    memory: bool
    browser_headless: bool
    vision_mode: str
    active_sessions: int
